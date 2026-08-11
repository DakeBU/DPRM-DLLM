# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import multiprocessing as mp
import os
import random
import safe as sf
import datamol as dm
from contextlib import suppress
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

# https://github.com/datamol-io/safe/blob/main/safe/sample.py
# https://github.com/jensengroup/GB_GA/blob/master/crossover.py
def safe_to_smiles(safe_str, fix=True):
    if fix:
        safe_str = '.'.join([frag for frag in safe_str.split('.')
                             if sf.decode(frag, ignore_errors=True) is not None])
    return sf.decode(safe_str, canonical=True, ignore_errors=True)


def filter_by_substructure(sequences, substruct):
    substruct = sf.utils.standardize_attach(substruct)
    substruct = Chem.DeleteSubstructs(Chem.MolFromSmarts(substruct), Chem.MolFromSmiles('*'))
    substruct = Chem.MolFromSmarts(Chem.MolToSmiles(substruct))
    return sf.utils.filter_by_substructure_constraints(sequences, substruct)


def mix_sequences(prefix_sequences, suffix_sequences, prefix, suffix, num_samples=1):
    mol_linker_slicer = sf.utils.MolSlicer(require_ring_system=False)

    prefix_linkers = []
    suffix_linkers = []
    prefix_query = dm.from_smarts(prefix)
    suffix_query = dm.from_smarts(suffix)

    for x in prefix_sequences:
        with suppress(Exception):
            x = dm.to_mol(x)
            out = mol_linker_slicer(x, prefix_query)
            prefix_linkers.append(out[1])

    for x in suffix_sequences:
        with suppress(Exception):
            x = dm.to_mol(x)
            out = mol_linker_slicer(x, suffix_query)
            suffix_linkers.append(out[1])

    n_linked = 0
    linked = []
    linkers = prefix_linkers + suffix_linkers
    linkers = [x for x in linkers if x is not None]
    for n_linked, linker in enumerate(linkers):
        linked.extend(mol_linker_slicer.link_fragments(linker, prefix, suffix))
        if n_linked > num_samples:
            break
        linked = [x for x in linked if x]
    return linked[:num_samples]


def _mix_and_filter_worker(queue, prefix_samples, suffix_samples, prefix_fragment, suffix_fragment, num_samples, fragment):
    try:
        mixed = mix_sequences(prefix_samples, suffix_samples, prefix_fragment, suffix_fragment, num_samples)
        queue.put(("ok", filter_by_substructure(mixed, fragment)))
    except BaseException as exc:
        queue.put(("error", repr(exc)))


def safe_mix_and_filter(prefix_samples, suffix_samples, fragment, num_samples):
    """Isolate unsafe SAFE/RDKit linker mixing so C-level crashes do not kill eval."""
    timeout = int(os.environ.get("GENMOL_LINKER_MIX_TIMEOUT", "20"))
    prefix_fragment, suffix_fragment = fragment.split(".")
    ctx = mp.get_context(os.environ.get("GENMOL_LINKER_MP_CONTEXT", "spawn"))
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_mix_and_filter_worker,
        args=(queue, prefix_samples, suffix_samples, prefix_fragment, suffix_fragment, num_samples, fragment),
    )
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        print("[genmol] linker mix/filter worker timed out; counting chunk as invalid", flush=True)
        return []
    if proc.exitcode != 0:
        print(
            f"[genmol] linker mix/filter worker exited with code {proc.exitcode}; counting chunk as invalid",
            flush=True,
        )
        return []
    if queue.empty():
        print("[genmol] linker mix/filter worker returned no result; counting chunk as invalid", flush=True)
        return []
    status, payload = queue.get()
    if status != "ok":
        print(f"[genmol] linker mix/filter worker error: {payload}; counting chunk as invalid", flush=True)
        return []
    return payload
    

def cut(smiles):
    def cut_nonring(mol):
        if not mol.HasSubstructMatch(Chem.MolFromSmarts('[*]-;!@[*]')):
            return None

        bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[*]-;!@[*]')))  # single bond not in ring
        bs = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]
        fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1)])

        try:
            return Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
        except ValueError:
            return None
        
    mol = Chem.MolFromSmiles(smiles)
    frags = set()
    # non-ring cut
    for _ in range(3):
        frags_nonring = cut_nonring(mol)
        if frags_nonring is not None:
            frags |= set([Chem.MolToSmiles(f) for f in frags_nonring])
    return frags


class Slicer:
    def __call__(self, mol):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        
        # non-ring single bonds
        bonds = mol.GetSubstructMatches(Chem.MolFromSmarts('[*]-;!@[*]'))
        for bond in bonds:
            yield bond
