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


import os
import sys
import torch
import datasets
import torch
from safe.tokenizer import SAFETokenizer
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import QED
from genmol.utils.bracket_safe_converter import safe2bracketsafe
from genmol.utils.utils_chem import safe_to_smiles
RDLogger.DisableLog('rdApp.*')

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer


ROOT_DIR = os.getcwd()


def get_last_checkpoint(save_dir):
    if os.path.exists(save_dir):
        filenames = os.listdir(save_dir)
        if filenames:
            last_filename = sorted(filenames, key=lambda x: int(x[:-5]))[-1]
            return os.path.join(save_dir, last_filename)
    

def get_tokenizer():
    tk = SAFETokenizer.from_pretrained('datamol-io/safe-gpt').get_pretrained()
    tk.add_tokens(['<', '>'])   # for bracket_safe
    return tk


class Collator:
    def __init__(self, config):
        self.tokenizer = get_tokenizer()
        self.max_length = config.model.max_position_embeddings
        self.use_bracket_safe = config.training.get('use_bracket_safe')
        self.dprm_reward_mode = str(
            config.training.get('dprm_reward_mode', 'selected_confidence')
        )

    @staticmethod
    def _molecular_benefits(safe_string):
        smiles = safe_to_smiles(safe_string, fix=False)
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return [0.0, 0.0]
        qed = float(QED.qed(mol))
        sa_benefit = float((10.0 - sascorer.calculateScore(mol)) / 9.0)
        return [max(0.0, min(1.0, qed)), max(0.0, min(1.0, sa_benefit))]
    
    def __call__(self, examples):
        for example in examples:
            if 'input' not in example:
                example['input'] = example.get('safe', example.get('text', example.get('smiles')))
            if example['input'] is None:
                raise KeyError("Expected one of input/safe/text/smiles in dataset example.")
        terminal_objectives = None
        if self.dprm_reward_mode in {'molecular_weighted_sum', 'molecular_tchebycheff'}:
            terminal_objectives = torch.tensor(
                [self._molecular_benefits(example['input']) for example in examples],
                dtype=torch.float32,
            )
        if self.use_bracket_safe:
            for example in examples: example['input'] = safe2bracketsafe(example['input'])

        batch = self.tokenizer([example['input'] for example in examples],
                               return_tensors='pt',
                               padding=True,
                               truncation=True,
                               max_length=self.max_length)
        del batch['token_type_ids']
        if terminal_objectives is not None:
            batch['terminal_objectives'] = terminal_objectives
        return batch
    

class UserDataset(datasets.Dataset):
    def __init__(self, data_path):
        with open(data_path) as f:
            self.safe_list = f.readlines()
        self.safe_list = [s.strip('\n') for s in self.safe_list]
        
    def __len__(self):
        return len(self.safe_list)

    def __getitem__(self, indices):
        return {'input': self.safe_list[i] for i in indices}
    

def get_dataloader(config):
    if config.data == 'safe':
        return torch.utils.data.DataLoader(
            datasets.load_dataset('datamol-io/safe-gpt', streaming=True, split='train'),
            batch_size=config.loader.batch_size,
            collate_fn=Collator(config),
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=False,  # streaming
            persistent_workers=True)

    # User-defined dataset
    return torch.utils.data.DataLoader(
        UserDataset(config.data),
        batch_size=config.loader.batch_size,
        collate_fn=Collator(config),
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
        shuffle=True,
        persistent_workers=True)
