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
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import warnings
warnings.filterwarnings('ignore')

import itertools
import pickle
import torch
import torch.nn.functional as F
import random
import safe as sf
from rdkit import Chem
from genmol.utils.utils_chem import safe_to_smiles, filter_by_substructure, safe_mix_and_filter, Slicer
from genmol.utils.bracket_safe_converter import BracketSAFEConverter, bracketsafe2safe
from genmol.model import GenMol

try:
    from dprm import HostDPRMBatch, OnlineDPRMController
except Exception:
    HostDPRMBatch = None
    OnlineDPRMController = None


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def load_model_from_path(path):
    model = GenMol.load_from_checkpoint(path)
    model.backbone.eval()
    if model.ema:
        model.ema.store(itertools.chain(model.backbone.parameters()))
        model.ema.copy_to(itertools.chain(model.backbone.parameters()))
    return model


class Sampler:
    def __init__(self, path):
        self.model = load_model_from_path(path)
        self.slicer = Slicer()
        self.dot_index = self.model.tokenizer('.')['input_ids'][1]
        self.pad_index = self.model.tokenizer.pad_token_id
        self.mdlm = self.model.mdlm
        self.mdlm.to_device(self.model.device)
        if getattr(self.model, 'dprm_controller', None) is not None:
            self.model.dprm_controller.to(self.model.device)
        self._ordering_diagnostics = {
            'decision_rows': 0,
            'order_changed_rows': 0,
            'candidate_events': 0,
            'score_changed_candidates': 0,
            'selected_events': 0,
            'selected_ready': 0,
            'selected_full_ready': 0,
            'selected_abs_score_delta_sum': 0.0,
        }

    def ordering_diagnostics(self):
        stats = dict(self._ordering_diagnostics)
        decisions = max(int(stats['decision_rows']), 1)
        candidates = max(int(stats['candidate_events']), 1)
        selected = max(int(stats['selected_events']), 1)
        stats.update({
            'order_changed_row_rate': stats['order_changed_rows'] / decisions,
            'score_changed_candidate_rate': stats['score_changed_candidates'] / candidates,
            'selected_ready_rate': stats['selected_ready'] / selected,
            'selected_full_ready_rate': stats['selected_full_ready'] / selected,
            'selected_mean_abs_score_delta': stats['selected_abs_score_delta_sum'] / selected,
        })
        return stats

    def _topk_mask(self, scores, candidate_mask, k, largest=True):
        picked = torch.zeros_like(candidate_mask, dtype=torch.bool)
        fill = float('-inf') if largest else float('inf')
        scores = scores.masked_fill(~candidate_mask, fill)
        for row in range(candidate_mask.shape[0]):
            active = torch.nonzero(candidate_mask[row], as_tuple=False).squeeze(1)
            if active.numel() == 0:
                continue
            kk = min(int(k[row].item()), int(active.numel()))
            if kk <= 0:
                continue
            chosen = active[torch.topk(scores[row, active], k=kk, largest=largest).indices]
            picked[row, chosen] = True
        return picked

    @torch.no_grad()
    def _step_ordered(self, logits, xt, curr_step, num_steps, softmax_temp, randomness, policy):
        log_p_x0 = self.mdlm._subs_parameterization(logits, xt)
        probs = torch.softmax(log_p_x0 / max(softmax_temp, 1e-6), dim=-1)
        preds = torch.distributions.Categorical(probs=probs).sample()
        confidence = probs.gather(-1, preds.unsqueeze(-1)).squeeze(-1).clamp(1e-6, 1.0)
        mask = xt == self.model.mask_index
        remaining_steps = max(num_steps - curr_step, 1)
        k = torch.ceil(mask.sum(dim=1).float() / remaining_steps).long().clamp_min(1)

        dprm_summary = None
        if policy in {'random_ordered'}:
            scores = torch.rand_like(confidence)
        elif policy in {'dprm', 'dprm_confidence', 'dprm_random'} and getattr(self.model, 'dprm_controller', None) is not None:
            phase = OnlineDPRMController.phase_from_progress(
                curr_step,
                num_steps,
                self.model.dprm_controller.cfg.num_phases,
                xt.shape[0],
                xt.device,
            )
            host = HostDPRMBatch(
                confidence=confidence,
                candidate_mask=mask,
                phase_ids=phase,
                global_step=10**12,
                force_full_dprm=True,
            )
            dprm_summary = self.model.dprm_controller.summarize(host)
            scores = dprm_summary.score
        else:
            ratio = curr_step / max(num_steps - 1, 1)
            gumbel = -torch.log(-torch.log(torch.rand_like(confidence).clamp_min(1e-6)))
            scores = torch.log(confidence) + gumbel * randomness * (1 - ratio)

        reveal = self._topk_mask(scores, mask, k, largest=True)
        if dprm_summary is not None:
            confidence_reveal = self._topk_mask(torch.log(confidence), mask, k, largest=True)
            selected_gate = dprm_summary.gate[reveal]
            selected_delta = (dprm_summary.score - dprm_summary.base_score)[reveal]
            stats = self._ordering_diagnostics
            stats['decision_rows'] += int(xt.shape[0])
            stats['order_changed_rows'] += int((reveal != confidence_reveal).any(dim=1).sum().item())
            stats['candidate_events'] += int(mask.sum().item())
            stats['score_changed_candidates'] += int(
                ((dprm_summary.score - dprm_summary.base_score).abs() > 1e-8)[mask].sum().item()
            )
            stats['selected_events'] += int(reveal.sum().item())
            stats['selected_ready'] += int((selected_gate > 0).sum().item())
            stats['selected_full_ready'] += int((selected_gate >= 1.0 - 1e-8).sum().item())
            stats['selected_abs_score_delta_sum'] += float(selected_delta.abs().sum().item())
        x_new = xt.clone()
        x_new[reveal] = preds[reveal]
        return x_new
        
    @torch.no_grad()
    def generate(self, x, softmax_temp=1.2, randomness=2, fix=True, gamma=0, w=2, **kwargs):
        x = x.to(self.model.device)
        num_steps = max(self.mdlm.get_num_steps_confidence(x), 2)
        attention_mask = x != self.pad_index
        
        for i in range(num_steps):
            logits = self.model(x, attention_mask)

            if gamma and w:
                x_poor = x.clone()
                context_tokens = (x_poor[0] != self.model.bos_index).to(int) * \
                    (x_poor[0] != self.model.eos_index).to(int) * \
                    (x_poor[0] != self.model.mask_index).to(int) * \
                    (x_poor[0] != self.pad_index).to(int)
                context_token_ids = context_tokens.nonzero(as_tuple=True)[0].tolist()
                # mask 100 * gamma % of the context (given fragments) tokens
                num_mask_poor = int(context_tokens.sum() * gamma)
                mask_idx_poor = random.sample(context_token_ids, num_mask_poor)
                x_poor[:, mask_idx_poor] = self.model.mask_index
                logits_poor = self.model(x_poor, attention_mask=attention_mask)
                logits = w * logits + (1 - w) * logits_poor

            order_policy = kwargs.get('order_policy', self.model.config.training.get('order_policy', 'confidence'))
            if order_policy in {'random_ordered', 'dprm', 'dprm_confidence', 'dprm_random'}:
                x = self._step_ordered(logits, x, i, num_steps, softmax_temp, randomness, order_policy)
            else:
                x = self.mdlm.step_confidence(logits, x, i, num_steps, softmax_temp, randomness)
            
        # decode to SAFE strings
        samples = self.model.tokenizer.batch_decode(x, skip_special_tokens=True)
        # convert to SMILES strings
        if self.model.config.training.get('use_bracket_safe'):
            samples = [safe_to_smiles(bracketsafe2safe(s), fix=fix) for s in samples]
        else:
            samples = [safe_to_smiles(s, fix=fix) for s in samples]
        # remove None and take the largest
        samples = [sorted(s.split('.'), key=len)[-1] for s in samples if s]
        return samples

    def _insert_mask(self, x, num_samples, min_add_len=18, **kwargs):
        with open(os.path.join(ROOT_DIR, 'data/len.pk'), 'rb') as f:
            seq_len_list = pickle.load(f)
        
        x = x[0]
        x_new = []
        for _ in range(num_samples):
            add_seq_len = max(random.choice(seq_len_list) - len(x), min_add_len)
            x_new.append(torch.hstack([x[:-1],
                                      torch.full((add_seq_len,), self.model.mask_index),
                                      x[-1:]]))
        pad_len = max([len(xx) for xx in x_new])
        x_new = [torch.hstack([xx,torch.full((pad_len - len(xx),), self.pad_index)]) for xx in x_new]
        return torch.stack(x_new)
    
    @torch.no_grad()
    def de_novo_generation(self, num_samples=1, softmax_temp=0.8, randomness=0.5, min_add_len=40, **kwargs):
        # Prepare Fully Masked Inputs
        x = torch.hstack([torch.full((1, 1), self.model.bos_index),
                          torch.full((1, 1), self.model.eos_index)])
        x = self._insert_mask(x, num_samples, min_add_len=min_add_len)
        x = x.to(self.model.device)
        return self.generate(x, softmax_temp, randomness, **kwargs)
    
    def fragment_linking_onestep(self, fragment, num_samples=1, softmax_temp=1.2, randomness=2, gamma=0, min_add_len=30, **kwargs):
        if self.model.config.training.get('use_bracket_safe'):
            encoded_fragment = BracketSAFEConverter(slicer=None).encoder(fragment, allow_empty=True)
        else:
            encoded_fragment = sf.SAFEConverter(slicer=None).encoder(fragment, allow_empty=True)
        
        x = self.model.tokenizer([encoded_fragment + '.'],
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=self.model.config.model.max_position_embeddings)['input_ids']
        x = self._insert_mask(x, num_samples, min_add_len=min_add_len)
        samples = self.generate(x, softmax_temp, randomness, gamma=gamma, **kwargs)
        samples = filter_by_substructure(samples, fragment)
        return samples
    
    def fragment_linking(self, fragment, num_samples=1, softmax_temp=1.2, randomness=2, gamma=0, min_add_len=30, **kwargs):
        encoded_fragment = sf.SAFEConverter(slicer=None).encoder(fragment, allow_empty=True)
        prefix, suffix = encoded_fragment.split('.')

        x = self.model.tokenizer([prefix + '.'],
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=self.model.config.model.max_position_embeddings)['input_ids']
        x = self._insert_mask(x, num_samples, min_add_len=min_add_len)
        prefix_samples = self.generate(x, softmax_temp, randomness, gamma=gamma, **kwargs)

        x = self.model.tokenizer([suffix + '.'],
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=self.model.config.model.max_position_embeddings)['input_ids']
        x = self._insert_mask(x, num_samples, min_add_len=min_add_len)
        suffix_samples = self.generate(x, softmax_temp, randomness, gamma=gamma, **kwargs)
        
        return safe_mix_and_filter(prefix_samples, suffix_samples, fragment, num_samples)
        
    def fragment_completion(self, fragment, num_samples=1, apply_filter=True, softmax_temp=1.2, randomness=2, gamma=0, **kwargs):
        if '*' not in fragment:     # superstructure generation
            cores = sf.utils.list_individual_attach_points(Chem.MolFromSmiles(fragment), depth=3)
            fragment = random.choice(cores)
            
        encoded_fragment = sf.SAFEConverter(ignore_stereo=True).encoder(fragment, allow_empty=True) + '.'
        x = self.model.tokenizer([encoded_fragment],
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=self.model.config.model.max_position_embeddings)['input_ids']
        x = self._insert_mask(x, num_samples)
        samples = self.generate(x, softmax_temp, randomness, gamma=gamma, **kwargs)

        if apply_filter:
            return filter_by_substructure(samples, fragment)
        return samples

    def mask_modification(self, smiles, min_len=30, **kwargs):
        encoded_smiles = sf.SAFEConverter(slicer=self.slicer, ignore_stereo=True).encoder(smiles, allow_empty=True)
        x = self.model.tokenizer([encoded_smiles],
                                  return_tensors='pt',
                                  truncation=True,
                                  max_length=self.model.config.model.max_position_embeddings)['input_ids']
        if x.shape[-1] < min_len:
            return self.addmask(smiles, num_edit=min_len-x.shape[-1]+1, **kwargs)
        return self.remask(smiles, input_ids=x, **kwargs)

    def addmask(self, smiles, num_edit=3, **kwargs):
        try:
            samples = self.fragment_completion(smiles, mask_len=num_edit, apply_filter=False, **kwargs)
        except:
            return smiles
        if samples:
            return samples[0]
        return smiles
    
    def remask(self, smiles, input_ids=None, **kwargs):
        x = input_ids
        if x is None:
            encoded_smiles = sf.SAFEConverter(slicer=self.slicer, ignore_stereo=True).encoder(smiles, allow_empty=True)
            x = self.model.tokenizer([encoded_smiles],
                                     return_tensors='pt',
                                     truncation=True,
                                     max_length=self.model.config.model.max_position_embeddings)['input_ids']
        
        # fragment mask replacement
        special_token_idx = [0] + (x[0] == self.dot_index).nonzero(as_tuple=True)[0].tolist() + [len(x[0]) - 1]
        frag_idx = random.randint(0, len(special_token_idx) - 2)
        mask_start_idx = special_token_idx[frag_idx] + 1
        mask_end_idx = special_token_idx[frag_idx + 1]
        num_insert_mask = random.randint(5, 15)
        num_insert_mask = min(num_insert_mask,
                              self.model.config.model.max_position_embeddings - x.shape[-1] + mask_end_idx - mask_start_idx)
        x = torch.hstack([x[:, :mask_start_idx],
                          torch.full((1, num_insert_mask), self.model.mask_index),
                          x[:, mask_end_idx:]])
        samples = self.generate(x, **kwargs)
        if samples:
            return samples[0]
        return smiles
