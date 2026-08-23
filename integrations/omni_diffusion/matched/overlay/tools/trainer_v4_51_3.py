# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import contextlib
import copy
import codecs
import functools
import glob
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union


# Integrations must be imported before ML frameworks:
# isort: off
from transformers.integrations import (
    get_reporting_integration_callbacks,
)

# isort: on

import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import ModelCard, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler

_DPRM_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_DPRM_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_DPRM_REPO_ROOT / "src"))

from dprm.omni_order import (
    OMNI_SINGLE_PATH_REPEAT_PENALTY,
    OmniOrderScorer,
    OmniBucketTableDPRM,
    OmniRankBucketDPRM,
    OmniStageRankCodeDPRM,
    OmniStageRankSpatialDPRM,
    load_omni_order_controller,
    adjusted_order_scores,
    build_action_features,
    entropy_penalty_order_scores,
    visual_token_runs,
)

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.optimization import Adafactor, get_scheduler
from transformers.processing_utils import ProcessorMixin
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    is_torch_greater_or_equal_than_2_3,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ExportableState,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    EvalLoopContainer,
    IterableDatasetShard,
    LabelSmoother,
    LayerWiseDummyOptimizer,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    SaveStrategy,
    TrainerMemoryTracker,
    TrainOutput,
    check_target_module_exists,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    XLA_FSDPV2_MIN_VERSION,
    PushInProgress,
    PushToHubMixin,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_galore_torch_available,
    is_grokadamw_available,
    is_in_notebook,
    is_ipex_available,
    is_liger_kernel_available,
    is_lomo_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_schedulefree_available,
    is_torch_compile_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    is_torchao_available,
    logging,
    strtobool,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.quantization_config import QuantizationMethod

try:
    from transformers.utils import is_torch_hpu_available
except ImportError:
    def is_torch_hpu_available():
        return False


DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_apex_available():
    from apex import amp

if is_datasets_available():
    import datasets

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
else:
    IS_XLA_FSDPV2_POST_2_2 = False


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_safetensors_available():
    import safetensors.torch

if is_peft_available():
    from peft import PeftModel


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.state import AcceleratorState
    from accelerate.utils import (
        AutocastKwargs,
        DistributedDataParallelKwargs,
        DistributedType,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("1.3.0"):
        from accelerate.utils import TorchTensorParallelPlugin
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

if is_accelerate_available("0.28.0"):
    from accelerate.utils import DataLoaderConfiguration


def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False


def _get_fsdp_ckpt_kwargs():
    # TODO: @AjayP13, @younesbelkada replace this check with version check at the next `accelerate` release
    if is_accelerate_available() and "adapter_only" in list(inspect.signature(save_fsdp_model).parameters):
        return {"adapter_only": True}
    else:
        return {}


def safe_globals():
    # Starting from version 2.4 PyTorch introduces a check for the objects loaded
    # with torch.load(weights_only=True). Starting from 2.6 weights_only=True becomes
    # a default and requires allowlisting of objects being loaded.
    # See: https://github.com/pytorch/pytorch/pull/137602
    # See: https://pytorch.org/docs/stable/notes/serialization.html#torch.serialization.add_safe_globals
    # See: https://github.com/huggingface/accelerate/pull/3036
    np_core = np._core if version.parse(np.__version__) >= version.parse("2.0.0") else np.core
    allowlist = [np_core.multiarray._reconstruct, np.ndarray, np.dtype, codecs.encode]
    # numpy >1.25 defines numpy.dtypes.UInt32DType, but below works for
    # all versions of numpy
    allowlist += [type(np.dtype(np.uint32))]

    if hasattr(torch.serialization, "safe_globals"):
        return torch.serialization.safe_globals(allowlist)
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals(allowlist)
    return contextlib.nullcontext()


def get_t2i_prompt():
    return [
        "Generate an image based on the provided text description.",
        "Produce an image that matches the details in the given text.",
        "Generate a picture that aligns with the provided text.",
        "Create an artwork from the given text prompt.",
        "Translate the given description into an image.",
        "Produce a high-quality image from the provided description.",
        "Synthesize an image that matches the provided text description.",
        "Generate a picture that captures the essence of the text.",
    ]


def get_task_from_inputs(input_text):
    t2i_prompt = get_t2i_prompt()
    for p in t2i_prompt:
        if p in input_text: return "loss_t2i"

    if "Convert the speech to text." in input_text:
        return "loss_asr"
    elif "Convert the text to speech." in input_text:
        return "loss_tts"
    elif "Please response the input audio." in input_text:
        return "loss_sqa"
    elif "Please generate an image based on the input audio." in input_text:
        return "loss_s2i"
    elif "Please response the input audio based on the given image." in input_text:
        return "loss_svqa"
    elif "<|image" not in input_text and "<|audio" not in input_text:
        return "loss_tqa"
    elif "<|image" in input_text and "<|audio" not in input_text:
        return "loss_vqa"

if TYPE_CHECKING:
    import optuna

    if is_datasets_available():
        import datasets

logger = logging.get_logger(__name__)
logger.setLevel("INFO")


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCALER_NAME = "scaler.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


DATA_PRINT_KEYS = []
def print_batch(batch, tokenizer, args):

    global DATA_PRINT_KEYS

    if batch is None:
        return

    if batch.keys() in DATA_PRINT_KEYS:
        return

    DATA_PRINT_KEYS.append(batch.keys())

    global_rank = torch.distributed.get_rank()
    f = open(os.path.join(args.output_dir, f"print_batch_{global_rank}.log"), "a")
    print("=" * 100, file=f)

    torch.set_printoptions(threshold=100_000)

    if "loss_mask" in batch and batch["loss_mask"] is not None:
        loss_mask = batch["loss_mask"]
        print(f"loss_mask {loss_mask} {loss_mask.size()}", file=f)

    if "position_ids" in batch and batch["position_ids"] is not None:
        position_ids = batch["position_ids"]
        print(f"position_ids {position_ids} {position_ids.size()}", file=f)

    if "attention_mask" in batch and batch["attention_mask"] is not None:
        attention_mask = batch["attention_mask"]
        if isinstance(attention_mask, list):
            attention_mask = attention_mask[0]
        print(f"attention_mask {attention_mask} {attention_mask.size()}", file=f)

    if "input_ids" in batch and batch["input_ids"] is not None:
        tokens = batch["input_ids"]
        print(f"tokens {tokens} {tokens.size()}", file=f)

        tokens_ = tokens.cpu().clone().detach()
        tokens_ = tokenizer.batch_decode(tokens_.tolist(), skip_special_tokens=False)
        print(f"tokens_ {tokens_[:]}", file=f)

    if "labels" in batch and batch["labels"] is not None:
        labels = batch["labels"]
        print(f"labels {labels} {labels.size()}", file=f)

        labels_ = labels.cpu().clone().detach()
        labels_[labels_==-100] = tokenizer("-", add_special_tokens=False).input_ids[0]
        labels_ = tokenizer.batch_decode(labels_.tolist(), skip_special_tokens=False)
        print(f"labels {labels_}", file=f)

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k} {v} {v.size()}", file=f)
        else:
            print(f"{k} {v}", file=f)

    f.close()




from transformers import Trainer as HFTrainer
class Trainer(HFTrainer):
    def _dprm_omni_policy(self, device: torch.device):
        policy = os.environ.get("DPRM_TRAIN_ORDER_POLICY", "random").lower()
        if policy not in {"random_matched", "confidence_matched", "dprm_matched"}:
            return policy, None
        if policy in {"random_matched", "confidence_matched"}:
            return policy, None
        if not hasattr(self, "_dprm_omni_order_scorer"):
            artifact = os.environ.get("DPRM_OMNI_DPRM_SCORER", "")
            if not artifact:
                raise RuntimeError("dprm_matched requires DPRM_OMNI_DPRM_SCORER")
            if artifact.endswith(".json"):
                scorer, metadata = load_omni_order_controller(artifact)
            else:
                scorer, metadata = OmniOrderScorer.load_artifact(artifact, map_location=device)
                scorer.to(device).eval().requires_grad_(False)
            self._dprm_omni_order_scorer = scorer
            self._dprm_omni_order_metadata = metadata
            logger.info("Loaded train-test-matched Omni DPRM scorer from %s", artifact)
        return policy, self._dprm_omni_order_scorer

    def _dprm_policy_logits(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        state: torch.Tensor,
    ) -> torch.Tensor:
        # Call the distributed wrapper so ZeRO/FSDP can gather partitioned
        # parameters correctly during the no-gradient policy pass.
        policy_model = model
        attention_mask = inputs["attention_mask"]
        position_ids = inputs["position_ids"]
        was_training = policy_model.training
        policy_model.eval()
        try:
            with torch.no_grad():
                outputs = policy_model(
                    input_ids=state,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    labels=None,
                    use_cache=False,
                    return_dict=True,
                )
            logits = outputs.logits
            # This is the same alignment used by DreamGenerationMixin before
            # entropy-penalty confidence is computed.
            return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        finally:
            policy_model.train(was_training)

    def _dprm_apply_matched_omni_order(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Construct teacher-forced states with the same order score used at inference."""
        policy, scorer = self._dprm_omni_policy(inputs["input_ids"].device)
        if policy not in {"random_matched", "confidence_matched", "dprm_matched"}:
            return inputs
        trajectory_mode = os.environ.get("DPRM_OMNI_PRECOMPUTED_TRAJECTORY", "0")
        if trajectory_mode == "1":
            # The dataset already contains exact teacher-forced canvases rolled
            # out by the deployed order policy. Re-ranking them here would
            # create a different training distribution.
            return inputs

        mask_id = 151666
        # Recover clean targets, then resume each example's cached teacher-forced
        # trajectory. The cache makes every visited canvas a state induced by
        # the deployed order policy rather than by independent random masking.
        state = inputs["input_ids"].clone()
        clean = state.clone()
        supervised = inputs["labels"].ne(-100)
        clean[supervised] = inputs["labels"][supervised]
        labels = torch.full_like(inputs["labels"], -100)
        guidance = float(os.environ.get("DPRM_OMNI_DPRM_GUIDANCE", "1.0"))
        total_steps = int(os.environ.get("DPRM_OMNI_DPRM_TOTAL_STEPS", "260"))
        reveal_budget = max(int(os.environ.get("DPRM_OMNI_MATCHED_REVEAL_BUDGET", "1")), 1)
        active_steps = {
            int(value)
            for value in os.environ.get("DPRM_OMNI_DPRM_ACTION_STEPS", "").split()
            if value
        }
        if not hasattr(self, "_dprm_omni_state_cache"):
            self._dprm_omni_state_cache = OrderedDict()
        if not hasattr(self, "_dprm_omni_rollin_stats"):
            self._dprm_omni_rollin_stats = {
                "actions": 0,
                "restarts": 0,
                "progress_sum": 0.0,
                "progress_min": 1.0,
                "progress_max": 0.0,
                "early": 0,
                "middle": 0,
                "late": 0,
                "candidate_sum": 0,
                "selected_sum": 0,
                "dprm_scorer_actions": 0,
                "dprm_direct_overrides": 0,
            }
        cache_limit = max(int(os.environ.get("DPRM_OMNI_MATCHED_CACHE_SIZE", "8192")), 1)
        run_records = []

        for batch_idx in range(clean.shape[0]):
            for run in visual_token_runs(clean[batch_idx]):
                if run.numel() < 2:
                    continue
                run = run[:256]
                clean_visual = clean[batch_idx, run]
                cache_key = hashlib.sha1(
                    clean_visual.detach().cpu().numpy().tobytes()
                ).hexdigest()
                if trajectory_mode != "hybrid":
                    cached = self._dprm_omni_state_cache.pop(cache_key, None)
                    if cached is None or cached.numel() != run.numel():
                        cached = torch.full_like(clean_visual, mask_id, device="cpu")
                    state[batch_idx, run] = cached.to(clean.device)
                run_records.append((batch_idx, run, cache_key))

        if not run_records:
            raise RuntimeError("matched Omni roll-in found no visual target sequence")
        logits = self._dprm_policy_logits(model, inputs, state)

        for batch_idx, run, cache_key in run_records:
                run_masked = state[batch_idx, run].eq(mask_id)
                candidates_local = torch.where(run_masked)[0]
                if candidates_local.numel() <= reveal_budget:
                    # Restart before taking the transition so the post-action
                    # state always retains at least one supervised mask.
                    state[batch_idx, run] = mask_id
                    run_masked = torch.ones_like(run, dtype=torch.bool)
                    candidates_local = torch.arange(run.numel(), device=clean.device)
                    logits = self._dprm_policy_logits(model, inputs, state)
                    self._dprm_omni_rollin_stats["restarts"] += 1
                candidate_positions = run[candidates_local]
                candidate_logits = logits[batch_idx, candidate_positions]
                confidence, provisional_local = entropy_penalty_order_scores(
                    candidate_logits,
                    top_p=0.9,
                    # The deployed single-path Omni configuration has
                    # return_dict_in_generate=False. Upstream therefore does
                    # not retain `histories`, and entropy-penalty calls
                    # sample_tokens with repeat_penalty=1.0 at every action.
                    repeat_penalty=OMNI_SINGLE_PATH_REPEAT_PENALTY,
                    max_position_penalty=2.0,
                    past_tokens=state[batch_idx],
                    mask_id=mask_id,
                )
                revealed = int(run.numel() - candidates_local.numel())
                progress = revealed / max(int(run.numel()) - 1, 1)
                rollin_stats = self._dprm_omni_rollin_stats
                rollin_stats["actions"] += 1
                rollin_stats["candidate_sum"] += int(candidates_local.numel())
                rollin_stats["progress_sum"] += progress
                rollin_stats["progress_min"] = min(rollin_stats["progress_min"], progress)
                rollin_stats["progress_max"] = max(rollin_stats["progress_max"], progress)
                if progress < 1.0 / 3.0:
                    rollin_stats["early"] += 1
                elif progress < 2.0 / 3.0:
                    rollin_stats["middle"] += 1
                else:
                    rollin_stats["late"] += 1
                # Cached canvases are post-action states. With one visual
                # reveal per deployed action, r visible codes are followed by
                # global decode step r.
                step = min(revealed, total_steps - 1)
                scores = torch.rand_like(confidence) if policy == "random_matched" else confidence
                scorer_active = False
                if isinstance(scorer, OmniRankBucketDPRM):
                    scores, _ = scorer.score(confidence, step=step)
                    scorer_active = step == int(scorer.active_step)
                elif isinstance(scorer, OmniStageRankCodeDPRM):
                    scores, _ = scorer.score(
                        confidence,
                        step=step,
                        provisional_token_ids=provisional_local,
                    )
                    scorer_active = step in set(scorer.active_steps)
                elif isinstance(scorer, (OmniStageRankSpatialDPRM, OmniBucketTableDPRM)):
                    scores, _ = scorer.score(
                        confidence,
                        step=step,
                        visual_indices=candidates_local,
                    )
                    if isinstance(scorer, OmniStageRankSpatialDPRM):
                        scorer_active = step in set(scorer.active_steps)
                    else:
                        scorer_active = (
                            not scorer.reward_action_steps
                            or step in set(scorer.reward_action_steps)
                        )
                elif scorer is not None and (not active_steps or step in active_steps):
                    local_candidates = torch.arange(
                        candidates_local.numel(), device=clean.device
                    )
                    features = build_action_features(
                        confidence=confidence,
                        candidate_indices=local_candidates,
                        visual_indices=candidates_local,
                        masked_visual=run_masked,
                        provisional_token_ids=provisional_local,
                        step=step,
                        config=scorer.config,
                    )
                    scores, _ = adjusted_order_scores(
                        confidence=confidence,
                        candidate_indices=local_candidates,
                        features=features,
                        scorer=scorer,
                        guidance_scale=guidance,
                    )
                    scorer_active = True
                take = min(reveal_budget, int(candidates_local.numel()) - 1)
                selected_local = candidates_local[torch.topk(scores, take).indices]
                if scorer_active:
                    rollin_stats["dprm_scorer_actions"] += 1
                    confidence_selected = candidates_local[
                        torch.topk(confidence, take).indices
                    ]
                    rollin_stats["dprm_direct_overrides"] += int(
                        set(selected_local.detach().cpu().tolist())
                        != set(confidence_selected.detach().cpu().tolist())
                    )
                selected_positions = run[selected_local]
                before_mask_count = int(run_masked.sum().item())
                state[batch_idx, selected_positions] = clean[batch_idx, selected_positions]
                still_masked = state[batch_idx, run].eq(mask_id)
                after_mask_count = int(still_masked.sum().item())
                if after_mask_count != before_mask_count - take:
                    raise RuntimeError(
                        "matched Omni transition did not reveal exactly the selected positions: "
                        f"before={before_mask_count}, selected={take}, after={after_mask_count}"
                    )
                rollin_stats["selected_sum"] += take
                labels[batch_idx, run[still_masked]] = clean[batch_idx, run[still_masked]]
                if trajectory_mode != "hybrid":
                    self._dprm_omni_state_cache[cache_key] = (
                        state[batch_idx, run].detach().cpu().clone()
                    )
                    self._dprm_omni_state_cache.move_to_end(cache_key)
                    while len(self._dprm_omni_state_cache) > cache_limit:
                        self._dprm_omni_state_cache.popitem(last=False)

        inputs["input_ids"] = state
        inputs["labels"] = labels
        if not labels.ne(-100).any():
            raise RuntimeError("matched Omni roll-in produced no masked visual targets")
        return inputs

    def _dprm_theory_stats(self) -> dict:
        if not hasattr(self, "_dprm_theory_state"):
            self._dprm_theory_state = {
                "masked": 0.0,
                "tokens": 0.0,
                "samples": 0.0,
                "pos_sum": 0.0,
                "pos_sumsq": 0.0,
                "low": 0.0,
                "mid": 0.0,
                "high": 0.0,
                "grad_norm_window": [],
                "last_grad": {},
            }
        return self._dprm_theory_state

    def _dprm_accumulate_input_stats(self, inputs) -> None:
        if os.environ.get("DPRM_OMNI_THEORY_LOG", "1") != "1":
            return
        labels = inputs.get("labels")
        if labels is None:
            return
        attention_mask = inputs.get("attention_mask")
        with torch.no_grad():
            pred_mask = labels.ne(-100)
            if attention_mask is not None:
                valid_mask = attention_mask.bool()
            else:
                valid_mask = torch.ones_like(pred_mask, dtype=torch.bool)

            valid_count = valid_mask.sum().float()
            masked_count = pred_mask.sum().float()
            if valid_count.item() <= 0:
                return

            seq_len = labels.shape[-1]
            denom = max(seq_len - 1, 1)
            pos = torch.arange(seq_len, device=labels.device, dtype=torch.float32) / denom
            pos = pos.unsqueeze(0).expand_as(labels)
            masked_pos = pos[pred_mask]

            stats = self._dprm_theory_stats()
            stats["masked"] += float(masked_count.item())
            stats["tokens"] += float(valid_count.item())
            stats["samples"] += float(labels.shape[0])
            if masked_pos.numel() > 0:
                stats["pos_sum"] += float(masked_pos.sum().item())
                stats["pos_sumsq"] += float((masked_pos * masked_pos).sum().item())
                stats["low"] += float((masked_pos < 1.0 / 3.0).sum().item())
                stats["mid"] += float(((masked_pos >= 1.0 / 3.0) & (masked_pos < 2.0 / 3.0)).sum().item())
                stats["high"] += float((masked_pos >= 2.0 / 3.0).sum().item())

    def _dprm_collect_grad_stats(self, model) -> None:
        if os.environ.get("DPRM_OMNI_THEORY_LOG", "1") != "1":
            return
        interval = int(os.environ.get("DPRM_GRAD_VARIANCE_INTERVAL", "100"))
        next_step = self.state.global_step + 1
        if interval <= 0 or next_step % interval != 0:
            return

        device = self.args.device
        local = torch.zeros(3, device=device, dtype=torch.float64)
        with torch.no_grad():
            for param in model.parameters():
                grad = getattr(param, "grad", None)
                if grad is None:
                    continue
                grad = grad.detach().float()
                local[0] += grad.sum(dtype=torch.float64)
                local[1] += (grad * grad).sum(dtype=torch.float64)
                local[2] += grad.numel()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local, op=dist.ReduceOp.SUM)
        count = max(float(local[2].item()), 1.0)
        mean = float(local[0].item() / count)
        var = max(float(local[1].item() / count - mean * mean), 0.0)
        self._dprm_theory_stats()["last_grad"] = {
            "theory/grad_element_mean": mean,
            "theory/grad_element_variance": var,
            "theory/grad_element_count": count,
        }

    def _dprm_pop_theory_logs(self, grad_norm) -> dict:
        if os.environ.get("DPRM_OMNI_THEORY_LOG", "1") != "1":
            return {}
        stats = self._dprm_theory_stats()
        logs = {
            "theory/order_policy": os.environ.get("DPRM_TRAIN_ORDER_POLICY", "unknown"),
            "theory/dprm_conf_position_weight": float(os.environ.get("DPRM_CONF_POSITION_WEIGHT", "0.8")),
            "theory/dprm_random_position_weight": float(os.environ.get("DPRM_RANDOM_POSITION_WEIGHT", "0.5")),
        }
        if stats["tokens"] > 0:
            logs["theory/mask_ratio"] = stats["masked"] / stats["tokens"]
        if stats["samples"] > 0:
            logs["theory/masked_tokens_per_sample"] = stats["masked"] / stats["samples"]
        if stats["masked"] > 0:
            mean = stats["pos_sum"] / stats["masked"]
            var = max(stats["pos_sumsq"] / stats["masked"] - mean * mean, 0.0)
            logs["theory/masked_position_mean"] = mean
            logs["theory/masked_position_variance"] = var
            logs["theory/occupancy_low"] = stats["low"] / stats["masked"]
            logs["theory/occupancy_mid"] = stats["mid"] / stats["masked"]
            logs["theory/occupancy_high"] = stats["high"] / stats["masked"]
        rollin = getattr(self, "_dprm_omni_rollin_stats", None)
        if rollin and rollin["actions"] > 0:
            actions = float(rollin["actions"])
            logs.update({
                "theory/rollin_progress_mean": rollin["progress_sum"] / actions,
                "theory/rollin_progress_min": rollin["progress_min"],
                "theory/rollin_progress_max": rollin["progress_max"],
                "theory/rollin_early_fraction": rollin["early"] / actions,
                "theory/rollin_middle_fraction": rollin["middle"] / actions,
                "theory/rollin_late_fraction": rollin["late"] / actions,
                "theory/rollin_restarts": rollin["restarts"],
                "theory/rollin_candidates_per_action": rollin["candidate_sum"] / actions,
                "theory/rollin_selected_per_action": rollin["selected_sum"] / actions,
                "theory/rollin_dprm_scorer_fraction": rollin["dprm_scorer_actions"] / actions,
                "theory/rollin_dprm_direct_override_fraction": (
                    rollin["dprm_direct_overrides"]
                    / max(float(rollin["dprm_scorer_actions"]), 1.0)
                ),
            })
            rollin.update({
                "actions": 0,
                "restarts": 0,
                "progress_sum": 0.0,
                "progress_min": 1.0,
                "progress_max": 0.0,
                "early": 0,
                "middle": 0,
                "late": 0,
                "candidate_sum": 0,
                "selected_sum": 0,
                "dprm_scorer_actions": 0,
                "dprm_direct_overrides": 0,
            })

        if grad_norm is not None:
            grad_norm_value = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            window = stats["grad_norm_window"]
            window.append(grad_norm_value)
            max_window = int(os.environ.get("DPRM_GRAD_NORM_WINDOW", "100"))
            del window[:-max_window]
            if len(window) >= 2:
                logs["theory/grad_norm_window_variance"] = float(np.var(window, ddof=1))
                logs["theory/grad_norm_window_mean"] = float(np.mean(window))

        logs.update(stats.get("last_grad", {}))
        stats.update({
            "masked": 0.0,
            "tokens": 0.0,
            "samples": 0.0,
            "pos_sum": 0.0,
            "pos_sumsq": 0.0,
            "low": 0.0,
            "mid": 0.0,
            "high": 0.0,
            "last_grad": {},
        })
        return logs

    def _dprm_write_theory_jsonl(self, logs: dict) -> None:
        log_dir = os.environ.get("DPRM_OMNI_THEORY_LOG_DIR")
        if not log_dir or not self.is_world_process_zero():
            return
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, "theory_metrics.jsonl")
        payload = {"global_step": self.state.global_step, **logs}
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _get_train_sampler(self):
        if os.environ.get("DPRM_OMNI_SEQUENTIAL_SAMPLER", "0") == "1":
            return SequentialSampler(self.train_dataset)
        return super()._get_train_sampler()

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if self.args.dataloader_num_workers > 0:
            dataloader_params["multiprocessing_context"] = "spawn"

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            if self.args.dataloader_num_workers > 0:
                dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _adjust_lr(self, opt_model):
        decay_parameters = self.get_decay_parameter_names(opt_model)

        if self.args.vision_model_lr_mult != 1.0 or self.args.vision_model_lr_decay_rate != 1.0:
            vision_parameters = [name for name, _ in opt_model.named_parameters() if "vision_model" in name]
            logger.info(f"{vision_parameters=}")
        else:
            vision_parameters = []

        exclude_parameters = vision_parameters

        def _init_optimizer_grouped_parameters(opt_model, exclude_parameters):
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and n not in exclude_parameters)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and n not in exclude_parameters)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            return optimizer_grouped_parameters

        optimizer_grouped_parameters = _init_optimizer_grouped_parameters(opt_model, exclude_parameters)

        def _adjust_lr_decay(opt_model, optimizer_grouped_parameters):
            for n, p in opt_model.named_parameters():
                if p.requires_grad and n in vision_parameters:
                    pass
                else:
                    continue

                if n in decay_parameters:
                    weight_decay = self.args.weight_decay
                else:
                    weight_decay = 0.0

                lr = self.args.learning_rate * get_vit_lr_decay_rate(n, opt_model.config.visual.num_hidden_layers, self.args.vision_model_lr_decay_rate)

                optimizer_grouped_parameters.append(
                    {
                        "params": [p],
                        "weight_decay": weight_decay,
                        "lr": lr,
                    }
                )
                logger.info(f"create_optimizer name {n} weight_decay {weight_decay} lr {lr}")
            return optimizer_grouped_parameters

        def _adjust_lr_mult(opt_model, optimizer_grouped_parameters):
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and n in vision_parameters)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate * self.args.vision_model_lr_mult,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and n in vision_parameters)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate * self.args.vision_model_lr_mult,
                    },
                ]
            )
            logger.info(f"create_optimizer name {[n for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and n in vision_parameters)]} weight_decay {self.args.weight_decay} lr_mult {self.args.vision_model_lr_mult}")
            logger.info(f"create_optimizer name {[n for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and n in vision_parameters)]} weight_decay {0.0} lr_mult {self.args.vision_model_lr_mult}")
            return optimizer_grouped_parameters

        if self.args.vision_model_lr_decay_rate != 1.0:
            optimizer_grouped_parameters = _adjust_lr_decay(opt_model, optimizer_grouped_parameters)

        elif self.args.vision_model_lr_mult != 1.0:
            optimizer_grouped_parameters = _adjust_lr_mult(opt_model, optimizer_grouped_parameters)

        return optimizer_grouped_parameters

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:

            optimizer_grouped_parameters = self._adjust_lr(opt_model)

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer


    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        print_batch(inputs, self.processing_class, self.args)

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        inputs = self._dprm_apply_matched_omni_order(model, inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            elif is_torch_hpu_available():
                logger.warning(
                    "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
                )
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # Finally we need to normalize the loss for reporting
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

            return loss.detach()


    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batch_samples = []
        num_items_in_batch = None

        for _ in range(num_batches):
            try:
                while True:
                    batch_sample = next(epoch_iterator)
                    if "input_ids" in batch_sample:
                        break
                batch_samples += [batch_sample]
            except StopIteration:
                break

        count_num_items_in_batch = (
            len(batch_samples) > 0
            and "labels" in batch_samples[0]
            and (
                # num_items_in_batch is passed to model forward
                # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3757
                self.model_accepts_loss_kwargs
                # num_items_in_batch is passed to compute_loss_func
                # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3773
                or self.compute_loss_func is not None
                # num_items_in_batch is also verified if (self.model_accepts_loss_kwargs or self.compute_loss_func)
                # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3790
            )
        )

        if count_num_items_in_batch:
            # For now we don't support object detection
            try:
                num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
            except (TypeError, AttributeError):
                pass

        if num_items_in_batch is not None:
            if self.args.average_tokens_across_devices:
                num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum()

            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(device)

                if self.args.n_gpu > 1 and num_items_in_batch.dim() == 0:
                    # In the DataParallel case, convert the scalar tensor into a 1-dim tensor
                    num_items_in_batch = num_items_in_batch.unsqueeze(0)

        return batch_samples, num_items_in_batch


    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the initial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)
        if os.environ.get("DPRM_OMNI_REPEAT_TO_MAX_STEPS", "0") == "1" and args.max_steps > 0:
            # Packed Omni samples can yield far fewer effective batches than len(dataloader).
            # Keep cycling epochs until DefaultFlowCallback stops at args.max_steps.
            num_train_epochs = sys.maxsize

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # Can't delay optimizer creation when using FSDP2: https://github.com/huggingface/accelerate/blob/3f636d626063ffcf9a337c7d3624d61b7d187d59/src/accelerate/accelerator.py#L1404
        is_fsdp2 = self.is_fsdp_enabled and (getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2)
        if is_fsdp2:
            delay_optimizer_creation = False

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        for attr in ("model", "optimizer", "lr_scheduler"):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader

        self.state.init_training_references(self, max_steps, num_train_epochs, trial)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0, device=args.device)
        tr_loss_of_various_task = {
            "loss_asr": torch.tensor(0.0, device=args.device),
            "loss_tts": torch.tensor(0.0, device=args.device),
            "loss_tqa": torch.tensor(0.0, device=args.device),
            "loss_sqa": torch.tensor(0.0, device=args.device),
            "loss_t2i": torch.tensor(0.0, device=args.device),
            "loss_s2i": torch.tensor(0.0, device=args.device),
            "loss_vqa": torch.tensor(0.0, device=args.device),
            "loss_svqa": torch.tensor(0.0, device=args.device),
        }
        tr_loss_count_of_various_task = {
            "loss_asr": torch.tensor(0.0, device=args.device),
            "loss_tts": torch.tensor(0.0, device=args.device),
            "loss_tqa": torch.tensor(0.0, device=args.device),
            "loss_sqa": torch.tensor(0.0, device=args.device),
            "loss_t2i": torch.tensor(0.0, device=args.device),
            "loss_s2i": torch.tensor(0.0, device=args.device),
            "loss_vqa": torch.tensor(0.0, device=args.device),
            "loss_svqa": torch.tensor(0.0, device=args.device),
        }
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        learning_rate = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = num_examples % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
            if args.gradient_accumulation_steps == 1:
                total_updates -= 1
            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                            self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).sum().item()
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1
                        and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                        else contextlib.nullcontext
                    )
                    with context():
                        tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                    self._dprm_accumulate_input_stats(inputs)

                    cur_input_text = self.processing_class.decode(inputs['input_ids'][0])
                    cur_task_name = get_task_from_inputs(cur_input_text)

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step
                        tr_loss_of_various_task[cur_task_name] = tr_loss_of_various_task[cur_task_name] + tr_loss_step.detach()
                        tr_loss_count_of_various_task[cur_task_name] = tr_loss_count_of_various_task[cur_task_name] + 1

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm

                        self._dprm_collect_grad_stats(model)

                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                        self.optimizer.step()

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        # get leaning rate before update
                        learning_rate = self._get_learning_rate()

                        if not self.accelerator.optimizer_step_was_skipped:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                        self._maybe_log_save_evaluate(
                            tr_loss,
                            tr_loss_of_various_task,
                            tr_loss_count_of_various_task,
                            grad_norm,
                            model,
                            trial,
                            epoch,
                            ignore_keys_for_eval,
                            start_time,
                            learning_rate=learning_rate,
                        )
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(
                tr_loss, tr_loss_of_various_task, tr_loss_count_of_various_task, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=learning_rate
            )

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)


    def _maybe_log_save_evaluate(
        self, tr_loss, tr_loss_of_various_task, tr_loss_count_of_various_task, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            for k in tr_loss_of_various_task:
                tr_loss_k_scalar = self._nested_gather(tr_loss_of_various_task[k]).sum().item()
                tr_loss_k_count = self._nested_gather(tr_loss_count_of_various_task[k]).sum().item()

                tr_loss_of_various_task[k] -= tr_loss_of_various_task[k]
                tr_loss_count_of_various_task[k] -= tr_loss_count_of_various_task[k]

                if tr_loss_k_count > 0:
                    logs[k] = round(tr_loss_k_scalar / tr_loss_k_count, 4)

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs.update(self._dprm_pop_theory_logs(grad_norm))
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)
            self._dprm_write_theory_jsonl(logs)


        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


def get_vit_lr_decay_rate(name, num_layers, lr_decay_rate):

    layer_id = num_layers + 1
    if "vision_model." in name:
        if ".position_embedding." in name or ".conv1." in name:
            layer_id = 0
        elif ".layers." in name:
            layer_id = int(name[name.find(".layers.") :].split(".")[2]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)
