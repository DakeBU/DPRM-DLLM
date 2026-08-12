import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable, Tuple
from pathlib import Path
import json
from tqdm import tqdm

from .graph import Graph, AbsorbingGraph
from .noise import NoiseSchedule

try:
    from dprm import (
        DPRMConfig,
        HostDPRMBatch,
        OnlineDPRMController,
        scalarize_benefits,
        sparse_reconstruction_benefits,
    )
except Exception:
    DPRMConfig = None
    HostDPRMBatch = None
    OnlineDPRMController = None
    scalarize_benefits = None
    sparse_reconstruction_benefits = None

Tensor = torch.Tensor


class SEDDTrainer:
    def __init__(
        self,
        model: nn.Module,
        graph: Graph,
        noise: NoiseSchedule,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = None,
        gradient_clip: float = 1.0,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
        order_policy: str = "all",
        loss_token_fraction: float = 1.0,
        dprm_config: Optional[Dict[str, Any]] = None,
        gene_aux_bin_ids: Optional[Tensor] = None,
    ):
        self.model = model
        self.graph = graph
        self.noise = noise
        self.gradient_clip = gradient_clip
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.order_policy = order_policy
        self.loss_token_fraction = float(loss_token_fraction)
        self.dprm_controller = None
        dcfg = dprm_config or {}
        self.dprm_reward_mode = str(dcfg.get("reward_mode", "selected_logprob"))
        self.dprm_objective_weights = tuple(
            float(value) for value in dcfg.get("objective_weights", [0.45, 0.35, 0.20])
        )
        self.dprm_tchebycheff_temperature = float(dcfg.get("tchebycheff_temperature", 0.05))
        self.dprm_tchebycheff_augmentation = float(dcfg.get("tchebycheff_augmentation", 0.05))
        self.dprm_aux_mode = str(dcfg.get("aux_mode", "none"))
        if self.dprm_aux_mode not in {"none", "gene", "predicted_zero"}:
            raise ValueError(f"Unknown DPRM auxiliary mode: {self.dprm_aux_mode}")
        self.gene_aux_bin_ids = (
            gene_aux_bin_ids.long().to(device or torch.device("cpu"))
            if gene_aux_bin_ids is not None
            else None
        )

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

        if self.order_policy in {"dprm_random", "dprm_confidence"}:
            if OnlineDPRMController is None:
                raise ImportError(
                    "DPRM order_policy selected, but dprm package is not importable. "
                    "Install the DPRM-DLLM package in the active environment."
                )
            self.dprm_controller = OnlineDPRMController(
                DPRMConfig(
                    num_phases=int(dcfg.get("num_phases", 8)),
                    confidence_bins=int(dcfg.get("confidence_bins", 16)),
                    aux_bins=(
                        2
                        if self.dprm_aux_mode == "predicted_zero"
                        else (
                            int(self.gene_aux_bin_ids.max().item()) + 1
                            if self.gene_aux_bin_ids is not None and self.gene_aux_bin_ids.numel()
                            else 1
                        )
                    ),
                    reward_temperature=float(dcfg.get("reward_temperature", 1.0)),
                    guidance_scale=float(dcfg.get("guidance_scale", 1.0)),
                    warmup_steps=int(dcfg.get("warmup_steps", 500)),
                    switch_steps=int(dcfg.get("switch_steps", 2000)),
                    ready_count=int(dcfg.get("ready_count", 128)),
                    sampled_soft_bon=bool(dcfg.get("sampled_soft_bon", True)),
                    candidate_multiplier=int(dcfg.get("candidate_multiplier", 4)),
                    min_candidates=int(dcfg.get("min_candidates", 8)),
                    max_candidates=int(dcfg.get("max_candidates", 64)),
                ),
                device=self.device,
            )
        if self.gene_aux_bin_ids is not None:
            self.gene_aux_bin_ids = self.gene_aux_bin_ids.to(self.device)

        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        self.scheduler = scheduler
        
        # GradScaler is only needed for fp16, not bf16
        self.scaler = None
        if self.use_amp and self.amp_dtype == torch.float16:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": []}

    def compute_loss(
        self,
        x_clean: Tensor,
        mask_ratio: float = 0.15,
    ) -> Tensor:

        batch_size, seq_len = x_clean.shape
        device = x_clean.device

        t = torch.rand(batch_size, device=device)
        sigma = self.noise.total(t)

        if isinstance(self.graph, AbsorbingGraph):
            if self._uses_ordered_training():
                x_noised = self._build_ordered_noised_state(
                    x_clean=x_clean,
                    sigma=sigma,
                )
            else:
                x_noised = self._mask_tokens(x_clean, mask_ratio, sigma)
        else:
            x_noised = self.graph.sample_transition(x_clean, sigma)

        # Wrap forward pass in autocast for mixed precision
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            pred_score = self.model.score(x_noised, sigma)

            mask_idx = getattr(self.graph, 'mask_index', self.graph.num_states - 1)
            is_masked = (x_noised == mask_idx)

            if not is_masked.any():
                return torch.tensor(0.0, device=device, requires_grad=True)

            if self._uses_ordered_training():
                # Ordered training changes the masked-state distribution itself.
                # The denoising CE is therefore computed on the remaining masked
                # positions of the teacher-forced trajectory, not on a subset of
                # a random-mask state.
                train_mask = is_masked
            else:
                train_mask = self._select_training_positions(
                    pred_score=pred_score,
                    x_noised=x_noised,
                    x_clean=x_clean,
                    is_masked=is_masked,
                    sigma=sigma,
                )
            if not train_mask.any():
                train_mask = is_masked

            pred_at_mask = pred_score[train_mask]
            target_at_mask = x_clean[train_mask]

            ce = F.cross_entropy(pred_at_mask, target_at_mask, reduction='none')
            scale = is_masked.sum().float() / train_mask.sum().clamp_min(1).float()
            loss = ce.sum() * scale / batch_size

            if self.model.training and self.dprm_controller is not None and not self._uses_ordered_training():
                self._observe_dprm(
                    pred_score=pred_score.detach(),
                    x_clean=x_clean,
                    train_mask=train_mask.detach(),
                    sigma=sigma.detach(),
                )

        return loss

    def _uses_ordered_training(self) -> bool:
        return self.order_policy in {
            "random_ordered",
            "confidence",
            "dprm_random",
            "dprm_confidence",
        }

    def _phase_from_sigma(self, sigma: Tensor, batch_size: int) -> Tensor:
        p_mask = (1 - torch.exp(-sigma)).detach().clamp(0.0, 1.0 - 1e-6)
        num_phases = int(self.dprm_controller.cfg.num_phases) if self.dprm_controller is not None else 8
        phase = torch.floor(p_mask * num_phases).long().clamp_(0, num_phases - 1)
        if phase.numel() == 1:
            phase = phase.expand(batch_size)
        return phase

    def _num_select_per_row(self, is_masked: Tensor) -> Tensor:
        masked = is_masked.sum(dim=1).long()
        if self.loss_token_fraction >= 1.0:
            return masked
        k = torch.ceil(masked.float() * max(self.loss_token_fraction, 0.0)).long()
        return torch.where(masked > 0, k.clamp_min(1), torch.zeros_like(masked))

    def _aux_grid(self, pred_score: Optional[Tensor], batch_size: int) -> Optional[Tensor]:
        if self.dprm_aux_mode == "predicted_zero":
            if pred_score is None:
                raise ValueError("predicted_zero auxiliary buckets require denoiser scores")
            return (pred_score[..., :-1].argmax(dim=-1) != 0).long()
        if self.gene_aux_bin_ids is not None:
            return self.gene_aux_bin_ids.unsqueeze(0).expand(batch_size, -1)
        return None

    def _select_random_positions(self, is_masked: Tensor, num_select: Tensor) -> Tensor:
        selected = torch.zeros_like(is_masked, dtype=torch.bool)
        for row in range(is_masked.shape[0]):
            active = torch.where(is_masked[row])[0]
            k = int(num_select[row].item())
            if active.numel() == 0 or k <= 0:
                continue
            k = min(k, active.numel())
            chosen = active[torch.randperm(active.numel(), device=is_masked.device)[:k]]
            selected[row, chosen] = True
        return selected

    def _topk_from_scores(self, scores: Tensor, candidate_mask: Tensor, num_select: Tensor) -> Tensor:
        selected = torch.zeros_like(candidate_mask, dtype=torch.bool)
        scores = scores.masked_fill(~candidate_mask, float("-inf"))
        for row in range(candidate_mask.shape[0]):
            active = torch.where(candidate_mask[row])[0]
            k = int(num_select[row].item())
            if active.numel() == 0 or k <= 0:
                continue
            k = min(k, active.numel())
            chosen_local = torch.topk(scores[row, active], k=k, largest=True).indices
            selected[row, active[chosen_local]] = True
        return selected

    def _build_ordered_noised_state(
        self,
        x_clean: Tensor,
        sigma: Tensor,
    ) -> Tensor:
        """Teacher-forced ordered noising for train-test aligned DCM.

        We first match the original marginal corruption level by choosing the
        same expected number of masked coordinates as q(x_t | x_0), then reveal
        ground-truth coordinates from an all-mask state according to the chosen
        ordering policy. This is the training analogue of ordered decoding.
        """
        batch_size, seq_len = x_clean.shape
        device = x_clean.device
        mask_idx = getattr(self.graph, "mask_index", self.graph.num_states - 1)

        p_mask = (1 - torch.exp(-sigma)).detach().clamp(0.0, 1.0)
        desired_mask = torch.round(p_mask * seq_len).long().clamp(0, seq_len)
        reveal_counts = (seq_len - desired_mask).clamp(0, seq_len)

        x_state = torch.full_like(x_clean, mask_idx)
        candidate_mask = torch.ones_like(x_clean, dtype=torch.bool)

        if reveal_counts.max().item() <= 0:
            return x_state

        with torch.no_grad():
            probe_score = self.model.score(x_state, sigma)
            probs = probe_score[..., :-1].exp()
            confidence = probs.max(dim=-1).values.clamp_(1e-6, 1.0 - 1e-6)

            if self.order_policy == "random_ordered":
                scores = torch.rand_like(confidence)
                revealed = self._topk_from_scores(scores, candidate_mask, reveal_counts)
            elif self.order_policy == "confidence":
                scores = torch.log(confidence)
                revealed = self._topk_from_scores(scores, candidate_mask, reveal_counts)
            elif self.order_policy in {"dprm_random", "dprm_confidence"}:
                phase_ids = torch.zeros(batch_size, device=device, dtype=torch.long)
                host = HostDPRMBatch(
                    confidence=confidence.detach(),
                    candidate_mask=candidate_mask,
                    phase_ids=phase_ids,
                    aux_bin_ids=self._aux_grid(probe_score.detach(), batch_size),
                    global_step=int(self.step),
                )
                summary = self.dprm_controller.summarize(host)
                if self.order_policy == "dprm_random":
                    random_base = torch.rand_like(summary.score)
                    dprm_score = (
                        summary.base_score
                        + self.dprm_controller.cfg.guidance_scale * summary.dprm_value
                    )
                    scores = (1.0 - summary.gate) * random_base + summary.gate * dprm_score
                    revealed = self._topk_from_scores(scores, candidate_mask, reveal_counts)
                else:
                    revealed = self.dprm_controller.select(host, reveal_counts).selected_mask
                if self.model.training:
                    self._observe_dprm_ordered(
                        pred_score=probe_score.detach(),
                        x_clean=x_clean,
                        revealed_mask=revealed.detach(),
                        phase_ids=phase_ids,
                    )
            else:
                raise ValueError(f"Unknown ordered training policy: {self.order_policy}")

        return torch.where(revealed, x_clean, x_state)

    def _observe_dprm_ordered(
        self,
        pred_score: Tensor,
        x_clean: Tensor,
        revealed_mask: Tensor,
        phase_ids: Tensor,
    ) -> None:
        if self.dprm_controller is None or not revealed_mask.any():
            return
        probs = pred_score[..., :-1].exp()
        confidence = probs.max(dim=-1).values.clamp_(1e-6, 1.0 - 1e-6)
        reward = self._dprm_reconstruction_reward(pred_score, x_clean, revealed_mask)
        host = HostDPRMBatch(
            confidence=confidence,
            candidate_mask=revealed_mask,
            phase_ids=phase_ids,
            aux_bin_ids=self._aux_grid(pred_score, x_clean.shape[0]),
            global_step=int(self.step),
        )
        self.dprm_controller.observe(host, revealed_mask, reward.detach())

    def _dprm_reconstruction_reward(
        self,
        pred_score: Tensor,
        x_clean: Tensor,
        selected_mask: Tensor,
    ) -> Tensor:
        logits = pred_score[..., :-1]
        denom = selected_mask.sum(dim=1).clamp_min(1).float()
        if self.dprm_reward_mode == "selected_logprob":
            gathered = logits.gather(
                -1,
                x_clean.clamp_max(logits.shape[-1] - 1).unsqueeze(-1),
            ).squeeze(-1)
            return (gathered * selected_mask).sum(dim=1) / denom

        predicted = logits.argmax(dim=-1)
        max_bin_distance = max(int(logits.shape[-1]) - 1, 1)
        objectives = sparse_reconstruction_benefits(
            predicted,
            x_clean,
            selected_mask,
            max_bin_distance=float(max_bin_distance),
        )
        weights = torch.as_tensor(
            self.dprm_objective_weights,
            dtype=objectives.dtype,
            device=objectives.device,
        )
        if self.dprm_reward_mode == "reconstruction_weighted_sum":
            method = "weighted_sum"
        elif self.dprm_reward_mode == "reconstruction_tchebycheff":
            method = "smooth_tchebycheff"
        else:
            raise ValueError(f"Unknown DPRM reward mode: {self.dprm_reward_mode}")
        return scalarize_benefits(
            objectives,
            weights,
            method=method,
            temperature=self.dprm_tchebycheff_temperature,
            augmentation=self.dprm_tchebycheff_augmentation,
        )

    def _select_training_positions(
        self,
        pred_score: Tensor,
        x_noised: Tensor,
        x_clean: Tensor,
        is_masked: Tensor,
        sigma: Tensor,
    ) -> Tensor:
        if self.order_policy == "all" or self.loss_token_fraction >= 1.0:
            return is_masked

        num_select = self._num_select_per_row(is_masked)
        if self.order_policy in {"random", "random_ordered"}:
            return self._select_random_positions(is_masked, num_select)

        probs = pred_score[..., :-1].exp()
        confidence = probs.max(dim=-1).values.clamp_(1e-6, 1.0 - 1e-6)

        if self.order_policy == "confidence":
            return self._topk_from_scores(torch.log(confidence), is_masked, num_select)

        if self.order_policy in {"dprm_random", "dprm_confidence"}:
            phase_ids = self._phase_from_sigma(sigma, x_noised.shape[0]).to(x_noised.device)
            host = HostDPRMBatch(
                confidence=confidence.detach(),
                candidate_mask=is_masked,
                phase_ids=phase_ids,
                aux_bin_ids=self._aux_grid(pred_score, x_noised.shape[0]),
                global_step=int(self.step),
            )
            summary = self.dprm_controller.summarize(host)
            if self.order_policy == "dprm_random":
                random_base = torch.rand_like(summary.score)
                dprm_score = summary.base_score + self.dprm_controller.cfg.guidance_scale * summary.dprm_value
                scores = (1.0 - summary.gate) * random_base + summary.gate * dprm_score
                return self._topk_from_scores(scores, is_masked, num_select)
            return self.dprm_controller.select(host, num_select).selected_mask

        raise ValueError(f"Unknown order_policy: {self.order_policy}")

    def _observe_dprm(
        self,
        pred_score: Tensor,
        x_clean: Tensor,
        train_mask: Tensor,
        sigma: Tensor,
    ) -> None:
        probs = pred_score[..., :-1].exp()
        confidence = probs.max(dim=-1).values.clamp_(1e-6, 1.0 - 1e-6)
        gathered = pred_score[..., :-1].gather(-1, x_clean.clamp_max(pred_score.shape[-1] - 2).unsqueeze(-1)).squeeze(-1)
        denom = train_mask.sum(dim=1).clamp_min(1).float()
        reward = (gathered * train_mask).sum(dim=1) / denom
        phase_ids = self._phase_from_sigma(sigma, x_clean.shape[0]).to(x_clean.device)
        host = HostDPRMBatch(
            confidence=confidence,
            candidate_mask=train_mask,
            phase_ids=phase_ids,
            aux_bin_ids=self._aux_grid(pred_score, x_clean.shape[0]),
            global_step=int(self.step),
        )
        self.dprm_controller.observe(host, train_mask, reward.detach())

    def _mask_tokens(
        self,
        x: Tensor,
        mask_ratio: float,
        sigma: Tensor,
    ) -> Tensor:

        batch_size, seq_len = x.shape
        device = x.device
        mask_idx = self.graph.mask_index

        p_mask = 1 - torch.exp(-sigma)  # [batch_size]
        p_mask = p_mask.view(-1, 1)  # [batch_size, 1]

        mask = torch.rand(batch_size, seq_len, device=device) < p_mask

        x_masked = x.clone()
        x_masked[mask] = mask_idx

        return x_masked

    def train_step(self, batch: Tensor, mask_ratio: float = 0.15) -> float:

        self.model.train()
        self.optimizer.zero_grad()

        batch = batch.to(self.device)
        loss = self.compute_loss(batch, mask_ratio)

        # Backward pass with gradient scaling for mixed precision
        if self.scaler is not None:
            # fp16 with gradient scaling
            self.scaler.scale(loss).backward()
            
            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # bf16 or fp32 - no gradient scaling needed
            loss.backward()

            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            self.optimizer.step()
        
        self.step += 1

        return loss.item()

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        mask_ratio: float = 0.15,
    ) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(self.device)
            
            # Use autocast for validation too
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                loss = self.compute_loss(batch, mask_ratio)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        mask_ratio: float = 0.15,
        log_interval: int = 100,
        val_interval: int = 1,
        checkpoint_dir: Optional[str] = None,
        save_interval: int = 10,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Start from current epoch if resuming from checkpoint
        start_epoch = self.epoch
        end_epoch = start_epoch + num_epochs
        for epoch in range(start_epoch, end_epoch):
            self.epoch = epoch

            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{end_epoch}")
            for batch in pbar:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]

                loss = self.train_step(batch, mask_ratio)
                epoch_loss += loss
                num_batches += 1

                if self.step % log_interval == 0:
                    pbar.set_postfix({"loss": f"{loss:.4f}"})

            avg_train_loss = epoch_loss / max(num_batches, 1)
            self.history["train_loss"].append(avg_train_loss)

            val_loss = None
            if val_loader and (epoch + 1) % val_interval == 0:
                val_loss = self.validate(val_loader, mask_ratio)
                self.history["val_loss"].append(val_loss)

                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    if checkpoint_dir:
                        self.save_checkpoint(checkpoint_dir / "best.pt")

            log_msg = f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}"
            if val_loss is not None:
                log_msg += f", val_loss={val_loss:.4f}"
            print(log_msg)

            if callback:
                metrics = {
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                }
                callback(self, epoch, metrics)

            if checkpoint_dir and (epoch + 1) % save_interval == 0:
                self.save_checkpoint(checkpoint_dir / f"epoch_{epoch + 1}.pt")

        if checkpoint_dir:
            self.save_checkpoint(checkpoint_dir / "final.pt")

        return self.history

    def save_checkpoint(self, path: str):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "history": self.history,
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.dprm_controller is not None:
            checkpoint["dprm_state_dict"] = self.dprm_controller.state_dict()
            checkpoint["order_policy"] = self.order_policy
            checkpoint["loss_token_fraction"] = self.loss_token_fraction
            checkpoint["dprm_reward_config"] = {
                "mode": self.dprm_reward_mode,
                "objective_names": [
                    "nonzero_recovery",
                    "nonzero_mae_benefit",
                    "zero_accuracy",
                ],
                "objective_weights": list(self.dprm_objective_weights),
                "tchebycheff_temperature": self.dprm_tchebycheff_temperature,
                "tchebycheff_augmentation": self.dprm_tchebycheff_augmentation,
                "aux_mode": self.dprm_aux_mode,
            }
            if self.gene_aux_bin_ids is not None:
                checkpoint["dprm_gene_aux_bin_ids"] = self.gene_aux_bin_ids.detach().cpu()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.step = checkpoint.get("step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.history = checkpoint.get("history", {"train_loss": [], "val_loss": []})

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.dprm_controller is not None and "dprm_state_dict" in checkpoint:
            self.dprm_controller.load_state_dict(checkpoint["dprm_state_dict"])


def create_trainer(
    model: nn.Module,
    graph: Graph,
    noise: NoiseSchedule,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    device: Optional[torch.device] = None,
) -> SEDDTrainer:

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return SEDDTrainer(
        model=model,
        graph=graph,
        noise=noise,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )


class PerturbationTrainer:
    """Trainer for perturbation prediction using discrete diffusion.

    This trainer handles the perturbation prediction task:
    - Input: control cell + perturbation label
    - Output: predicted perturbed cell

    Training procedure:
    1. Sample diffusion time t
    2. Apply discrete diffusion (masking) to the perturbed cell
    3. Model predicts the perturbed cell from: masked_perturbed + perturbation_label
    4. Loss: cross-entropy at masked positions

    Note: We could also condition on control cells, but following STATE,
    we primarily condition on the perturbation label and learn the perturbation effect.
    """

    def __init__(
        self,
        model: nn.Module,
        graph: Graph,
        noise: NoiseSchedule,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = None,
        gradient_clip: float = 1.0,
        cond_label_lookup: Optional[Tensor] = None,
        cell_type_lookup: Optional[dict] = None,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
    ):
        self.model = model
        self.graph = graph
        self.noise = noise
        self.gradient_clip = gradient_clip
        self.cond_label_lookup = cond_label_lookup
        self.cell_type_lookup = cell_type_lookup
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

        # Move cond_label_lookup to device if provided
        if self.cond_label_lookup is not None:
            self.cond_label_lookup = self.cond_label_lookup.to(self.device)
            print(f"Using conditional label lookup with shape: {self.cond_label_lookup.shape}")
        
        # Store cell type lookup (dictionary mapping cell_type names to indices)
        if self.cell_type_lookup is not None:
            print(f"Using cell type conditioning with {len(self.cell_type_lookup)} cell types")

        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        self.scheduler = scheduler
        
        # GradScaler is only needed for fp16, not bf16
        self.scaler = None
        if self.use_amp and self.amp_dtype == torch.float16:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": []}

    def compute_loss(
        self,
        pert_labels: Tensor,
        perturbed: Tensor,
        mask_ratio: float = 0.15,
        cell_type_labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute perturbation prediction loss.

        Args:
            control: Control cell expression [batch, seq_len] (currently unused but available)
            pert_labels: Perturbation labels [batch]
            perturbed: True perturbed expression [batch, seq_len]
            mask_ratio: Masking ratio (used by absorbing graph)
            cell_type_labels: Optional cell type labels [batch]

        Returns:
            Cross-entropy loss at masked positions
        """
        batch_size, seq_len = perturbed.shape
        device = perturbed.device

        # Sample diffusion time
        t = torch.rand(batch_size, device=device)
        sigma = self.noise.total(t)

        # Apply discrete diffusion (masking) to perturbed cells
        if isinstance(self.graph, AbsorbingGraph):
            x_noised = self._mask_tokens(perturbed, mask_ratio, sigma)
        else:
            x_noised = self.graph.sample_transition(perturbed, sigma)

        # Model predicts perturbed from noised + perturbation label + cell type
        # Wrap forward pass in autocast for mixed precision
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            loss = self.model.get_loss(
                x_perturbed=perturbed,
                x_noised=x_noised,
                sigma=sigma,
                pert_labels=pert_labels,
                graph=self.graph,
                cell_type_labels=cell_type_labels,
            )

        return loss

    def _mask_tokens(
        self,
        x: Tensor,
        mask_ratio: float,
        sigma: Tensor,
    ) -> Tensor:
        """Apply masking to tokens based on diffusion time."""
        batch_size, seq_len = x.shape
        device = x.device
        mask_idx = self.graph.mask_index

        # Masking probability based on diffusion time
        p_mask = 1 - torch.exp(-sigma)  # [batch_size]
        p_mask = p_mask.view(-1, 1)  # [batch_size, 1]

        # Sample mask positions
        mask = torch.rand(batch_size, seq_len, device=device) < p_mask

        # Apply masking
        x_masked = x.clone()
        x_masked[mask] = mask_idx

        return x_masked

    def train_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor],
        mask_ratio: float = 0.15
    ) -> float:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Unpack batch - handle cell-load dictionary format
        cell_type_labels = None
        if isinstance(batch, dict):
            # Cell-load batch format
            perturbed = batch['pert_cell_emb'].to(self.device)
            pert_emb = batch['pert_emb'].to(self.device)

            # Extract cell type labels if available
            if 'cell_type' in batch and self.cell_type_lookup is not None:
                cell_type_names = batch['cell_type']
                # Convert cell type names to indices
                cell_type_indices = []
                for ct_name in cell_type_names:
                    if ct_name in self.cell_type_lookup:
                        cell_type_indices.append(self.cell_type_lookup[ct_name])
                    else:
                        # Default to first cell type if not found
                        cell_type_indices.append(0)
                cell_type_labels = torch.tensor(cell_type_indices, device=self.device, dtype=torch.long)

            # Convert one-hot perturbation embeddings to indices
            # If pert_emb is one-hot, use argmax to get indices
            if pert_emb.dim() == 2 and pert_emb.shape[1] > 1:
                pert_labels = pert_emb.argmax(dim=-1)
            else:
                pert_labels = pert_emb.squeeze(-1).long()
        else:
            # Legacy tuple format: (pert_labels, perturbed)
            pert_labels, perturbed = batch
            pert_labels = pert_labels.to(self.device)
            perturbed = perturbed.to(self.device)

        pert_labels = self._normalize_pert_labels(pert_labels)

        # Apply conditional label lookup if provided
        pert_labels = self._apply_cond_label_lookup(pert_labels)

        # Ensure labels are long type (handle both scalar and vector labels)
        if pert_labels.dim() == 1:
            pert_labels = pert_labels.long()

        # Round and convert to long for discrete tokens
        perturbed = torch.round(perturbed).long()

        # Compute loss with cell type conditioning
        loss = self.compute_loss(pert_labels, perturbed, mask_ratio, cell_type_labels)

        # Backward pass with gradient scaling for mixed precision
        if self.scaler is not None:
            # fp16 with gradient scaling
            self.scaler.scale(loss).backward()
            
            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # bf16 or fp32 - no gradient scaling needed
            loss.backward()

            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            self.optimizer.step()
        
        self.step += 1

        return loss.item()

    def _apply_cond_label_lookup(self, pert_labels: Tensor) -> Tensor:
        """Replace perturbation labels with conditional labels from .pt file.

        Args:
            pert_labels: Original perturbation indices from dataloader [batch]

        Returns:
            Conditional labels from lookup table [batch] or [batch, emb_dim]
        """
        if self.cond_label_lookup is None:
            return pert_labels

        # Use the original indices to lookup conditional labels
        # pert_labels are indices into the perturbation list, which we use to index cond_label_lookup
        cond_labels = self.cond_label_lookup[pert_labels]

        # For scalar labels, fall back to original labels when missing or out of range
        if cond_labels.dim() == 1:
            num_perturbations = self.model.num_perturbations
            invalid = (cond_labels < 0) | (cond_labels >= num_perturbations)
            if invalid.any():
                invalid_count = invalid.sum().item()
                print(
                    f"WARNING: {invalid_count} conditional labels out of range; "
                    "falling back to original perturbation indices for those samples."
                )
                cond_labels = cond_labels.clone()
                cond_labels[invalid] = pert_labels[invalid]

        return cond_labels

    def _normalize_pert_labels(self, pert_labels: Tensor) -> Tensor:
        """Ensure perturbation labels are in [0, num_perturbations - 1]."""
        pert_labels = pert_labels.long()
        num_perturbations = self.model.num_perturbations

        if pert_labels.numel() == 0:
            return pert_labels

        min_label = int(pert_labels.min().item())
        max_label = int(pert_labels.max().item())

        # Common case: labels are 1-based but embedding expects 0-based
        if min_label == 1 and max_label == num_perturbations:
            pert_labels = pert_labels - 1
            return pert_labels

        if min_label < 0 or max_label >= num_perturbations:
            raise ValueError(
                "Perturbation labels out of range for embedding. "
                f"Expected [0, {num_perturbations - 1}], got [{min_label}, {max_label}]. "
                "Ensure dataset labels are zero-based or match num_perturbations."
            )

        return pert_labels

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        mask_ratio: float = 0.15,
    ) -> float:
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            # Unpack batch - handle cell-load dictionary format
            cell_type_labels = None
            if isinstance(batch, dict):
                # Cell-load batch format
                perturbed = batch['pert_cell_emb'].to(self.device)
                pert_emb = batch['pert_emb'].to(self.device)

                # Extract cell type labels if available
                if 'cell_type' in batch and self.cell_type_lookup is not None:
                    cell_type_names = batch['cell_type']
                    # Convert cell type names to indices
                    cell_type_indices = []
                    for ct_name in cell_type_names:
                        if ct_name in self.cell_type_lookup:
                            cell_type_indices.append(self.cell_type_lookup[ct_name])
                        else:
                            # Default to first cell type if not found
                            cell_type_indices.append(0)
                    cell_type_labels = torch.tensor(cell_type_indices, device=self.device, dtype=torch.long)

                # Convert one-hot perturbation embeddings to indices
                if pert_emb.dim() == 2 and pert_emb.shape[1] > 1:
                    pert_labels = pert_emb.argmax(dim=-1)
                else:
                    pert_labels = pert_emb.squeeze(-1).long()
            else:
                # Legacy tuple format: (control, pert_labels, perturbed)
                control, pert_labels, perturbed = batch
                pert_labels = pert_labels.to(self.device)
                perturbed = perturbed.to(self.device)

            pert_labels = self._normalize_pert_labels(pert_labels)

            # Apply conditional label lookup if provided
            pert_labels = self._apply_cond_label_lookup(pert_labels)

            # Ensure labels are long type (handle both scalar and vector labels)
            if pert_labels.dim() == 1:
                pert_labels = pert_labels.long()

            # Round and convert to long for discrete tokens
            perturbed = torch.round(perturbed).long()

            # Use autocast for validation too
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                loss = self.compute_loss(pert_labels, perturbed, mask_ratio, cell_type_labels)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        mask_ratio: float = 0.15,
        log_interval: int = 100,
        val_interval: int = 1,
        checkpoint_dir: Optional[str] = None,
        save_interval: int = 10,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:

        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Start from current epoch if resuming from checkpoint
        start_epoch = self.epoch
        end_epoch = start_epoch + num_epochs
        for epoch in range(start_epoch, end_epoch):
            self.epoch = epoch

            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{end_epoch}")
            for batch in pbar:
                loss = self.train_step(batch, mask_ratio)
                epoch_loss += loss
                num_batches += 1

                if self.step % log_interval == 0:
                    pbar.set_postfix({"loss": f"{loss:.4f}"})

            avg_train_loss = epoch_loss / max(num_batches, 1)
            self.history["train_loss"].append(avg_train_loss)

            val_loss = None
            if val_loader and (epoch + 1) % val_interval == 0:
                val_loss = self.validate(val_loader, mask_ratio)
                self.history["val_loss"].append(val_loss)

                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    if checkpoint_dir:
                        self.save_checkpoint(checkpoint_dir / "best.pt")

            log_msg = f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}"
            if val_loss is not None:
                log_msg += f", val_loss={val_loss:.4f}"
            print(log_msg)

            if callback:
                metrics = {
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                }
                callback(self, epoch, metrics)

            if checkpoint_dir and (epoch + 1) % save_interval == 0:
                self.save_checkpoint(checkpoint_dir / f"epoch_{epoch + 1}.pt")

        if checkpoint_dir:
            self.save_checkpoint(checkpoint_dir / "final.pt")

        return self.history

    def save_checkpoint(self, path: str):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "history": self.history,
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.step = checkpoint.get("step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.history = checkpoint.get("history", {"train_loss": [], "val_loss": []})

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])  
