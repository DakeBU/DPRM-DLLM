from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dprm import DPRMConfig, HostDPRMBatch, OnlineDPRMController


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controller = OnlineDPRMController(
        DPRMConfig(
            num_phases=4,
            confidence_bins=8,
            reward_temperature=1.0,
            guidance_scale=1.0,
            warmup_steps=2,
            switch_steps=6,
            ready_count=2,
            sampled_soft_bon=True,
            min_candidates=4,
            max_candidates=16,
        ),
        device=device,
    )

    batch_size, seq_len = 2, 12
    for step in range(8):
        confidence = torch.rand(batch_size, seq_len, device=device)
        candidate_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        phase_ids = OnlineDPRMController.phase_from_progress(
            step_index=step,
            total_steps=8,
            num_phases=4,
            batch_size=batch_size,
            device=device,
        )
        host = HostDPRMBatch(
            confidence=confidence,
            candidate_mask=candidate_mask,
            phase_ids=phase_ids,
            global_step=step,
        )
        selected = controller.select(host, torch.full((batch_size,), 3, device=device))

        # In a real host, this is the task utility already available from training
        # or decoding. Here we use a toy reward so the example is self-contained.
        reward = confidence.masked_fill(~selected.selected_mask, 0.0).sum(dim=1)
        reward = reward / selected.selected_mask.sum(dim=1).clamp_min(1)
        controller.observe(host, selected.selected_mask, reward)

        print(
            f"step={step:02d}",
            f"selected={selected.selected_mask.sum(dim=1).tolist()}",
            f"mean_gate={selected.gate.mean().item():.3f}",
        )


if __name__ == "__main__":
    main()
