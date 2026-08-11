import torch

from dprm import DPRMConfig, HostDPRMBatch, OnlineDPRMController


def make_host(step: int = 0) -> HostDPRMBatch:
    return HostDPRMBatch(
        confidence=torch.tensor([[0.99, 0.90, 0.80, 0.70, 0.60, 0.50]]),
        candidate_mask=torch.ones((1, 6), dtype=torch.bool),
        phase_ids=torch.tensor([0]),
        global_step=step,
    )


def test_confidence_warmup_selects_from_confidence_proposal() -> None:
    controller = OnlineDPRMController(
        DPRMConfig(
            num_phases=1,
            confidence_bins=4,
            warmup_steps=10,
            warmup_policy="confidence",
            sampled_soft_bon=False,
        )
    )
    picked = controller.select(make_host(), torch.tensor([2])).selected_mask
    assert torch.where(picked[0])[0].tolist() == [0, 1]


def test_random_warmup_uses_uniform_active_positions() -> None:
    torch.manual_seed(7)
    controller = OnlineDPRMController(
        DPRMConfig(
            num_phases=1,
            confidence_bins=4,
            warmup_steps=10,
            warmup_policy="random",
            sampled_soft_bon=False,
        )
    )
    picked = controller.select(make_host(), torch.tensor([2])).selected_mask
    selected = torch.where(picked[0])[0].tolist()
    assert len(selected) == 2
    assert selected != [0, 1]


def test_random_warmup_switches_to_reward_tilt() -> None:
    controller = OnlineDPRMController(
        DPRMConfig(
            num_phases=1,
            confidence_bins=2,
            reward_temperature=1.0,
            guidance_scale=4.0,
            warmup_steps=1,
            switch_steps=2,
            warmup_policy="random",
            ready_count=1,
            sampled_soft_bon=False,
        )
    )
    controller.counts[0, :, 0] = 1
    controller.exp_reward_sums[0, 0, 0] = torch.exp(torch.tensor(1.0))
    controller.exp_reward_sums[0, 1, 0] = 1.0
    host = HostDPRMBatch(
        confidence=torch.tensor([[0.49, 0.90]]),
        candidate_mask=torch.tensor([[True, True]]),
        phase_ids=torch.tensor([0]),
        global_step=2,
    )
    picked = controller.select(host, torch.tensor([1])).selected_mask
    assert torch.where(picked[0])[0].tolist() == [0]
