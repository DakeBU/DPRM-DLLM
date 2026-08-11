import pytest
import torch

from dprm.multiobjective import scalarize_benefits


def test_weighted_sum() -> None:
    benefits = torch.tensor([[0.8, 0.4], [0.2, 1.0]])
    values = scalarize_benefits(
        benefits, torch.tensor([3.0, 1.0]), method="weighted_sum"
    )
    torch.testing.assert_close(values, torch.tensor([0.7, 0.4]))


def test_smooth_tchebycheff_rewards_balanced_solution() -> None:
    benefits = torch.tensor([[0.95, 0.30], [0.70, 0.70]])
    values = scalarize_benefits(
        benefits,
        torch.tensor([0.5, 0.5]),
        method="smooth_tchebycheff",
        temperature=0.02,
    )
    assert values[1] > values[0]


def test_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError):
        scalarize_benefits(torch.ones(2, 2), torch.tensor([1.0, -1.0]))
