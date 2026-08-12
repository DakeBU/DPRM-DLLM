import pytest
import torch

from dprm.multiobjective import (
    molecular_token_class_ids,
    scalarize_benefits,
    sparse_reconstruction_benefits,
)


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


def test_smooth_tchebycheff_preference_changes_the_selected_endpoint() -> None:
    benefits = torch.tensor([[0.95, 0.45], [0.70, 0.70], [0.45, 0.95]])
    first_focused = scalarize_benefits(
        benefits,
        torch.tensor([0.8, 0.2]),
        method="smooth_tchebycheff",
        temperature=0.02,
        augmentation=0.0,
    )
    second_focused = scalarize_benefits(
        benefits,
        torch.tensor([0.2, 0.8]),
        method="smooth_tchebycheff",
        temperature=0.02,
        augmentation=0.0,
    )
    assert int(first_focused.argmax()) == 0
    assert int(second_focused.argmax()) == 2


def test_smooth_tchebycheff_rejects_nonfinite_inputs() -> None:
    with pytest.raises(ValueError):
        scalarize_benefits(
            torch.tensor([[1.0, float("nan")]]),
            torch.tensor([0.5, 0.5]),
            method="smooth_tchebycheff",
        )


def test_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError):
        scalarize_benefits(torch.ones(2, 2), torch.tensor([1.0, -1.0]))


def test_sparse_reconstruction_separates_zero_and_nonzero_terms() -> None:
    target = torch.tensor([[0, 0, 2, 3]])
    predicted = torch.tensor([[0, 1, 2, 1]])
    benefits = sparse_reconstruction_benefits(
        predicted, target, torch.ones_like(target, dtype=torch.bool), max_bin_distance=7
    )
    torch.testing.assert_close(
        benefits,
        torch.tensor([[0.5, 6.0 / 7.0, 0.5]]),
    )


def test_molecular_token_classes_group_chemistry_and_syntax() -> None:
    classes = molecular_token_class_ids(
        ["[MASK]", "%12", "[C@@H]", "[NH+]", "[O-]", "[S]", "Cl", "X"]
    )

    assert classes.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
