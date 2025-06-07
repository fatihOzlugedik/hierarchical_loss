import torch
import numpy as np
from hiarachical_loss import HierarchicalLoss


def build_sample_hierarchy():
    return {
        "root": ["top1", "top2"],
        "top1": ["mid1", "mid2"],
        "top2": ["mid3"],
        "mid1": ["leaf1", "leaf2"],
        "mid2": ["leaf3"],
        "mid3": ["leaf4"],
    }


def test_transition_matrices_shapes_and_values():
    hierarchy = build_sample_hierarchy()
    hl = HierarchicalLoss(hierarchy, device="cpu")

    assert hl.T32.shape == (4, 3)
    assert hl.T21.shape == (3, 2)

    # Build expected matrices using the index mappings
    leaves = ["leaf1", "leaf2", "leaf3", "leaf4"]
    mids = ["mid1", "mid2", "mid3"]
    expected_T32 = np.zeros((4, 3), dtype=np.float32)
    mapping = {
        "leaf1": ("mid1"),
        "leaf2": ("mid1"),
        "leaf3": ("mid2"),
        "leaf4": ("mid3"),
    }
    for leaf, parent in mapping.items():
        i = hl.leaf_to_idx[leaf]
        j = hl.mid_to_idx[parent]
        expected_T32[i, j] = 1.0
    assert np.allclose(hl.T32.cpu().numpy(), expected_T32)

    expected_T21 = np.zeros((3, 2), dtype=np.float32)
    mapping_mid = {
        "mid1": "top1",
        "mid2": "top1",
        "mid3": "top2",
    }
    for mid, parent in mapping_mid.items():
        i = hl.mid_to_idx[mid]
        j = hl.top_to_idx[parent]
        expected_T21[i, j] = 1.0
    assert np.allclose(hl.T21.cpu().numpy(), expected_T21)


def test_get_loss_perfect_prediction_zero():
    hierarchy = build_sample_hierarchy()
    hl = HierarchicalLoss(hierarchy, device="cpu")
    # Choose one leaf and set logits to very high for that leaf
    leaf_name = "leaf2"
    truth_idx = hl.leaf_to_idx[leaf_name]
    logits = torch.full((1, len(hl.leaf_to_idx)), -10.0)
    logits[0, truth_idx] = 10.0

    one_hot = torch.zeros(1, len(hl.leaf_to_idx))
    one_hot[0, truth_idx] = 1.0

    loss = hl.get_loss(logits, one_hot)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4)


def test_hierarchical_vs_ce_loss_severity():
    hierarchy = build_sample_hierarchy()
    hl = HierarchicalLoss(hierarchy, device="cpu")

    truth_idx = hl.leaf_to_idx["leaf1"]
    target = torch.tensor([truth_idx])
    one_hot = torch.zeros(1, len(hl.leaf_to_idx))
    one_hot[0, truth_idx] = 1.0

    logits_same_branch = torch.log(torch.tensor([[0.1, 0.8, 0.05, 0.05]]))
    logits_other_branch = torch.log(torch.tensor([[0.1, 0.05, 0.05, 0.8]]))

    ce_same = torch.nn.functional.cross_entropy(logits_same_branch, target)
    ce_other = torch.nn.functional.cross_entropy(logits_other_branch, target)
    assert torch.isclose(ce_same, ce_other, atol=1e-6)

    hl_same = hl.get_loss(logits_same_branch, one_hot)
    hl_other = hl.get_loss(logits_other_branch, one_hot)
    assert hl_same < hl_other


def test_index_mappings_consistent():
    hierarchy = build_sample_hierarchy()
    hl = HierarchicalLoss(hierarchy, device="cpu")

    assert hl.leaf_to_idx == {
        "leaf1": 0,
        "leaf2": 1,
        "leaf3": 2,
        "leaf4": 3,
    }
    assert hl.mid_to_idx == {
        "mid1": 0,
        "mid2": 1,
        "mid3": 2,
    }
    assert hl.top_to_idx == {
        "top1": 0,
        "top2": 1,
    }

