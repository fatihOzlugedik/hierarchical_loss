import numpy as np
import pandas as pd
import h5py
import torch
from dataset_hiarachical import MllDataset


def create_dummy_dataset(tmp_path):
    dataset_dir = tmp_path / "data"
    fold_dir = dataset_dir / "fold_1"
    fold_dir.mkdir(parents=True)

    df = pd.DataFrame({"patient_files": ["patient0"], "diagnose": [2]})
    df.to_csv(fold_dir / "train.csv", index=False)

    features = np.random.rand(5, 3).astype(np.float32)
    with h5py.File(dataset_dir / "patient0.h5", "w") as f:
        f.create_dataset("features", data=features)

    return dataset_dir, features


def test_mll_dataset_get_item(tmp_path):
    dataset_dir, features = create_dummy_dataset(tmp_path)

    ds = MllDataset(str(dataset_dir), current_fold=1, split="train", aug_im_order=False)
    assert len(ds) == 1

    bag, label, pid = ds[0]
    assert pid == "patient0"
    assert isinstance(bag, np.ndarray)
    assert bag.shape == features.shape
    assert torch.equal(label, torch.tensor([2], dtype=torch.long))

    dist = ds.get_class_distribution()
    assert dist == {2: 1}
