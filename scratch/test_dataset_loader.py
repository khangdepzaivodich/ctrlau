import os
import sys
import torch
import numpy as np
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import DISFADataset

def test_list_mode():
    print("Testing List Mode (MultiviewSymAU standard)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        list_dir = os.path.join(tmpdir, "list")
        img_dir = os.path.join(tmpdir, "img", "SN001")
        os.makedirs(list_dir)
        os.makedirs(img_dir)
        
        # Create dummy image
        from PIL import Image
        img = Image.new("RGB", (224, 224), color="blue")
        img.save(os.path.join(img_dir, "0.png"))
        img.save(os.path.join(img_dir, "1.png"))
        
        # Create list files for fold 2
        with open(os.path.join(list_dir, "DISFA_train_img_path_fold2.txt"), "w") as f:
            f.write("SN001/0.png\nSN001/1.png\n")
        with open(os.path.join(list_dir, "DISFA_train_label_fold2.txt"), "w") as f:
            f.write("0 1 0 0 1 0 1 0\n1 0 0 1 0 0 0 1\n")
            
        with open(os.path.join(list_dir, "DISFA_test_img_path_fold2.txt"), "w") as f:
            f.write("SN001/0.png\n")
        with open(os.path.join(list_dir, "DISFA_test_label_fold2.txt"), "w") as f:
            f.write("0 1 0 0 1 0 1 0\n")

        # Test loading train
        ds_train = DISFADataset(data_root=tmpdir, fold=2, train=True)
        assert len(ds_train) == 2, f"Expected 2, got {len(ds_train)}"
        sample = ds_train[0]
        assert sample["image"].shape == (3, 224, 224)
        assert sample["au_labels"].shape == (8,)
        assert sample["au_labels"][1] == 1.0
        print("  Train dataset loaded:", len(ds_train))
        
        # Test class weights
        weights, rates = ds_train.calculate_class_weights()
        assert weights is not None
        assert weights.shape == (8,)
        print("  Class weights calculated successfully!")

        # Test loading val
        ds_val = DISFADataset(data_root=tmpdir, fold=2, train=False)
        assert len(ds_val) == 1, f"Expected 1, got {len(ds_val)}"
        print("  Val dataset loaded:", len(ds_val))
        print("List Mode: PASSED!")

def test_raw_mode():
    print("\nTesting Raw Folder Mode (DISFA_Data)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        label_dir = os.path.join(tmpdir, "ActionUnit_Labels", "SN001")
        img_dir = os.path.join(tmpdir, "img", "SN001")
        os.makedirs(label_dir)
        os.makedirs(img_dir)
        
        from PIL import Image
        img = Image.new("RGB", (224, 224), color="green")
        img.save(os.path.join(img_dir, "0.png"))
        
        from config import DISFA_AUS
        for au in DISFA_AUS:
            with open(os.path.join(label_dir, f"SN001_au{au}.txt"), "w") as f:
                f.write("0,3\n")
                
        ds = DISFADataset(data_root=tmpdir, subjects=["SN001"], train=True)
        assert len(ds) == 1
        sample = ds[0]
        assert sample["au_labels"][0] == 1.0  # intensity 3 >= 2 -> 1.0
        print("  Raw dataset loaded:", len(ds))
        print("Raw Mode: PASSED!")

def test_nested_auto_discovery():
    print("\nTesting Nested Auto-Discovery (/kaggle/input/disfa/DISFA)...")
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_dir = os.path.join(tmpdir, "DISFA")
        list_dir = os.path.join(nested_dir, "list")
        img_dir = os.path.join(nested_dir, "img", "SN001")
        os.makedirs(list_dir)
        os.makedirs(img_dir)
        
        from PIL import Image
        img = Image.new("RGB", (224, 224), color="red")
        img.save(os.path.join(img_dir, "0.png"))
        
        with open(os.path.join(list_dir, "DISFA_train_img_path_fold1.txt"), "w") as f:
            f.write("SN001/0.png\n")
        with open(os.path.join(list_dir, "DISFA_train_label_fold1.txt"), "w") as f:
            f.write("0 1 0 0 1 0 1 0\n")
            
        ds = DISFADataset(data_root=tmpdir, fold=1, train=True)
        assert len(ds) == 1
        print("Nested Auto-Discovery: PASSED!")

if __name__ == "__main__":
    test_list_mode()
    test_raw_mode()
    test_nested_auto_discovery()
    print("\n>>> ALL DATASET TESTS PASSED 100%! <<<")
