"""
DISFA Dataset loader.
Supports BOTH:
1. List-based dataset (official MultiviewSymAU format with data/list/DISFA_train_...txt)
2. Raw directory dataset (DISFA_Data with img/ and ActionUnit_Labels/ folders)

Auto-resolves case-sensitivity and nested subdirectories (common on Kaggle / Linux).
Transforms match the original MultiviewSymAU repo exactly.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from config import DISFA_AUS, AU_INDEX, ModelConfig


def _resolve_data_root(data_root):
    """Auto-detect nested directory if data_root doesn't directly contain img/list/labels."""
    if not os.path.isdir(data_root):
        return data_root
    
    # Check directly
    for item in ["img", "list", "List", "ActionUnit_Labels", "actionunit_labels"]:
        if os.path.isdir(os.path.join(data_root, item)):
            return data_root
            
    # Search 1 level of subdirectories (e.g. /kaggle/input/disfa/DISFA)
    try:
        for sub in sorted(os.listdir(data_root)):
            sub_path = os.path.join(data_root, sub)
            if os.path.isdir(sub_path):
                for item in ["img", "list", "List", "ActionUnit_Labels", "actionunit_labels"]:
                    if os.path.isdir(os.path.join(sub_path, item)):
                        print(f"[Dataset] Auto-detected nested dataset folder: {sub_path}")
                        return sub_path
    except Exception:
        pass
    return data_root


def _find_dir_ci(parent, target_name):
    """Find subdirectory case-insensitively."""
    if not os.path.isdir(parent):
        return os.path.join(parent, target_name)
    target_clean = target_name.lower().replace("_", "")
    for name in os.listdir(parent):
        if name.lower().replace("_", "") == target_clean:
            return os.path.join(parent, name)
    return os.path.join(parent, target_name)


class DISFADataset(Dataset):
    """
    DISFA dataset supporting both list-based and raw folder layouts.
    """
    
    def __init__(self, data_root, subjects=None, transform=None, train=True, 
                 intensity_threshold=2, fold=None):
        """
        Args:
            data_root: path to DISFA or DISFA_Data directory
            subjects: list of subject IDs (e.g., ['SN001', 'SN002']). None = all.
            transform: torchvision transforms for images. If None, uses original repo defaults.
            train: if True, use training transforms (with ColorJitter). If False, use val transforms.
            intensity_threshold: threshold for binarizing AU intensities (default: 2)
            fold: fold number (1, 2, or 3) for loading official list files if available.
        """
        self.data_root = _resolve_data_root(data_root)
        self.intensity_threshold = intensity_threshold
        self.train = train
        self.fold = fold
        
        # Image root (case-insensitive)
        self.img_root = _find_dir_ci(self.data_root, "img")
        if not os.path.isdir(self.img_root):
            self.img_root = _find_dir_ci(self.data_root, "images")
            
        # Label root (case-insensitive)
        self.label_root = _find_dir_ci(self.data_root, "ActionUnit_Labels")
        
        # List root
        self.list_root = _find_dir_ci(self.data_root, "list")
        
        # Transforms matching MultiviewSymAU/utils.py
        if transform is None:
            if train:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
                ])
        else:
            self.transform = transform

        self.samples = []
        self.labels = []
        self.labels_cache = {}
        self.mode = "raw"

        # ------------------------------------------------------------
        # Priority 1: Check for official list files (MultiviewSymAU format)
        # ------------------------------------------------------------
        has_list_mode = False
        if os.path.isdir(self.list_root) and self.fold is not None:
            prefix = "train" if train else "test"
            img_candidates = [
                os.path.join(self.list_root, f"DISFA_{prefix}_img_path_fold{self.fold}.txt"),
                os.path.join(self.list_root, f"DISFA_{prefix}_img_path_fold_{self.fold}.txt"),
                os.path.join(self.list_root, f"DISFA_{prefix}_path_fold{self.fold}.txt"),
            ]
            label_candidates = [
                os.path.join(self.list_root, f"DISFA_{prefix}_label_fold{self.fold}.txt"),
                os.path.join(self.list_root, f"DISFA_{prefix}_label_fold_{self.fold}.txt"),
            ]
            img_file = next((f for f in img_candidates if os.path.isfile(f)), None)
            label_file = next((f for f in label_candidates if os.path.isfile(f)), None)

            if img_file and label_file:
                has_list_mode = True
                self.mode = "list"
                print(f"[Dataset] Loading from official list files ({prefix}, fold {self.fold}):")
                print(f"  Img list:   {img_file}")
                print(f"  Label list: {label_file}")

                raw_imgs = [line.strip().replace("\\", "/") for line in open(img_file, "r").readlines() if line.strip()]
                raw_labels = np.loadtxt(label_file)

                # Ensure labels 2D
                if raw_labels.ndim == 1:
                    raw_labels = raw_labels.reshape(1, -1)

                for idx, rel_path in enumerate(raw_imgs):
                    full_img_path = os.path.join(self.img_root, rel_path)
                    self.samples.append((rel_path, full_img_path))
                    self.labels.append(raw_labels[idx])

                self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)

        # ------------------------------------------------------------
        # Priority 2: Raw folder layout (ActionUnit_Labels/SNxxx/...)
        # ------------------------------------------------------------
        if not has_list_mode and os.path.isdir(self.label_root):
            self.mode = "raw"
            # Discover subjects
            if subjects is None:
                subjects = sorted([
                    d for d in os.listdir(self.label_root)
                    if os.path.isdir(os.path.join(self.label_root, d))
                ])

            for subj in subjects:
                subj_label_dir = os.path.join(self.label_root, subj)
                if not os.path.isdir(subj_label_dir):
                    continue

                au_labels = {}
                frame_nums = None

                for au in DISFA_AUS:
                    candidates = [
                        os.path.join(subj_label_dir, f"{subj}_au{au}.txt"),
                        os.path.join(subj_label_dir, f"{subj}_AU{au}.txt"),
                        os.path.join(subj_label_dir, f"{subj}_au_{au}.txt"),
                        os.path.join(subj_label_dir, f"{subj}_AU_{au}.txt"),
                    ]
                    label_file = next((f for f in candidates if os.path.isfile(f)), None)
                    if not label_file:
                        continue

                    au_labels[au] = {}
                    with open(label_file, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.replace("\t", ",").split(",")
                            frame_num = int(parts[0])
                            intensity = int(parts[1])
                            au_labels[au][frame_num] = intensity

                    if frame_nums is None:
                        frame_nums = set(au_labels[au].keys())
                    else:
                        frame_nums = frame_nums.intersection(au_labels[au].keys())

                if frame_nums is None:
                    continue

                for au in DISFA_AUS:
                    if au in au_labels:
                        self.labels_cache[(subj, au)] = au_labels[au]

                for fn in sorted(frame_nums):
                    img_path = os.path.join(self.img_root, subj, f"{fn}.png")
                    if os.path.exists(img_path):
                        self.samples.append((subj, fn, img_path))

        # Diagnostic if 0 samples
        if len(self.samples) == 0:
            print(f"\n[ERROR] DISFADataset found 0 samples in: {self.data_root}")
            print(f"  Checked img_root:   {self.img_root} (exists: {os.path.isdir(self.img_root)})")
            print(f"  Checked list_root:  {self.list_root} (exists: {os.path.isdir(self.list_root)})")
            print(f"  Checked label_root: {self.label_root} (exists: {os.path.isdir(self.label_root)})")
            if os.path.isdir(self.data_root):
                print(f"  Available in data_root: {os.listdir(self.data_root)}")
            print()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.mode == "list":
            rel_path, full_img_path = self.samples[idx]
            image = Image.open(full_img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            au_labels = self.labels[idx]
            return {
                "image": image,
                "au_labels": au_labels,
                "au_intensities": au_labels,
            }
        else:
            subj, frame_num, img_path = self.samples[idx]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)

            au_labels = torch.zeros(len(DISFA_AUS), dtype=torch.float32)
            au_intensities = torch.zeros(len(DISFA_AUS), dtype=torch.float32)

            for au in DISFA_AUS:
                i = AU_INDEX[au]
                intensity = self.labels_cache.get((subj, au), {}).get(frame_num, 0)
                au_intensities[i] = intensity / 5.0
                au_labels[i] = 1.0 if intensity >= self.intensity_threshold else 0.0

            return {
                "image": image,
                "au_labels": au_labels,
                "au_intensities": au_intensities,
                "subject": subj,
                "frame": frame_num,
            }

    def calculate_class_weights(self):
        """
        Calculate fold-specific AU class weights matching MultiviewSymAU / SymGraphAU.
        """
        num_samples = len(self)
        if num_samples == 0:
            return None, None

        if self.mode == "list":
            counts = self.labels.sum(dim=0)
            occur_rates = counts / float(num_samples)
            occur_rates_clamped = torch.clamp(occur_rates, min=1e-5)
            raw_weights = 1.0 / occur_rates_clamped
            normalized_weights = raw_weights / raw_weights.sum() * len(DISFA_AUS)
            return normalized_weights, occur_rates
        else:
            counts = torch.zeros(len(DISFA_AUS), dtype=torch.float32)
            for subj, fn, _ in self.samples:
                for au in DISFA_AUS:
                    i = AU_INDEX[au]
                    if self.labels_cache.get((subj, au), {}).get(fn, 0) >= self.intensity_threshold:
                        counts[i] += 1.0

            occur_rates = counts / float(num_samples)
            occur_rates_clamped = torch.clamp(occur_rates, min=1e-5)
            raw_weights = 1.0 / occur_rates_clamped
            normalized_weights = raw_weights / raw_weights.sum() * len(DISFA_AUS)
            return normalized_weights, occur_rates
