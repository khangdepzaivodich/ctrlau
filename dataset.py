"""
DISFA Dataset loader.
Loads images and AU intensity labels (0-5 scale, binarized at threshold >= 2).
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from config import DISFA_AUS, AU_INDEX, ModelConfig


class DISFADataset(Dataset):
    """
    DISFA dataset.
    Structure:
        DISFA_Data/
            img/SN001/0.png, 1.png, ...
            ActionUnit_Labels/SN001/SN001_au1.txt, SN001_au12.txt, ...
    
    Each AU label file has lines: frame_num,intensity (0-5).
    We binarize: intensity >= 2 -> 1, else 0.
    """
    
    def __init__(self, data_root, subjects=None, transform=None, intensity_threshold=2):
        """
        Args:
            data_root: path to DISFA_Data directory
            subjects: list of subject IDs (e.g., ['SN001', 'SN002']). None = all.
            transform: torchvision transforms for images
            intensity_threshold: threshold for binarizing AU intensities
        """
        self.data_root = data_root
        self.img_root = os.path.join(data_root, "img")
        self.label_root = os.path.join(data_root, "ActionUnit_Labels")
        self.intensity_threshold = intensity_threshold
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((ModelConfig.img_size, ModelConfig.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transform
        
        # Discover subjects
        if subjects is None:
            subjects = sorted([
                d for d in os.listdir(self.label_root)
                if os.path.isdir(os.path.join(self.label_root, d))
            ])
        
        # Build sample list: (subject_id, frame_num)
        self.samples = []
        self.labels_cache = {}   # (subject_id, au_num) -> {frame_num: intensity}
        
        for subj in subjects:
            # Load all AU labels for this subject
            subj_label_dir = os.path.join(self.label_root, subj)
            au_labels = {}
            frame_nums = None
            
            for au in DISFA_AUS:
                label_file = os.path.join(subj_label_dir, f"{subj}_au{au}.txt")
                if not os.path.exists(label_file):
                    continue
                
                au_labels[au] = {}
                with open(label_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(",")
                        frame_num = int(parts[0])
                        intensity = int(parts[1])
                        au_labels[au][frame_num] = intensity
                
                if frame_nums is None:
                    frame_nums = set(au_labels[au].keys())
                else:
                    frame_nums = frame_nums.intersection(au_labels[au].keys())
            
            if frame_nums is None:
                continue
            
            # Cache labels
            for au in DISFA_AUS:
                if au in au_labels:
                    self.labels_cache[(subj, au)] = au_labels[au]
            
            # Add samples
            for fn in sorted(frame_nums):
                img_path = os.path.join(self.img_root, subj, f"{fn}.png")
                if os.path.exists(img_path):
                    self.samples.append((subj, fn))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        subj, frame_num = self.samples[idx]
        
        # Load image
        img_path = os.path.join(self.img_root, subj, f"{frame_num}.png")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Load AU labels (binarized)
        au_labels = torch.zeros(len(DISFA_AUS), dtype=torch.float32)
        au_intensities = torch.zeros(len(DISFA_AUS), dtype=torch.float32)
        
        for au in DISFA_AUS:
            i = AU_INDEX[au]
            intensity = self.labels_cache.get((subj, au), {}).get(frame_num, 0)
            au_intensities[i] = intensity / 5.0  # normalize to [0, 1]
            au_labels[i] = 1.0 if intensity >= self.intensity_threshold else 0.0
        
        return {
            "image": image,              # (3, H, W)
            "au_labels": au_labels,      # (NUM_AUS,) binary
            "au_intensities": au_intensities,  # (NUM_AUS,) in [0, 1]
            "subject": subj,
            "frame": frame_num,
        }
