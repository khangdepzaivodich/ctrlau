"""
Dataset loaders for the two CtrlAU benchmark protocols.
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
from config import DISFA_AUS, BP4D_AUS, AU_INDEX, ModelConfig


def _default_transform(train):
    """The image preprocessing used by the reference MultiviewSymAU code."""
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _read_list_file(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip().replace("\\", "/") for line in handle if line.strip()]


def _read_label_matrix(path, expected_rows, expected_cols, dataset_name):
    labels = np.loadtxt(path, dtype=np.float32)
    if labels.ndim == 1:
        labels = labels.reshape(1, -1)
    if labels.shape != (expected_rows, expected_cols):
        raise ValueError(
            f"{dataset_name} label file {path} has shape {labels.shape}; "
            f"expected ({expected_rows}, {expected_cols})"
        )
    if not np.isfinite(labels).all() or not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError(f"{dataset_name} labels must be finite binary values: {path}")
    return torch.from_numpy(labels)


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


def _resolve_disfa_frame_paths(subject_image_dir, label_frames):
    """Verify exact frame correspondence before accepting a numbering offset.

    DISFA annotations are one-based. Extracted images may keep those IDs or
    use zero-based IDs. Never infer the offset from an individual missing
    file: compare the complete image and annotation frame sets instead.
    """
    if not os.path.isdir(subject_image_dir):
        raise FileNotFoundError(f"DISFA subject image directory not found: {subject_image_dir}")
    image_paths = {}
    with os.scandir(subject_image_dir) as entries:
        for entry in entries:
            stem, extension = os.path.splitext(entry.name)
            if not entry.is_file() or extension.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            if not stem.isdecimal():
                raise ValueError(f"DISFA raw images require numeric frame names: {entry.path}")
            frame_id = int(stem)
            if frame_id in image_paths:
                raise ValueError(
                    f"Ambiguous DISFA image frame {frame_id}: "
                    f"{image_paths[frame_id]} and {entry.path}"
                )
            image_paths[frame_id] = entry.path

    label_frames = set(label_frames)
    if not label_frames:
        raise ValueError(f"DISFA subject has no annotated frames: {subject_image_dir}")
    image_frames = set(image_paths)
    for offset in (0, -1):
        expected_frames = {frame + offset for frame in label_frames}
        if image_frames == expected_frames:
            if offset:
                print(
                    f"[Dataset] Verified zero-based images in {subject_image_dir}: "
                    f"{len(image_frames)} frames; annotation frame k -> image k-1."
                )
            return {frame: image_paths[frame + offset] for frame in label_frames}

    missing_direct = sorted(label_frames - image_frames)
    missing_zero_based = sorted({frame - 1 for frame in label_frames} - image_frames)
    raise FileNotFoundError(
        f"DISFA frame alignment failed in {subject_image_dir}: "
        f"{len(label_frames)} annotated frames and {len(image_frames)} images. "
        f"Neither exact one-based nor zero-based numbering matches the complete frame set. "
        f"Missing image IDs (one-based): {missing_direct[:8]}; "
        f"missing image IDs (zero-based): {missing_zero_based[:8]}. "
        "Check for missing/extra images or incorrect extraction; no frames were skipped."
    )


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
        
        self.transform = transform if transform is not None else _default_transform(train)

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

                raw_imgs = _read_list_file(img_file)
                raw_labels = _read_label_matrix(
                    label_file, len(raw_imgs), len(DISFA_AUS), "DISFA"
                )

                for idx, rel_path in enumerate(raw_imgs):
                    full_img_path = os.path.join(self.img_root, rel_path)
                    if not os.path.isfile(full_img_path):
                        raise FileNotFoundError(f"DISFA image listed but not found: {full_img_path}")
                    self.samples.append((rel_path, full_img_path))
                    self.labels.append(raw_labels[idx])

                self.labels = torch.stack(self.labels)

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
                    raise FileNotFoundError(f"DISFA subject label directory not found: {subj_label_dir}")

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
                        raise FileNotFoundError(
                            f"DISFA subject {subj} is missing the label file for AU{au}"
                        )

                    au_labels[au] = {}
                    with open(label_file, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.replace("\t", ",").split(",")
                            if len(parts) < 2:
                                raise ValueError(f"Malformed DISFA label line in {label_file}: {line!r}")
                            frame_num = int(parts[0])
                            intensity = int(parts[1])
                            if frame_num <= 0 or intensity < 0 or intensity > 5:
                                raise ValueError(
                                    f"DISFA labels must have frame>0 and intensity in [0,5]; "
                                    f"got {frame_num}, {intensity} in {label_file}"
                                )
                            au_labels[au][frame_num] = intensity

                    if frame_nums is None:
                        frame_nums = set(au_labels[au].keys())
                    elif frame_nums != set(au_labels[au]):
                        raise ValueError(
                            f"DISFA AU annotation frame sets disagree for {subj}, AU{au}: "
                            f"{label_file}. Frame alignment cannot be verified."
                        )

                if frame_nums is None:
                    continue

                for au in DISFA_AUS:
                    if au in au_labels:
                        self.labels_cache[(subj, au)] = au_labels[au]

                frame_paths = _resolve_disfa_frame_paths(
                    os.path.join(self.img_root, subj), frame_nums
                )
                for fn in sorted(frame_nums):
                    self.samples.append((subj, fn, frame_paths[fn]))

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
                intensity = self.labels_cache[(subj, au)][frame_num]
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
            if torch.any(occur_rates <= 0):
                missing = [DISFA_AUS[i] for i, rate in enumerate(occur_rates) if rate <= 0]
                raise ValueError(f"DISFA training split has no positive samples for AU(s): {missing}")
            raw_weights = 1.0 / occur_rates
            normalized_weights = raw_weights / raw_weights.sum() * len(DISFA_AUS)
            return normalized_weights, occur_rates
        else:
            counts = torch.zeros(len(DISFA_AUS), dtype=torch.float32)
            for subj, fn, _ in self.samples:
                for au in DISFA_AUS:
                    i = AU_INDEX[au]
                    if self.labels_cache[(subj, au)][fn] >= self.intensity_threshold:
                        counts[i] += 1.0

            occur_rates = counts / float(num_samples)
            if torch.any(occur_rates <= 0):
                missing = [DISFA_AUS[i] for i, rate in enumerate(occur_rates) if rate <= 0]
                raise ValueError(f"DISFA training split has no positive samples for AU(s): {missing}")
            raw_weights = 1.0 / occur_rates
            normalized_weights = raw_weights / raw_weights.sum() * len(DISFA_AUS)
            return normalized_weights, occur_rates


class BP4DDataset(Dataset):
    """BP4D loader for the official three-fold prepared list protocol.

    The prepared BP4D lists contain one binary occurrence label per AU.  BP4D
    does not provide DISFA-style intensity files in this protocol, so the
    returned ``au_intensities`` field is explicitly the same binary occurrence
    tensor; no intensity value is invented or inferred.
    """

    def __init__(self, data_root, transform=None, train=True, fold=1):
        if fold not in (1, 2, 3):
            raise ValueError(f"BP4D fold must be 1, 2, or 3; got {fold}")

        self.data_root = _resolve_data_root(data_root)
        self.train = train
        self.fold = fold
        self.au_ids = tuple(BP4D_AUS)
        self.img_root = _find_dir_ci(self.data_root, "img")
        if not os.path.isdir(self.img_root):
            raise FileNotFoundError(f"BP4D image directory not found: {self.img_root}")
        self.list_root = _find_dir_ci(self.data_root, "list")
        if not os.path.isdir(self.list_root):
            raise FileNotFoundError(f"BP4D list directory not found: {self.list_root}")
        self.transform = transform if transform is not None else _default_transform(train)

        prefix = "train" if train else "test"
        img_file = os.path.join(
            self.list_root, f"BP4D_{prefix}_img_path_fold{fold}.txt"
        )
        label_file = os.path.join(
            self.list_root, f"BP4D_{prefix}_label_fold{fold}.txt"
        )
        if not os.path.isfile(img_file) or not os.path.isfile(label_file):
            raise FileNotFoundError(
                "BP4D requires the official prepared list files for the selected split: "
                f"{img_file} and {label_file}"
            )

        rel_paths = _read_list_file(img_file)
        labels = _read_label_matrix(label_file, len(rel_paths), len(self.au_ids), "BP4D")
        self.samples = []
        for rel_path in rel_paths:
            full_path = os.path.join(self.img_root, rel_path)
            if not os.path.isfile(full_path):
                raise FileNotFoundError(f"BP4D image listed but not found: {full_path}")
            self.samples.append((rel_path, full_path))
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, image_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        au_labels = self.labels[idx]
        return {
            "image": image,
            "au_labels": au_labels,
            "au_intensities": au_labels,
            "sample_id": rel_path,
        }

    def calculate_class_weights(self):
        if len(self) == 0:
            raise ValueError("BP4D split contains no samples; class weights are undefined")
        occurrence_rates = self.labels.mean(dim=0)
        if torch.any(occurrence_rates <= 0):
            missing = [self.au_ids[i] for i, rate in enumerate(occurrence_rates) if rate <= 0]
            raise ValueError(f"BP4D training split has no positive samples for AU(s): {missing}")
        inverse_rates = 1.0 / occurrence_rates
        normalized_weights = inverse_rates / inverse_rates.sum() * len(self.au_ids)
        return normalized_weights, occurrence_rates
