"""
HPatches Dataset Loader

Supports:
- Triplet sampling for metric learning
- Full image pairs for homography estimation
- Lazy loading for memory efficiency
- Hard negative mining
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import random
import numpy as np
from typing import Optional, Tuple, List, Dict

HPATCHES_ROOT = os.environ.get("HPATCHES_ROOT", "/Data/adrian.garcia/hpatches/hpatches")


def warp_point(H: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    """Warp a point using a homography matrix."""
    pt = np.array([x, y, 1.0])
    x2, y2, z = H @ pt
    return x2 / z, y2 / z


def warp_points(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Warp multiple points using a homography matrix."""
    ones = np.ones((points.shape[0], 1))
    pts_h = np.hstack([points, ones])
    warped = (H @ pts_h.T).T
    return warped[:, :2] / warped[:, 2:3]


def extract_patch(img: np.ndarray, x: float, y: float, size: int = 32) -> Optional[np.ndarray]:
    """Extract a square patch centered at (x, y)."""
    half = size // 2
    x, y = int(x), int(y)
    h, w = img.shape[:2]

    if x - half < 0 or x + half >= w or y - half < 0 or y + half >= h:
        return None

    return img[y - half : y + half, x - half : x + half]


class HPatchesTriplets(Dataset):
    """
    HPatches dataset for triplet learning with lazy loading.
    
    Args:
        root: Path to HPatches dataset
        domain: "illumination", "viewpoint", or "both"
        size: Patch size
        hard_negatives: Use hard negative mining (same sequence)
    """

    def __init__(
        self,
        root: Optional[str] = None,
        domain: str = "illumination",
        size: int = 32,
        hard_negatives: bool = True,
    ):
        self.root = root or HPATCHES_ROOT
        self.size = size
        self.domain = domain
        self.hard_negatives = hard_negatives

        # Determine which sequences to use
        if domain == "illumination":
            prefixes = ["i_"]
        elif domain == "viewpoint":
            prefixes = ["v_"]
        else:  # both
            prefixes = ["i_", "v_"]

        all_seqs = os.listdir(self.root)
        self.sequences = sorted([s for s in all_seqs if any(s.startswith(p) for p in prefixes)])

        # Store metadata only (lazy loading)
        self.items = []
        self.seq_to_indices = {}
        self.homographies = {}
        self.image_paths = {}

        sift = cv2.SIFT_create()

        for seq in self.sequences:
            seq_path = os.path.join(self.root, seq)

            img_files = sorted([f for f in os.listdir(seq_path) if f.endswith(".ppm")])
            if len(img_files) < 2:
                continue

            self.image_paths[seq] = [os.path.join(seq_path, f) for f in img_files]

            # Load homographies
            Hs = []
            for k in range(2, len(img_files) + 1):
                Hfile = os.path.join(seq_path, f"H_1_{k}")
                if os.path.exists(Hfile):
                    Hs.append(np.loadtxt(Hfile))
                else:
                    Hs.append(np.eye(3))
            self.homographies[seq] = Hs

            # Detect keypoints in reference image
            ref_img = cv2.imread(self.image_paths[seq][0])
            if ref_img is None:
                continue
            kps, _ = sift.detectAndCompute(ref_img, None)
            h, w = ref_img.shape[:2]

            self.seq_to_indices[seq] = []

            for kp in kps:
                x1, y1 = kp.pt

                for k, H in enumerate(Hs):
                    x2, y2 = warp_point(H, x1, y1)

                    half = self.size // 2
                    if (x1 - half < 0 or x1 + half >= w or y1 - half < 0 or y1 + half >= h):
                        continue
                    if (x2 - half < 0 or x2 + half >= w or y2 - half < 0 or y2 + half >= h):
                        continue

                    idx = len(self.items)
                    self.items.append((seq, 0, k + 1, x1, y1, x2, y2))
                    self.seq_to_indices[seq].append(idx)

    def __len__(self) -> int:
        return len(self.items)

    def _load_patch(self, seq: str, img_idx: int, x: float, y: float) -> Optional[np.ndarray]:
        """Load an image and extract a patch."""
        img_path = self.image_paths[seq][img_idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        return extract_patch(img, x, y, size=self.size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq, img1_idx, img2_idx, x1, y1, x2, y2 = self.items[idx]

        a_patch = self._load_patch(seq, img1_idx, x1, y1)
        p_patch = self._load_patch(seq, img2_idx, x2, y2)

        # Negative sampling
        if self.hard_negatives:
            same_seq_indices = self.seq_to_indices[seq]
            candidates = [i for i in same_seq_indices if i != idx]
            if candidates:
                neg_idx = random.choice(candidates)
            else:
                neg_idx = random.randint(0, len(self.items) - 1)
        else:
            neg_idx = random.randint(0, len(self.items) - 1)

        neg_seq, neg_img_idx, _, neg_x, neg_y, _, _ = self.items[neg_idx]
        n_patch = self._load_patch(neg_seq, neg_img_idx, neg_x, neg_y)

        if a_patch is None or p_patch is None or n_patch is None:
            return self.__getitem__((idx + 1) % len(self.items))

        to_tensor = lambda x: torch.tensor(x).float().permute(2, 0, 1) / 255.0

        return to_tensor(a_patch), to_tensor(p_patch), to_tensor(n_patch)


class HPatchesSequences(Dataset):
    """
    HPatches dataset returning full image pairs for homography estimation evaluation.
    
    Returns:
        img1: Reference image
        img2: Target image
        H: Ground truth homography from img1 to img2
        seq_name: Sequence name
    """

    def __init__(
        self,
        root: Optional[str] = None,
        domain: str = "illumination",
    ):
        self.root = root or HPATCHES_ROOT
        self.domain = domain

        if domain == "illumination":
            prefixes = ["i_"]
        elif domain == "viewpoint":
            prefixes = ["v_"]
        else:
            prefixes = ["i_", "v_"]

        all_seqs = os.listdir(self.root)
        self.sequences = sorted([s for s in all_seqs if any(s.startswith(p) for p in prefixes)])

        self.pairs = []

        for seq in self.sequences:
            seq_path = os.path.join(self.root, seq)
            img_files = sorted([f for f in os.listdir(seq_path) if f.endswith(".ppm")])

            if len(img_files) < 2:
                continue

            img1_path = os.path.join(seq_path, img_files[0])

            for k in range(2, len(img_files) + 1):
                img2_path = os.path.join(seq_path, img_files[k - 1])
                Hfile = os.path.join(seq_path, f"H_1_{k}")

                if os.path.exists(Hfile):
                    H = np.loadtxt(Hfile)
                else:
                    H = np.eye(3)

                self.pairs.append((img1_path, img2_path, H, seq))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        img1_path, img2_path, H, seq = self.pairs[idx]

        img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2GRAY)

        return img1, img2, H, seq


# In get_triplet_dataloader(), add max_samples parameter:

def get_triplet_dataloader(
    domain: str,
    batch_size: int = 32,
    root: Optional[str] = None,
    num_workers: int = 4,  # CHANGED from 0
    hard_negatives: bool = True,
    max_samples: Optional[int] = None,  # ADD THIS
) -> DataLoader:
    """Get DataLoader for triplet training."""
    from torch.utils.data import Subset
    
    ds = HPatchesTriplets(root=root, domain=domain, hard_negatives=hard_negatives)
    
    # Limit samples if specified
    if max_samples and len(ds) > max_samples:
        indices = np.random.choice(len(ds), max_samples, replace=False)
        ds = Subset(ds, indices)
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  
        drop_last=True,
    )

def get_sequence_dataloader(
    domain: str,
    root: Optional[str] = None,
) -> DataLoader:
    """Get DataLoader for sequence-level evaluation."""
    ds = HPatchesSequences(root=root, domain=domain)
    return DataLoader(ds, batch_size=1, shuffle=False)


# Backwards compatibility
def get_dataloader(domain: str, batch_size: int = 32, **kwargs) -> DataLoader:
    return get_triplet_dataloader(domain, batch_size, **kwargs)
