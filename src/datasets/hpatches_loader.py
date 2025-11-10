import torch
from torch.utils.data import Dataset, DataLoader
import cv2, os, numpy as np

class HPatchesPatches(Dataset):
    def __init__(self, root, domain="illumination", size=32):
        self.root = os.path.join(root, domain)
        self.pairs = self._collect_pairs()
        self.size = size

    def _collect_pairs(self):
        pairs = []
        # load sample patch triplets (anchor, pos, neg)
        for scene in os.listdir(self.root):
            folder = os.path.join(self.root, scene)
            patches = [os.path.join(folder, f) for f in os.listdir(folder)]
            if len(patches) >= 3:
                pairs.append(patches[:3])
        return pairs

    def __getitem__(self, idx):
        a, p, n = [cv2.imread(f, 0)/255.0 for f in self.pairs[idx]]
        a, p, n = [torch.tensor(x).unsqueeze(0).float() for x in [a, p, n]]
        return a, p, n

    def __len__(self):
        return len(self.pairs)

def get_dataloader(domain, batch_size=32):
    ds = HPatchesPatches("./data/hpatches", domain=domain)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

def get_meta_tasks(domains, meta_batch_size=4):
    return [tuple(get_dataloader(d, batch_size=8)) for d in domains[:meta_batch_size]]
