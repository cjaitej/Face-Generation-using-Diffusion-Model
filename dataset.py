import os
import csv
import random
import torch
from torch.utils.data import Dataset
from PIL import Image

# Subset of CelebA's 40 binary attributes used for conditioning (see spec).
SELECTED_ATTRIBUTES = [
    "Male", "Young", "Smiling",
    "Black_Hair", "Blond_Hair", "Brown_Hair", "Bald",
    "Mustache", "Goatee",
    "Eyeglasses",
]


# These four are mutually exclusive in reality (a face has one hair state), so random combos
# below never activate more than one at a time.
_HAIR_ATTRIBUTES = {"Black_Hair", "Blond_Hair", "Brown_Hair", "Bald"}


def random_attribute_batch(n, seed=None):
    """Generate n random, non-contradictory attribute combos for preview sampling.

    Each sample gets 1-4 randomly chosen attributes set to 1 (at most one hair state among
    them). Used for periodic training previews so they show the model's general range rather
    than the same fixed set of faces every time. `seed` makes a given call reproducible without
    touching the global RNG."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        k = rng.randint(1, 4)
        pool = SELECTED_ATTRIBUTES[:]
        rng.shuffle(pool)
        chosen = []
        used_hair = False
        for name in pool:
            if len(chosen) >= k:
                break
            if name in _HAIR_ATTRIBUTES:
                if used_hair:
                    continue
                used_hair = True
            chosen.append(name)
        rows.append({name: 1 for name in chosen})
    return build_attribute_batch(rows)


def build_attribute_vector(requested, n=1):
    """Turn a {attribute_name: 0/1} dict into an (n, len(SELECTED_ATTRIBUTES)) tensor.
    Unspecified attributes default to 0; unknown names raise rather than silently doing nothing."""
    unknown = set(requested) - set(SELECTED_ATTRIBUTES)
    if unknown:
        raise KeyError(f"Unknown attribute(s): {sorted(unknown)}. Valid names: {SELECTED_ATTRIBUTES}")
    values = [requested.get(name, 0) for name in SELECTED_ATTRIBUTES]
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0).repeat(n, 1)


def build_attribute_batch(requested_list):
    """Stack one attribute vector per dict, giving a batch of distinct conditioning combos."""
    return torch.cat([build_attribute_vector(r) for r in requested_list], dim=0)


def load_attributes(attr_file, selected_attributes=SELECTED_ATTRIBUTES):
    """Parse CelebA's list_attr_celeba.csv into {filename: 0/1 tensor} for the selected attributes."""
    attributes = {}
    with open(attr_file, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        indices = [header.index(name) for name in selected_attributes]
        for row in reader:
            if not row:
                continue
            filename = row[0]
            selected = [(int(row[i]) + 1) // 2 for i in indices]  # -1/1 -> 0/1
            attributes[filename] = torch.tensor(selected, dtype=torch.float32)
    return attributes


def list_celeba_filenames(attr_file):
    """Read just the filenames (column 0) out of list_attr_celeba.csv.

    The attribute file already enumerates every image in the dataset, so this is the list of
    all image filenames for free -- no need to scan the (possibly network-backed) image directory.
    """
    with open(attr_file, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader)  # header
        return [row[0] for row in reader if row]


class FaceDataset(Dataset):
    def __init__(self, data_folder, transform, attr_file=None):
        self.transform = transform
        with open(data_folder, 'r') as f:
            self.images = [line for line in f.read().split('\n') if line]
        self.attributes = load_attributes(attr_file) if attr_file else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        path = self.images[i]
        image = Image.open(path)
        image = self.transform(image)
        if self.attributes is None:
            return image
        filename = os.path.basename(path)
        attr = self.attributes.get(filename, torch.zeros(len(SELECTED_ATTRIBUTES)))
        return image, attr


if __name__ == '__main__':
    dataset = FaceDataset('data_set.txt')
    for i in dataset:
        print(i.shape)
