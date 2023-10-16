import os
import glob
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict
from semantic_aug.few_shot_dataset import FewShotDataset
from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Tuple

ROOT_IMAGE_DIR = ""

class NaturalImages(FewShotDataset):
    
    class_names = [
        "airplane",
        "car",
        "cat",
        "dog",
        "flower",
        "fruit",
        "motorbike",
        "person",
    ]
    
    num_classes: int = len(class_names)
    
    def __init__(self, *args, split: str = "train", seed: int = 0,
                 image_dir: str = ROOT_IMAGE_DIR,
                 examples_per_class: int = None,
                 generative_aug: GenerativeAugmentation = None,
                 synthetic_probability: float = 0.5,
                 use_randaugment: bool = False,
                 image_size: Tuple[int] = (256, 256), **kwargs):
    
        super(NaturalImages, self).__init__(
            *args, examples_per_class=examples_per_class, 
            synthetic_probability=synthetic_probability,
            generative_aug=generative_aug, **kwargs
        )
        
        class_to_images = defaultdict(list)
        search_pattern = os.path.join(ROOT_IMAGE_DIR, "**", "*.jpg")
        paths = glob.glob(search_pattern, recursive=True)
        
        for path in paths:
            subfolder_name = os.path.relpath(os.path.dirname(path), ROOT_IMAGE_DIR)
            class_to_images[subfolder_name].append(path)
            
        class_to_images = dict(class_to_images)
        
        rng = np.random.default_rng(seed)
        class_to_ids = {key: rng.permutation(
            len(class_to_images[key])) for key in self.class_names}
        
        class_to_ids = {key: np.array_split(class_to_ids[key], 2)[0 if split == "train" else 1] for key in self.class_names}
        
        if examples_per_class is not None:
            class_to_ids = {key: ids[:examples_per_class] 
                            for key, ids in class_to_ids.items()}
        
        self.class_to_images = {
            key: [class_to_images[key][i] for i in ids]
            for key, ids in class_to_ids.items()
        }
        
        self.all_images = sum([
            self.class_to_images[key]
            for key in self.class_names], []
        )
        
        self.all_labels = [
            i for i, key in enumerate(
                self.class_names
            ) for _ in self.class_to_images[key]
        ]
        
        if use_randaugment:
            train_transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.RandAugment(),
                    transforms.ToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Lambda(lambda x: x.expand(3, *image_size)),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]
                    )
                ]
            )
            
        else:
            train_transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15.0),
                    transforms.ToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Lambda(lambda x: x.expand(3, *image_size)),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]
                    )
                ]
            )
        
        val_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Lambda(lambda x: x.expand(3, *image_size)),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ]
        )
        
        self.transform = {"train": train_transform, "val": val_transform}[split]
        
    def __len__(self):
        return len(self.all_images)
    
    def get_image_by_idx(self, idx: int) -> Image.Image:
        return Image.open(self.all_images[idx]).convert("RGB")
    
    def get_label_by_idx(self, idx: int) -> int:
        return self.all_labels[idx]
    
    def get_metadata_by_idx(self, idx: int) -> dict:
        return dict(name=self.class_names[self.all_labels[idx]])
    
        
        