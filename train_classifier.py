from semantic_aug.datasets.coco import COCODataset
from semantic_aug.datasets.spurge import SpurgeDataset
from semantic_aug.datasets.imagenet import ImageNetDataset
from semantic_aug.augmentations.real_guidance import RealGuidance
from semantic_aug.augmentations.textual_inversion import TextualInversion
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from itertools import product
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed

import pandas as pd
import numpy as np
import random
import os
from tqdm import trange


datasets = {"spurge": SpurgeDataset, "imagenet": ImageNetDataset, "coco": COCODataset}


def run_experiment(examples_per_class, seed=0, 
                   dataset="spurge", aug="real-guidance", 
                   num_synthetic=100, iterations_per_epoch=200, 
                   num_epochs=50, batch_size=32,
                   strength: float = 0.5, 
                   guidance_scale: float = 7.5, 
                   synthetic_probability: float = 0.5, 
                   model_path: str = "CompVis/stable-diffusion-v1-4",
                   prompt: str = "a drone image of a brown field"):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if aug == "real-guidance":

        aug = RealGuidance(
            model_path=model_path, 
            prompt=prompt,
            strength=strength, 
            guidance_scale=guidance_scale)

    elif aug == "textual-inversion":

        aug = TextualInversion(
            "embeddings/absent_101022.bin", 
            "embeddings/apparent_101022.bin",
            model_path=model_path, 
            prompt=prompt,
            strength=strength, 
            guidance_scale=guidance_scale)

    elif aug == "none":

        synthetic_probability = 0.0
        num_synthetic = 0
        aug = None

    train_dataset = datasets[dataset](
        split="train", examples_per_class=examples_per_class, 
        synthetic_probability=synthetic_probability, 
        generative_aug=aug, seed=seed)

    if num_synthetic > 0 and aug is not None:
        train_dataset.generate_augmentations(num_synthetic)

    train_sampler = torch.utils.data.RandomSampler(
        train_dataset, replacement=True, 
        num_samples=batch_size * iterations_per_epoch)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, 
        sampler=train_sampler, num_workers=4)

    val_dataset = datasets[dataset](split="val", seed=seed)

    val_sampler = torch.utils.data.RandomSampler(
        val_dataset, replacement=True, 
        num_samples=batch_size * iterations_per_epoch)

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, 
        sampler=val_sampler, num_workers=4)

    model = ClassificationModel(train_dataset.num_classes).cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    records = []

    for epoch in trange(num_epochs, desc="Training Classifier"):

        model.train()

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_size = 0.0

        for image, label in train_dataloader:
            image, label = image.cuda(), label.cuda()

            logits = model(image)
            prediction = logits.argmax(dim=1)
            loss = F.cross_entropy(logits, label).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.detach().cpu().numpy() * logits.shape[0]
            epoch_accuracy += (prediction == label).float().sum().cpu().numpy()
            epoch_size += float(image.shape[0])

        training_loss = epoch_loss / epoch_size
        training_accuracy = epoch_accuracy / epoch_size

        model.eval()

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_size = 0.0

        for image, label in val_dataloader:
            image, label = image.cuda(), label.cuda()

            logits = model(image)
            prediction = logits.argmax(dim=1)
            loss = F.cross_entropy(logits, label).mean()

            epoch_loss += loss.detach().cpu().numpy() * logits.shape[0]
            epoch_accuracy += (prediction == label).float().sum().cpu().numpy()
            epoch_size += float(image.shape[0])

        validation_loss = epoch_loss / epoch_size
        validation_accuracy = epoch_accuracy / epoch_size

        records.append(dict(
            seed=seed, 
            examples_per_class=examples_per_class,
            epoch=epoch, 
            value=training_loss, 
            metric="Loss", 
            split="Training"
        ))

        records.append(dict(
            seed=seed, 
            examples_per_class=examples_per_class,
            epoch=epoch, 
            value=validation_loss, 
            metric="Loss", 
            split="Validation"
        ))

        records.append(dict(
            seed=seed, 
            examples_per_class=examples_per_class,
            epoch=epoch, 
            value=training_accuracy, 
            metric="Accuracy", 
            split="Training"
        ))

        records.append(dict(
            seed=seed, 
            examples_per_class=examples_per_class,
            epoch=epoch, 
            value=validation_accuracy, 
            metric="Accuracy", 
            split="Validation"
        ))
            
    return records


class ClassificationModel(nn.Module):
    
    def __init__(self, num_classes: int):
        
        super(ClassificationModel, self).__init__()
        
        self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.out = nn.Linear(2048, num_classes)
        
    def forward(self, image):
        
        x = image
        
        with torch.no_grad():

            x = self.base_model.conv1(x)
            x = self.base_model.bn1(x)
            x = self.base_model.relu(x)
            x = self.base_model.maxpool(x)

            x = self.base_model.layer1(x)
            x = self.base_model.layer2(x)
            x = self.base_model.layer3(x)
            x = self.base_model.layer4(x)

            x = self.base_model.avgpool(x)
            x = torch.flatten(x, 1)
        
        return self.out(x)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Few-Shot Baseline")

    parser.add_argument("--logdir", type=str, default="few_shot_combined")
    parser.add_argument("--checkpoint", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--prompt", type=str, default="a photo of a {name}")

    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--synthetic-probability", type=float, default=0.5)

    parser.add_argument("--iterations-per-epoch", type=int, default=200)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--num-synthetic", type=int, default=20)
    parser.add_argument("--num-trials", type=int, default=4)
    parser.add_argument("--examples-per-class", nargs='+', default=[1, 5, 10, 15, 20, 25])
    
    parser.add_argument("--aug", type=str, default="real-guidance", 
                        choices=["real-guidance", "textual-inversion", "none"])
    
    parser.add_argument("--dataset", type=str, default="spurge", 
                        choices=["spurge", "imagenet", "coco"])
    
    args = parser.parse_args()

    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except KeyError:
        rank, world_size = 0, 1

    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    print(f'Initialized process {rank} / {world_size}')
    os.makedirs(args.logdir, exist_ok=True)

    all_trials = []

    options = product(range(args.num_trials), args.examples_per_class)
    options = np.array(list(options))
    options = np.array_split(options, world_size)[rank]

    for seed, examples_per_class in options.tolist():

        all_trials.extend(run_experiment(
            examples_per_class, seed=seed, 
            dataset=args.dataset, aug=args.aug,
            num_synthetic=args.num_synthetic, 
            iterations_per_epoch=args.iterations_per_epoch,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            model_path=args.checkpoint,
            prompt=args.prompt,
            strength=args.strength, 
            guidance_scale=args.guidance_scale, 
            synthetic_probability=args.synthetic_probability))

        path = f"results_{seed}_{examples_per_class}.csv"
        path = os.path.join(args.logdir, path)

        pd.DataFrame.from_records(all_trials).to_csv(path)
        print(f"[rank {rank}] n={examples_per_class} saved to: {path}")