import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import jaccard_score
from dataset import RoadSegmentationDataset, get_transform
from model import ConvNeXtRoadSegmentation, CombinedLoss, ConvNeXt_Archs

class RoadSegmentationTrainer:
    def __init__(self, model, train_loader, val_loader, device, learning_rate=1e-4, num_epochs=50, model_save_path="best_model.pth"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = CombinedLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        self.num_epochs = num_epochs
        self.best_iou = 0.0
        self.model_save_path = model_save_path

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        for images, masks in self.train_loader:
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        return running_loss / len(self.train_loader.dataset)

    def evaluate(self):
        self.model.eval()
        iou_scores = []
        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                masks = masks.cpu().numpy()
                iou = jaccard_score(masks.flatten(), preds.flatten(), average='binary')
                iou_scores.append(iou)
        return np.mean(iou_scores)

    def save_model(self, iou):
        if iou > self.best_iou:
            self.best_iou = iou
            torch.save(self.model.state_dict(), self.model_save_path)
            print(f"Saved best model with IoU: {iou:.4f}")

    def train(self):
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            val_iou = self.evaluate()
            self.scheduler.step()
            
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {train_loss:.4f}, Val IoU: {val_iou:.4f}")
            self.save_model(val_iou)

def main():
    parser = argparse.ArgumentParser(description="Train a ConvNeXt model for EPFL ML Road Segmentation.")
    parser.add_argument('--data_root', type=str, required=True, help="Root directory of the dataset (containing training/images and training/groundtruth)")
    parser.add_argument('--model_arch', type=str, default="Base", choices=ConvNeXt_Archs.keys())
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--model_save_path', type=str, default="best_convnext_road_segmentation.pth")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Data transforms
    train_transform, train_mask_transform = get_transform(input_size=args.input_size, is_train=True)
    val_transform, val_mask_transform = get_transform(input_size=args.input_size, is_train=False)
    
    # Dataset
    train_dataset = RoadSegmentationDataset(
        image_dir=os.path.join(args.data_root, "train/images"),
        mask_dir=os.path.join(args.data_root, "train/masks"),
        transform=train_transform,
        mask_transform=train_mask_transform,
        input_size=args.input_size
    )
    
    # Train/validation split
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.8 * dataset_size)
    train_indices, val_indices = indices[:split], indices[split:]
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(train_indices), num_workers=4
    )
    val_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(val_indices), num_workers=4
    )
    
    # Model and Trainer
    model = ConvNeXtRoadSegmentation(model_arch=args.model_arch, input_size=args.input_size)
    trainer = RoadSegmentationTrainer(
        model, train_loader, val_loader, device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        model_save_path=args.model_save_path
    )
    
    trainer.train()

if __name__ == "__main__":
    main()