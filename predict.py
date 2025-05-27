import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2
from dataset import RoadSegmentationDataset, get_transform
from model import ConvNeXtRoadSegmentation

def predict():
    parser = argparse.ArgumentParser(description="Generate predictions for EPFL ML Road Segmentation test set.")
    parser.add_argument('--data_root', type=str, required=True, help="Root directory of the dataset (containing test/)")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model")
    parser.add_argument('--model_arch', type=str, default="Base", choices=['Tiny', 'Small', 'Base', 'Large', 'XLarge'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--output_size', type=int, default=400, help="Output size for submission (e.g., 400 for EPFL)")
    parser.add_argument('--output_dir', type=str, default="submission", help="Directory to save predictions")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Transform
    transform, mask_transform = get_transform(input_size=args.input_size, is_train=False)
    
    # Test dataset (images in subdirectories)
    test_dataset = RoadSegmentationDataset(
        image_dir=os.path.join(args.data_root, "test"),
        transform=transform,
        mask_transform=mask_transform,
        input_size=args.input_size,
        is_test=True
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Load model
    model = ConvNeXtRoadSegmentation(model_arch=args.model_arch, input_size=args.input_size)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Generate predictions
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)  # [batch_size, 2, input_size, input_size]
            if args.output_size != args.input_size:
                outputs = F.interpolate(outputs, size=(args.output_size, args.output_size), mode='nearest')
            preds = torch.argmax(outputs, dim=1).cpu().numpy()  # [batch_size, output_size, output_size]
            
            # Save predictions as PNGs
            for j, pred in enumerate(preds):
                pred = (pred * 255).astype(np.uint8)  # Convert 0,1 to 0,255
                # Name files as test_001.png, test_002.png, etc.
                filename = f"test_{(i * args.batch_size + j + 1):03d}.png"
                cv2.imwrite(os.path.join(args.output_dir, filename), pred)
    
    print(f"Predictions saved to {args.output_dir}")

if __name__ == "__main__":
    predict()