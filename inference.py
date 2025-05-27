import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dataset import get_transform
from model import ConvNeXtRoadSegmentation

def visualize_inference(image_path, mask_path, model_path, model_arch, input_size, device):
    """
    Run inference on a single image and visualize input, predicted mask, and ground truth mask.
    
    Args:
        image_path (str): Path to input image (e.g., data/training/images/satImage_001.png).
        mask_path (str): Path to ground truth mask (e.g., data/training/groundtruth/satImage_001.png).
        model_path (str): Path to trained model checkpoint.
        model_arch (str): ConvNeXt architecture (e.g., Base).
        input_size (int): Input size for model (e.g., 256).
        device (str): Device to run inference on (cuda or cpu).
    """
    # Set device
    device = torch.device(device)
    
    # Load transforms
    image_transform, mask_transform = get_transform(input_size=input_size, is_train=False)
    
    # Load image and mask
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')  # Grayscale
    
    # Apply transforms
    image = image_transform(image)  # [3, input_size, input_size]
    mask = mask_transform(mask)    # [1, input_size, input_size]
    
    # Process mask
    mask = torch.squeeze(mask, dim=0)  # [input_size, input_size]
    mask = (mask > 0.5).long()        # Binary {0, 1}
    
    # Add batch dimension
    image = image.unsqueeze(0)  # [1, 3, input_size, input_size]
    
    # Load model
    model = ConvNeXtRoadSegmentation(model_arch=model_arch, input_size=input_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Run inference
    with torch.no_grad():
        image = image.to(device)
        output = model(image)  # [1, 2, input_size, input_size]
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]  # [input_size, input_size]
    
    # Visualize
    plt.figure(figsize=(15, 5))
    
    # Input image
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(image[0].cpu().permute(1, 2, 0).numpy())  # Convert [3, H, W] to [H, W, 3]
    plt.axis('off')
    
    # Predicted mask
    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(pred, cmap='gray')  # 0 (black) for background, 1 (white) for road
    plt.axis('off')
    
    # Ground truth mask
    plt.subplot(1, 3, 3)
    plt.title("Ground Truth Mask")
    plt.imshow(mask.cpu().numpy(), cmap='gray')  # 0 (black) for background, 1 (white) for road
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize road segmentation prediction for a single image.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to input image (e.g., data/training/images/satImage_001.png)")
    parser.add_argument('--mask_path', type=str, required=True, help="Path to ground truth mask (e.g., data/training/groundtruth/satImage_001.png)")
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained model checkpoint (e.g., best_convnext_road_segmentation.pth)")
    parser.add_argument('--model_arch', type=str, default="Base", choices=['Tiny', 'Small', 'Base', 'Large', 'XLarge'])
    parser.add_argument('--input_size', type=int, default=256, help="Input size for model (e.g., 256)")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    visualize_inference(
        image_path=args.image_path,
        mask_path=args.mask_path,
        model_path=args.model_path,
        model_arch=args.model_arch,
        input_size=args.input_size,
        device=args.device
    )

if __name__ == "__main__":
    main()