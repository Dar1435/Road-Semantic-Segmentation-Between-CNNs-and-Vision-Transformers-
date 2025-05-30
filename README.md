# Road Semantic Segmentation

This project compares three models—ConvNeXt, C-UNet, and SegFormer—for semantic road segmentation using high-resolution satellite imagery. It includes training, fine-tuning, and inference notebooks.

## 📁 Contents

- `SegFormer_implmentation.ipynb`: Original SegFormer training
- `Fine_Tuned_SegFormer.ipynb`: Fine-tuning SegFormer on a new dataset
- `Light_C-Unet_implementation.ipynb`: C-UNet implementation
- `model.py`: convnext model
- `train2.py`: training file for convnext
- `predict.py`: prediction method for convnext
- `inference.py`: inference implementaion for convnext
- `dataset.py`: dataset manipulation for convnext 


## 📂 Dataset

We used the EPFL Road Segmentation Dataset. You can download it from:  
https://www.kaggle.com/datasets/timothlaborie/roadsegmentation-boston-losangeles?select=images

We fine tuned the SegFormer model using the satellite-road-segmentation kaggle Dataset. You can download it from:
https://www.kaggle.com/datasets/timothlaborie/roadsegmentation-boston-losangeles?select=images

## Results

| Model     | Accuracy | Dice Score | F1 Score |
| --------- | -------- | ---------- | -------- |
| SegFormer | 96.58%   | 0.9625     | 0.9629   |
| C-UNet    | \~85.4%  | \~0.854    | \~0.854  |
| ConvNeXt  | \~87.8%  | \~0.8826   | \~0.8834 |


---

> Developed by Dareen Deeb, Rand Ahed, Aseel Sharsheer  
> University of Jordan, AI Department
