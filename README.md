# Road Semantic Segmentation

This project compares three models ConvNeXt, C-UNet, and SegFormer for semantic road segmentation using high-resolution satellite imagery. It includes training, fine-tuning, and inference notebooks.

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

Used the EPFL Road Segmentation Dataset. You can download it from:  
https://www.kaggle.com/datasets/timothlaborie/roadsegmentation-boston-losangeles?select=images

Fine tuned the SegFormer model using the satellite-road-segmentation kaggle Dataset. You can download it from:
https://www.kaggle.com/datasets/timothlaborie/roadsegmentation-boston-losangeles?select=images

## Results

| Model        | Dataset    | Accuracy | Loss   |
|--------------|------------|----------|--------|
| SegFormer-B2 | EPFL       | 0.9658   | 0.0539 |
| SegFormer-B2 | Boston/LA  | 0.8782   | 0.1532 |
| C-UNet       | EPFL       | 0.9404   | 0.412  |
| ConvNeXt     | EPFL       | 0.8732   | 0.5523 |


---

> Developed by Dareen Deeb, Rand Ahed, Aseel Sharsheer  
> University of Jordan, AI Department
