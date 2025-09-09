# Ultrasound Spleen Length Measurement (Simplified Code)

This repository provides a simplified implementation based on the paper:

> **Yuan, Z., Puyol-Antón, E., Jogeesvaran, H., Smith, N., Inusa, B., & King, A. P. (2022). Deep learning-based quality-controlled spleen assessment from ultrasound images. *Biomedical Signal Processing and Control, 76*, 103724.**  
> [https://doi.org/10.1016/j.bspc.2022.103724](https://doi.org/10.1016/j.bspc.2022.103724)

⚠️ This code is a **lightweight version** of the full framework. It focuses on U-Net segmentation and length measurement, without the full quality-control pipeline described in the article.

---

## Repository Structure

- **`unet.py`** – standard U-Net implementation  
- **`unet_extrablock.py`** – modified, deeper U-Net with extra encoder/decoder blocks  
- **`T_Unet.py`** –  example training script 
- **`train_unet.py`** – example training script  
- **`Inf_unet.py`** – example training script  
- **`utils.py`** – helper functions:
  - data loaders  
  - augmentations  
  - length measurement methods (`len_measurement_all`, `len_measurement_VarPCA`, `len_measurement_points`)  

---

## Length Measurement

Three strategies are implemented inside **`utils.py`**:

- **`len_measurement_all`** – projects all spleen pixels onto the PCA axis.  
- **`len_measurement_VarPCA`** – shifts the PCA axis across the contour region to find the longest axis.  
- **`len_measurement_points`** – computes the maximum distance between all contour points.  

---

## Usage

### Training
```bash
python train_unet.py
