# MixBranchNet

Author: Yanduo Li, Lin Chen*

Email:  chenlin21@xmu.edu.cn chenlin0430@163.com   

Affiliations:
Department of Electronic Science, Xiamen University, Xiamen, Fujian, China

Python version: 3.11.5

Matlab version: R2023b

---

## Overview

This toolbox contains demo for the following papers:

**Li Y, Chen Z, Huang Y, Fan Y, Meng Y, Chen L\*.**  
MixBranchNet: A Task-Adaptive Network for Glioma Segmentation and Genotype Prediction by Exploiting Spatio-spectral Correlations in CEST MRI.
*Magn Reson Med.2025*;

Welcome your comments and suggestions.

---
## Setup
1.Clone this repository:
```bash
git clone https://github.com/LinChenMRI/MixBranchNet.git
```
2.Install the required dependencies:
```bash
pip install -r requirements.txt
```
3.Ensure you have the necessary dataset available in the data/ folder.

4.For Matlab, make sure you have **R2023b** or later and the necessary toolboxes for image processing.


---

## Pretrained Model Access

Due to file size limitations on GitHub, the pretrained model weights are hosted on **Zenodo**:

- **Zenodo record:** [https://doi.org/10.5281/zenodo.17453687](https://doi.org/10.5281/zenodo.17453687)

**Included files:**
- `IDH_genotype.pth` 
- `IDH_genotype.pth` 
- `Segment.pth` 
---

## Repository Structure

```text
MixBranchNet/
├─ data/  
│  ├─ Savedata.mat                           # Data for Segmentation                          
│  ├─ IDH_patch/                             # Extracted Patch data for IDH genotype prediction
│  ├─ MGMT_patch/                            # Extracted Patch data for MGMT genotype prediction
│  └─ test_case/                             # Example raw data 
│     ├─ Mask/                                   # Brain and tumor mask for evaluation
│     ├─ MRI_image/                              # The CE-T1w and FLAIR MRI
│     ├─ CEST.mat                                # The preprocessed CEST images with 41 frequency offsets
│     ├─ M0_Image.mat                            # The M0 image acquired at 100 ppm
│     └─ Case_information.txt                    # The case information 
│
├─ mat_code/                              # Core Matlab functions 
│  ├─ m1_Segmentation_data_generation.m      
│  ├─ m2_Segmentation_post_processing.m   
│  ├─ m3_Segmentation_visualization.m     
│  ├─ m4_Genotype_patch_extraction.m        
│  ├─ m5_Genotype_patch_aggregation.m        
│  └─ m6_Patient_result.m                       
│
├─ model/                                 # Deep learning models
│  ├─ MixingBlock.py                          # The Mixing Block implementation
│  └─ MixUNet.py                              # The MixBranchNet
│
├─ weights/                               # Pretrained weights (full weights on Zenodo)
│  ├─ IDH_genotype.pth                
│  ├─ MGMT_genotype.pth               
│  └─ Segment.pth
│
├─ result/                                # Example results
│  ├─ IDH_genotype/                           # IDH genotype prediction results and maps
│  ├─ MGMT_genotype/                          # MGMT genotype prediction results and maps
│  └─ Segmentation/                           # Segmentation results
│
├─ dice_coefficient_loss.py               # Custom Dice coefficient loss function
├─ distributed_utils.py                   # Utilities for distributed training
├─ my_dataset.py                          # Custom dataset handling
├─ utils.py                               # Initial model parameters and Helper functions
├─ test_segmentation.py                   # Script for testing segmentation
├─ test_genotype.py                       # Script for testing genotype prediction
├─ requirements.txt                       # Python dependencies
└─ README.md
```
---
## Usage
You can run the code for testing and training using the provided scripts.

## Segmentation Testing

1.Runs Matlab code: `m1_Segmentation_data_generation.m` using the bundled example data in `./data/test_case` and 
generate **Savedata.mat** to `./data`

2.Runs Segmentation Script using the generated **Savedata.mat** and saves segmentation outputs to `./result/Segmentation`:
```bash
python test_segmentation.py 
```

3.Runs Matlab code: `m2_Segmentation_post_processing.m` to process the segmentation outputs and 
generate **pred_mask_post_processing.mat** to `./result/Segmentation`. 
We can visualize the segmentation results through Matlab code: `m3_Segmentation_visualization.m`.

## Genotype Prediction Testing
4.Runs Matlab code: `m4_Genotype_patch_extraction.m` using the segmented tumor mask **pred_mask_post_processing.mat** 
in `./result/Segmentation` to generate patches to `./data/IDH_patch` or `./data/MGMT_patch`. 

5.Runs Genotype Prediction Script using the generated patch and saves outputs to `./result/IDH_genotype` or `./result/MGMT_genotype`:
```bash
python test_genotype.py 
```

6.Runs Matlab code: `m5_Genotype_patch_aggregation.m` to aggregate the predicted patches and 
generate **Result_Map.mat** to `./result/IDH_genotype` or `./result/MGMT_genotype`. 
We can visualize the genotype prediction results through Matlab code: `m6_Patient_result.m`.


---
## Contact

Welcome your comments and suggestions.

**Last updated:** Oct 27, 2025

