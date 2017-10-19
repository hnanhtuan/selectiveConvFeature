What is it?
===========

This is a Matlab that we provide to implement the methods of our [ACM MM 2017](http://www.acmmm.org/2017/) paper.
 (paper homepage: https://arxiv.org/abs/1707.00809).
This code implements 
a) The framework of produce global image representation
    + Pre-processing: PCA + l2-normalize
    + Masking scheme: MAX-mask/SUM-mask
    + Embedding: Triangular Embedding (Temb) [1] and Fast Function Approximation Embedding (F-FAemb) [2]
    + Aggregating: Democractic pooling [1]
    + Post-processing: rotation and whitening, Power-law normalization.
b) Extract conv. feature of images

@article{tuan2017selective,
  title={Selective Deep Convolutional Features for Image Retrieval},
  author={Tuan, Hoang and Thanh-Toan, Do and Dang-Khoa, Le Tan and Ngai-Man, Cheung},
  booktitle={ACM MultiMedia},
  year={2017}
}

[1] Hervé Jégou and Andrew Zisserman. 2014. Triangulation embedding and democratic aggregation for image search. In CVPR.
[2] Thanh-Toan Do and Ngai-Man Cheung. 2017. Embedding based on function approximation for large scale image search. TPAMI (2017).

Prerequisites
=============
The code include:
- The Yael library (obtained from the website https://gforge.inria.fr/frs/?group_id=2151)
  A copy of this library is included in 'tools' folder.

Usage
=============
1) Modify the parameters in 'opt.m' file appropriately. 
1) Run the following script
>> main

Files
=====================
|README                      | This file|

|main.m                      | The main script for running whole process|
|opt.m                       | The script contains all parameter setting.|

|extract_feature_map/        |Contains files for extracting conv. features. See the README file inside this folder for more information.|


|tools/make.m                |      Script to build the mex file for faemb and temb methods.|

|tools/democratic/           |contains matlab script files for democratic pooling methods.|
|tools/faemb/                |contains matlab script files for F-FAemb method.|
|tools/faemb_mex/            |contains mex files for F-FAemb method.|
|tools/triemb/               |contains matlab script/mex files for Triangular Embedding method.|
|tools/evaluation/           |contains matlab script files for evaluation.|
|tools/yael/                 |contains the yael library|
          
|utils/embedding.m           |Process embedding and aggregating|
|utils/apply_mask.m          |Compute and then apply mask on the conv. features.|
|utils/crop_qim.m            |Crop query images.|
|utils/learn_rn.m            |Learn projection (PCA/whitening) matrix|
|utils/vecpostproc.m         |Apply post-processing|

|data/                       |contains ground truth and data. In case you may want to retrain, please download the provided data and put them in this folder|
|data/workdir/               |contains output files (i.e., parameters, embedded features)|
=====================