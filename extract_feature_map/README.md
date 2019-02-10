What is it?
===========

This is a Matlab script to extract conv. features for various datasets, 
including Oxford5k, Paris6k, Holidays, UKB, and Flickr100k

Prerequisites
=============

The prerequisites are:
* MatConvNet MATLAB toolbox 1.0-beta25

* Images of Oxford5k and Paris6k datasets: http://www.robots.ox.ac.uk/~vgg/data/
* Images of Holidays datasets: http://lear.inrialpes.fr/~jegou/data.php
* Images of UKB datasets: http://www.mediafire.com/file/duhba0actb5c90d/UKB.tar (Nister D, Stewenius H (2006) Scalable recognition with a vocabulary tree. In: CVPR)
* Images of Flickr100k: http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/flickr100k.html

* Pre-trained CNN models: VGG16. 
  The mat files containing the models can be downloaded at [MatConvnet website](http://www.vlfeat.org/matconvnet/pretrained/) or [my backup file](https://www.mediafire.com/file/rx1liu6xl4ii9l0/imagenet-vgg-verydeep-16.mat).

Usage
=============
1. Modify the parameters in 'main.m' file appropriately:
* *lid*:          The index of conv. layer to extract features.
* *max_img_dim*:  Resize to have max(W, H)=max_img_dim
* *baseDir*:      The directory contains subfolders, which contains images
2. Select the dataset to extract features.
3. Run the following script
```
main
```

Files
==============
|Filename|Description|
|---|---|
|README   |                   This file|
|main.m    |                  The main script to extract conv. features|
|extract_feature.m|           The function that execute the forward pass to get the conv. feature.|
|crop_qim.m        |          Function to crop image based on provided bounding box for Oxford5k and Paris6k datasets.|
|random_flickr5k.m 	|		Randomly select 5000 images for training holidays datasets.|
|||
|gnd_oxford5k.mat    |        File contains all ground truth information of Oxford5k dataset|
|gnd_paris6k.mat      |       File contains all ground truth information of Paris6k dataset|