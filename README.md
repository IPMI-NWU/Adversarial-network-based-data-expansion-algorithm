# An algorithm based on the adversarial network to expand the data of chest X-rays

We propose an algorithm based on the adversarial network to expand the data 
of chest X-rays, aiming at the problem that low data volume of chest X-rays segmentation 
dataset leads to overfitting of the network and poor performance of target segmentation.

The algorithm is different from the common image generation algorithms for one-to-one 
mapping. It decomposes an image into content and style codes, and then recombines the 
content codes of the image with various style codes in the target domain to generate a new 
image that can present diversity in style. Experimental results show that the annotation can 
be shared between the original and generated images after expanding the segmentation 
datasets. Compared with the training strategy that only contains the traditional data 
expansion, with the addition of data expanded by this algorithm, the segmentation 
algorithm improves the DSC and Jaccard metrics of the anterior ribs by 2.64% and 3.55%, 
respectively.

# Requirements
Some important required packages include:
* torch == 2.3.0
* torchvision == 0.18.0
* Python == 3.10.14
* numpy == 1.26.4