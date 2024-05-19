# AE-XAD repo

This repository contains the code of AE-XAD method. 

## Dataset
Dataset classes are used to load the train and test datasets for model training and evaluation. In the dataset package two classes are contained:
  * **CustomAD**, recommended for images of dimensions of up to 448x448, in this class no resize is applied to images
  * **MvtecAD**, recommended for images greater than 448x448, in this class the images are reshaped to 448x448

Both classes accept channel-first RGB images, images, labels and ground truth need to be stored in ndarrays saved in the same path. 
An example of how to arrange data can be found in **tools.create_dataset.mvtec**.
To insert a new dataset class it is sufficient to include a new class in the dataset package extending **torch.utils.data.Dataset**
