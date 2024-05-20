# AE-XAD repo

This repository contains the code of the AE-XAD method. 

## Dataset
Dataset classes are used to load the train and test datasets for model training and evaluation. In the dataset package two classes are contained:
  * **CustomAD**, recommended for images of dimensions of up to $448 \times 448$, in this class no resize is applied to images
  * **MvtecAD**, recommended for images greater than $448 \times 448$, in this class the images are reshaped to $448 \times 448$

Both classes accept channel-first RGB images, images, labels and ground truth that need to be stored in ndarrays saved in the same path. 
An example of how to arrange data can be found in **tools.create_dataset.mvtec**.
To insert a new dataset class it is sufficient to include a new class in the dataset package extending **torch.utils.data.Dataset**

## Networks
Different network architectures are available:
 *  **shallow** - both the encoder and the decoder are one-layered networks. It works with flattenized input which is mapped to a spece of dimension *flat_dim*. Requires: *dim*, *flat_dim*, *latent_dim*
 *  **deep** - both the encoder and the decoder are two-layered networks. It works with flattenized input which is mapped to a spece of dimension *flat_dim*. Requires: *dim*, *flat_dim*, *intermediate_dim*, *latent_dim*
 *  **conv** - convolutional neural network with two image scale down steps. Works with $(3 \times H \times W)$ or $(1 \times H \times W)$ images. Requires: *dim*
 *  **conv_deep** - convolutional neural network with two image scale down steps. Differently from the latter one, it uses $(5 \times 5)$ filters instead of $(3 \times 3)$. Works with $(3 \times H \times W)$ or $(1 \times H \times W)$ images. Requires: *dim*
 *  **pca** - As **shallow** but without bias. Requires: *dim*, *flat_dim*, *latent_dim*


*dim* represents the shape of the $(3 \times H \times W)$ or $(1 \times H \times W)$ images.
*flat_dim* represents the flattenized dimension of the $(3 \times H \times W)$ or $(1 \times H \times W)$ images which, respectively, results to be $3 * H * W$ or $H * W$.
*latent_dim* represents the dimension of the latent space.
*intermediate_dim* intermediate dimention between *flat_dim* and *latent_dim*. it is recommended that its value is major than *latent_dim* and minor then *flat_dim*.

The recomended network for the mvtec dataset is the **conv_deep** one.

## Trainer

The **Trainer** class is devoted to taking care of inizializing and network and carrying out its training. It allows also to extracts the heatmaps, the outlierness scores and the reconstructed images of the the considered test set.
To perform the inizialization the following code can be used:

```
trainer = Trainer(latent_dim, lambda_p, lambda_s, f, path, AE_type, batch_size=None, silent=False, use_cuda=True,
                 loss='aexad', save_intermediate=False, dataset='mvtec')
```

where:
 * *lambda_p* is the anomalous pixels weight, when equal to None, is inizialized according to the number of anomalous pixels
 * *lambda_s* is the anomalous samples weight, when equal to None, is inizialized according to the number of anomalous samples
 * *f* is the function used to map the anomalous samples, the recomended one is `lambda x: torch.where(x>0.5, 0., 1.)`
 * *path* is the path in which the npy files of the dataset are stored
 * *AE_type* is the type of architecture to use
 * *batch_size* is the naumber of samples to include into a batch, if None it is set equal to $1$, defaults to None
 * *silent* if true, training information are not displayed, defaults to False
 * *use_cuda* if True, the network use the cuda memory for the training
 * *loss* the loss to use, possible values are 'aexad' and 'mse', it defaults to 'aexad'
 * *save_intermediate* if True, it saves model weights every $100$ epochs
 * *dataset* the name of the dataset to use, defaults to 'mvtec'

It follows an example with the recomended parameters:

```
trainer = Trainer(latent_dim=None, lambda_p=None, lambda_s=None, f=lambda x: torch.where(x>0.5, 0., 1.), path=path, AE_type='conv_deep', batch_size=16, silent=False, use_cuda=True,
                 loss='aexad', save_intermediate=False, dataset='mvtec')
```
