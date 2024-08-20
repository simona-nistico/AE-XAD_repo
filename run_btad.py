from argparse import ArgumentParser

import torch

from tools.argument_parser import RealDSParser
import os
import shutil
import numpy as np
from tools.create_dataset import extract_dataset
from trainer import Trainer


def compute_weights(Y, GT):
    weights = np.ones_like(Y)
    tot_pixels = GT[Y == 1].sum(axis=(1, 2))
    w_anoms = 1 / tot_pixels
    n_norm = np.sum(1 - Y)
    norm_f = n_norm / w_anoms.sum()
    return float(norm_f)


if __name__ == '__main__':
    # Parse run configuration, it allows to change all the model parameters and the dataset used as well
    conf = RealDSParser()
    conf = conf(ArgumentParser())
    args = conf.parse_args()

    # Create the dataset
    data_path = os.path.join('data', args.ds)
    X_train, Y_train, X_test, Y_test, GT_train, GT_test = extract_dataset(data_path, args.na)

    if args.dp is None:
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        if args.l == 'mse':
            # If we use the standard autoencoder the train set have no anomalies
            np.save(open(os.path.join(data_path, 'X_train.npy'), 'wb'), X_train[Y_train == 0])
            np.save(open(os.path.join(data_path, 'Y_train.npy'), 'wb'), Y_train[Y_train == 0])
            np.save(open(os.path.join(data_path, 'GT_train.npy'), 'wb'), GT_train[Y_train == 0])
        else:
            np.save(open(os.path.join(data_path, 'X_train.npy'), 'wb'), X_train)
            np.save(open(os.path.join(data_path, 'Y_train.npy'), 'wb'), Y_train)
            np.save(open(os.path.join(data_path, 'GT_train.npy'), 'wb'), GT_train)

        np.save(open(os.path.join(data_path, 'X_test.npy'), 'wb'), X_test)
        np.save(open(os.path.join(data_path, 'Y_test.npy'), 'wb'), Y_test)
        np.save(open(os.path.join(data_path, 'GT_test.npy'), 'wb'), GT_test)

    # Create the path for the results
    ret_path = os.path.join('results', args.ds, str(args.na))
    if not os.path.exists(ret_path):
        os.makedirs(ret_path)

    # Mapping function for the loss
    def f(x):
        return torch.where(x >= 0.5, 0.0, 1.0)

    if args.l == 'aexad_norm':
        lambda_s = compute_weights(Y_train, GT_train)
    else:
        lambda_s = None


    # Create the trainer (latent dim is declared for compatibility but never used) and train the model
    trainer = Trainer(latent_dim=64, lambda_p=None, lambda_s=lambda_s, f=f, path=data_path, AE_type=args.net, batch_size=16,
                      use_cuda=args.cuda, loss=args.l, save_intermediate=args.si, dataset=args.ds)
    trainer.train(epochs=args.e, save_path=ret_path)
    trainer.save_weights(os.path.join(ret_path, f'{args.l}_model_{args.net}.pt'))

    # Collect results on the test dataset
    heatmaps, scores, gtmaps, labels = trainer.test()
    np.save(open(os.path.join(ret_path, f'{args.l}_htmaps_{args.net}.npy'), 'wb'), heatmaps)
    np.save(open(os.path.join(ret_path, f'{args.l}_scores_{args.net}.npy'), 'wb'), scores)
    np.save(open(os.path.join(ret_path, 'aexad_gtmaps_norm.npy'), 'wb'), gtmaps)
    np.save(open(os.path.join(ret_path, 'aexad_labels_norm.npy'), 'wb'), np.array(labels))

