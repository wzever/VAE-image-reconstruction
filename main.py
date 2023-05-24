import os
import numpy as np
from dataset import get_image_path, load_image, normalize
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from model import VAE
from tqdm import tqdm
import random

def get_train_loader(batch_size, train_num):
    '''
    Read images and prepare dataloader for training.
    '''
    image_paths = get_image_path('dataset')
    X_train = TensorDataset(torch.from_numpy(load_image(image_paths[:train_num])))
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
    return train_loader

def test_reconstruct(batch_size, num_batches, pretrained, shuffle=False):
    '''
    Save reconstructed images from randomly picked batches.
    '''
    if not os.path.exists('reconstResult'):
        os.makedirs('reconstResult')
    image_paths = get_image_path('dataset')
    X_test = TensorDataset(torch.from_numpy(load_image(image_paths[8000:12000])))
    test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=shuffle)

    vae = VAE(3, 256)
    vae.load_state_dict(torch.load(pretrained, map_location={'cuda:1': 'cpu'}))
    
    for i, img in enumerate(test_loader):
        img = img[0]
        reconst_img = vae.reconstruct(img)
        concat_img = torch.cat([img, reconst_img], dim=3)
        save_image(concat_img, f'reconstResult/reconstruct_{i}.png')
        # save_image(img, f'original_{i}.png')
        # save_image(reconst_img, f'reconst_{i}.png')
        if i == num_batches - 1:
            break

def gen_feature_fusion(x, y, alpha, pretrained):
    '''
    Generated new images from linear interpolation of latent features.
    '''
    vae = VAE(3, 256)
    vae.load_state_dict(torch.load(pretrained, map_location={'cuda:1': 'cpu'}))

    z1, z2 = vae.get_z(x), vae.get_z(y)
    z_mix = z1 * alpha + z2 * (1 - alpha)
    out = vae.generate_from_z(z_mix)
    return out

def test_generation(batch_size, num_batches, pretrained):
    if not os.path.exists('fusionResult'):
        os.makedirs('fusionResult')

    image_paths = get_image_path('dataset')
    X_test = TensorDataset(torch.from_numpy(load_image(image_paths[3000:8000])))
    test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=True)

    loader_iter = iter(test_loader)

    for k in range(num_batches):
        X = next(loader_iter)[0]
        Y = next(loader_iter)[0]
        im_show = Y
        # feature fusion
        for i in range(11):
            alpha = i / 10
            new_img = gen_feature_fusion(X, Y, alpha, pretrained)
            im_show = torch.cat([im_show, new_img], dim=3)
        im_show = torch.cat([im_show, X], dim=3)
        save_image(im_show, f'fusionResult/fusion_{k}.png', nrow=1)

if __name__ == '__main__':
    pretrained = 'pretrained/05-23_01_02-1000.pt'

    # randomly chosen batches of reconstructed images will be saved in ./reconstResult
    print('Start reconstructing...')
    test_reconstruct(batch_size=64, num_batches=3, pretrained=pretrained)
    # randomly chosen batches of fused images will be saved in ./fusionResult
    print('Done!')
    print('Start genrating, and this may take a while...')
    test_generation(batch_size=8, num_batches=3, pretrained=pretrained)
    print('Done!')