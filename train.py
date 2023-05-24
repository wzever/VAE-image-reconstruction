import torch
from torch import optim
import torch.nn.functional as F
from torchvision.utils import save_image
from model import *
import argparse
import os
import numpy as np
from main import get_train_loader
import random
import time

parser = argparse.ArgumentParser(description="VAE for face reconstruction")
parser.add_argument('--result_dir', type=str, default='./VAEResult', metavar='DIR', help='output saving directory')
parser.add_argument('--save_dir', type=str, default='./pretrained', metavar='N', help='model saving directory')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size for training')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs for training')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed for reproduction')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--z_dim', type=int, default=256, metavar='N', help='the dim of latent embedding z')
parser.add_argument('--input_channel', type=int, default=3, metavar='N', help='input channel')
parser.add_argument('--train_num', type=int, default=13233, help='Number of images loaded for training')
parser.add_argument('--cuda', default=1, type=int, help='gpu index for acceleration')
args = parser.parse_args()

device = torch.device(f"cuda:{args.cuda}" if args.cuda >= 0 and torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def loss_function(x_hat, x, mu, log_var):
    # reconstruction loss.
    BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')

    # KL-divergence
    KLD = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())

    loss = BCE + KLD
    return loss, BCE, KLD

def main():
    batch_size = args.batch_size
    num_imgs = args.train_num
    model = VAE(3, args.z_dim).to(device) 

    optimizer = optim.Adam(model.parameters(), lr=args.lr)  
    scheduler = optim.lr_scheduler.StepLR(optimizer, 25, 0.95)

    start_epoch = 0

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_set = get_train_loader(batch_size, num_imgs)

    # Step 4: 开始迭代
    loss_epoch = []
    print(f'Start training on {device}...')
    for epoch in range(start_epoch, args.epochs):
        t1 = time.time()
        loss_batch = []
        for batch_index, x in enumerate(train_set):
            x = x[0]
            x = x.to(device)

            # 前向传播
            x_hat, mu, log_var = model(x)
            loss, BCE, KLD = loss_function(x_hat, x, mu, log_var)  
            loss_batch.append(loss.item()) 

            # 后向传播
            optimizer.zero_grad()  
            loss.backward()
            optimizer.step() 

            t = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time()))[5:-3]
            if (epoch + 1) % 10 == 0 and batch_index == 0:
                x_concat = torch.cat([x.view(-1, 3, 224, 224), x_hat.view(-1, 3, 224, 224)], dim=3)
                save_image(x_concat, f'./{args.result_dir}/{t}-{epoch + 1}.png')
                if (epoch + 1) % 100 == 0:
                    torch.save(model.state_dict(), f'{args.save_dir}/{t}-{epoch + 1}.pt')

        print('Epoch [{}/{}]: Total-loss = {:.4f}, BCE-Loss = {:.4f}, KLD-loss = {:.4f}'
                    .format(epoch + 1, args.epochs, 
                            loss.item() / batch_size, BCE.item() / batch_size,
                            KLD.item() / batch_size))

        loss_epoch.append(np.sum(loss_batch) / num_imgs) 
        t2 = time.time()
        scheduler.step()
        rest = (args.epochs - epoch - 1) * (t2-t1)
        print(f'time = {t2-t1:.2f}, Estimated remaining: {int(rest // 3600)} h {int((rest % 3600) // 60)} m {int((rest % 3600) % 60)} s')

    return loss_epoch

if __name__ == '__main__':
    set_seed(args.seed)
    loss_epoch = main()