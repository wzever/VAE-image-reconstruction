import time
import torch
from model import *

# l = time.localtime(time.time())

# print(l)

# h=time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time()))

# print(h, type(h))

# print(h[5:-3])

vae = VAE(3, 256)

vae.load_state_dict(torch.load('05-19_10_29-1.pt', map_location={'cuda:1': 'cpu'}))

print(vae)