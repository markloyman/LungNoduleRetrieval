import numpy as np
import matplotlib.pyplot as plt
from Network import FileManager
from scipy.misc import imsave
from PIL import Image

net_type = 'dirD'
dset = 'Valid'
run = '821'

path = r'./Plots/images/'

for config in [0]:
    data = FileManager.Embed(net_type).load(run + 'c{}'.format(config), dset)
    embed, epochs, meta, images, classes, labels, masks, conf, rating_weights, z = data
    for idx, img in enumerate(images):
        filename = path + '{}-{:03}.png'.format(config, idx)
        img = (((img - img.min()) / (img.max() - img.min())) * 255.9).astype(np.uint8)
        Image.fromarray(img.squeeze()).save(filename)