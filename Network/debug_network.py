import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model

import sys
sys.path.insert(0, 'E:\LungNoduleRetrieval')

from Network.model import  miniXception_loader
from Network.siameseArch import siamArch
from Network.data import load_nodule_dataset, prepare_data
from Network.dataUtils import crop_center
from Analysis.analysis import calc_embedding_statistics
import FileManager

# Setup
# ===================

inp_size = 144
net_size = 128
out_size = 128
input_shape = (net_size, net_size, 1)
res     = 'Legacy'
sample  = 'Normal' #'UniformNC'

# 0     Test
# 1     Validation
# 2     Training
DataSubSet = 2

Weights = FileManager.Weights('siam')

wRuns = ['039']
wEpchs= [1]

run = wRuns[0]
epoch = wEpchs[0]

# Load Data
# =================

images, labels, masks, meta = \
                    prepare_data(load_nodule_dataset(size=inp_size, res=res, sample=sample)[DataSubSet],
                                 categorize=False,
                                 reshuffle=False,
                                 return_meta=True,
                                 verbose=1)

images = np.array([crop_center(im, msk, size=net_size)[0] for im, msk in zip(images, masks)])

# Run
# =================

siam_model = siamArch(miniXception_loader, input_shape, distance='l2', output_size=out_size, normalize=False)
embed_model = siam_model.extract_core(weights=Weights(run=run, epoch=epoch))
embed_model.layers[1].summary()

layer_names = ['block1_conv1', 'block1_conv1_bn', 'block1_conv1_act']
layers = [embed_model.layers[1].get_layer(name).output for name in layer_names]
intermediate_layer_model = Model(inputs=embed_model.layers[1].layers[0].input, outputs=layers)

intermediate_outputs = intermediate_layer_model.predict(images)

for output, name in zip(intermediate_outputs, layer_names):
    calc_embedding_statistics(output, title=name, data_dim=(0,1,2))

print("Plots Ready...")
plt.show()