import pickle
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np

from Analysis.analysis import MalignancyConfusionMatrix, MalignancyBySize
from LIDC.lidcUtils import getAnnotation
from Network.Direct.directArch import directArch
from Network.data_loader import load_nodule_dataset, load_nodule_raw_dataset, prepare_data
from Network.model import miniXception_loader

size = 128
input_shape = (128,128,1)
load = False

# 0     Test
# 1     Validation
# 2     Training
DataSubSet = 2

if   DataSubSet == 0:
    print("Test Set Analysis")
elif DataSubSet == 1:
    print("Validation Set Analysis")
elif DataSubSet == 2:
    print("Training Set Analysis")
else:
    assert False

dataset = load_nodule_raw_dataset()[DataSubSet]
print("Raw Data Loaded: {} entries".format(len(dataset)))

# prepare test data
images_test, labels_test = prepare_data(load_nodule_dataset()[0], classes=2, size=size)
print("Data ready: images({}), labels({})".format(images_test.shape, labels_test.shape))
print("Range = [{},{}]".format(np.min(images_test), np.max(images_test)))

assert len(dataset) == images_test.shape[0]

#model = miniXception(None, (size, size,1),'avg', weights='w_002_37-0.95-0.82.h5')
#compile(model, learning_rate=0.01)

model = directArch(miniXception_loader, input_shape, 2)
#model.summary()
#model.compile()
model.load_weights('w_007_36-0.91-0.86.h5')


start = timer()
try:

    if load:
        pred, Size = pickle.load(open('pred_test.p','br'))
        print("loaded saved dump of predications")
    else:
        Size = [getAnnotation(entry['info']).estimate_diameter() for entry in dataset]
        pred = model.predict(images_test, round=False)

        pickle.dump((pred, Size),open('pred_test.p','bw'))

    MalignancyConfusionMatrix(pred, labels_test[:])
    MalignancyBySize(pred, labels_test[:], Size[:])

    plt.show()

finally:
    total_time = (timer() - start) / 60 / 60
    print("Total runtime is {:.1f} hours".format(total_time))