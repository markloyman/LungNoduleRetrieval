import os
import numpy as np
import random
import matplotlib.pyplot as plt
from Network.Siamese.DataGenSiam import DataGeneratorSiam
from Network.Direct.DataGenDirect import DataGeneratorDir
# for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1337)
random.seed(1337)

data_augment_params = {'max_angle': 0, 'flip_ratio': 0.0, 'crop_stdev': 0.1, 'epoch': 0}

if False:
    gen = DataGeneratorSiam(data_size=144, model_size=128, res='Legacy', sample='Normal', batch_sz=64,
                            do_augment=False, augment=data_augment_params,
                            use_class_weight=True, class_weight='dummy',
                            debug=True)
    imgs, lbl, cnf = gen.next_val().__next__()

    for i in [0, 20]:
        plt.figure()

        plt.subplot(231)
        plt.title('L:{}'.format(lbl[i]))
        plt.imshow(np.squeeze(imgs[0][i]))
        plt.subplot(232)
        plt.title('L:{}'.format(lbl[i + 2]))
        plt.imshow(np.squeeze(imgs[0][i + 2]))
        plt.subplot(233)
        plt.title('L:{}'.format(lbl[i + 5]))
        plt.imshow(np.squeeze(imgs[0][i + 5]))

        plt.subplot(234)
        plt.title('C:{}'.format(cnf[i]))
        plt.imshow(np.squeeze(imgs[1][i]))
        plt.subplot(235)
        plt.title('C:{}'.format(cnf[i + 2]))
        plt.imshow(np.squeeze(imgs[1][i + 2]))
        plt.subplot(236)
        plt.title('C:{}'.format(cnf[i + 5]))
        plt.imshow(np.squeeze(imgs[1][i + 5]))

else:
    gen = DataGeneratorDir(configuration=0, data_size=128, model_size=128, res=0.5, sample='Normal', batch_sz=64,
                           objective='malignancy',
                           do_augment=False, augment=data_augment_params,
                           use_class_weight=False, class_weight='dummy',
                           debug=True)
    imgs, lbl = gen.next_val().__next__()

    for i in [0, 10, 20, 30, 40]:

        plt.figure()

        plt.subplot(231)
        plt.title('L:{}'.format(lbl[i]))
        plt.imshow(np.squeeze(imgs[i]), cmap='gray')
        plt.subplot(232)
        plt.title('L:{}'.format(lbl[i+2]))
        plt.imshow(np.squeeze(imgs[i+2]), cmap='gray')
        plt.subplot(233)
        plt.title('L:{}'.format(lbl[i+5]))
        plt.imshow(np.squeeze(imgs[i+5]), cmap='gray')

plt.show()
