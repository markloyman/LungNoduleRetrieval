import os
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
np.random.seed(1337)  # for reproducibility
import random
random.seed(1337)
import matplotlib.pyplot as plt

try:
    from Network.DataGenSiam import DataGenerator
    from Network.DataGenDirect import DataGeneratorDir
except:
    from DataGenSiam import DataGenerator
	
data_augment_params = {'max_angle': 0, 'flip_ratio': 0.0, 'crop_stdev': 0.1, 'epoch': 0}

if True:
    gen = DataGenerator(    data_size=144, model_size=128, res='Legacy', sample='Normal', batch_sz=64,
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
    gen = DataGeneratorDir(data_size=144, model_size=128, res='0.5I', sample='Normal', batch_sz=64,
                        objective='rating',
                        do_augment=False, augment=data_augment_params,
                        use_class_weight=False, class_weight='dummy',
                        debug=True)
    imgs, lbl = gen.next_val().__next__()

    for i in [0, 10, 20, 30, 40]:

        plt.figure()

        plt.subplot(231)
        plt.title('L:{}'.format(lbl[i]))
        plt.imshow(np.squeeze(imgs[i]))
        plt.subplot(232)
        plt.title('L:{}'.format(lbl[i+2]))
        plt.imshow(np.squeeze(imgs[i+2]))
        plt.subplot(233)
        plt.title('L:{}'.format(lbl[i+5]))
        plt.imshow(np.squeeze(imgs[i+5]))




plt.show()