import numpy as np
import matplotlib.pyplot as plt

from Network.DataGenSiam import DataGenerator


gen = DataGenerator(data_size=144, model_size=128, batch_sz=64, use_class_weight=True, do_augment=True, debug=True)


imgs, lbl, cnf = gen.next_val().__next__()

for i in [0, 10, 20]:

    plt.figure()

    plt.subplot(231)
    plt.title('L:{}'.format(lbl[i]))
    plt.imshow(np.squeeze(imgs[0][i]))
    plt.subplot(232)
    plt.title('L:{}'.format(lbl[i+2]))
    plt.imshow(np.squeeze(imgs[0][i+2]))
    plt.subplot(233)
    plt.title('L:{}'.format(lbl[i+5]))
    plt.imshow(np.squeeze(imgs[0][i+5]))

    plt.subplot(234)
    plt.title('C:{}'.format(cnf[i]))
    plt.imshow(np.squeeze(imgs[1][i]))
    plt.subplot(235)
    plt.title('C:{}'.format(cnf[i+2]))
    plt.imshow(np.squeeze(imgs[1][i+2]))
    plt.subplot(236)
    plt.title('C:{}'.format(cnf[i+5]))
    plt.imshow(np.squeeze(imgs[1][i+5]))

plt.show()