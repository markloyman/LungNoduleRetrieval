import os
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
np.random.seed(1337)  # for reproducibility
import random
random.seed(1337)
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'G:\LungNoduleRetrieval')


def factor(a, b):
    return np.maximum(a, b)/np.minimum(a, b)


def malig2(a, b):
    return np.abs(np.mean(a['rating'], axis=0)[-1] - np.mean(b['rating'], axis=0)[-1])

def malig(a):
    return np.mean(a['rating'], axis=0)[-1]

#filename = 'eval_dump_063e34.p'
filename = 'eval_dump_064Xe24.p'
failed_same, failed_diff = pickle.load(open(filename, 'br'))

distances = np.squeeze(np.array([a[2] for a in failed_diff]))
malignancy = np.array([malig2(a[0], a[1]) for a in failed_diff])
size = np.array([np.maximum(a[0]['size'], a[1]['size']) for a in failed_diff])

to_check = np.bitwise_and(distances<0.1, malignancy<1.5)

plt.figure('failed_diff')
plt.subplot(211)
plt.title('Malignancy')
plt.hist2d(distances, malignancy)
plt.subplot(212)
plt.title('Size')
plt.hist2d(distances, size, bins=[10, 20])

plt.show()

'''
#nodule_ids = []
nodule_ids = {}
for pair, select in zip(failed_diff, to_check):
    if select:
        try:
            nodule_ids[pair[0]['info'][0]].append(pair[0]['info'][3])
        except:
            nodule_ids[pair[0]['info'][0]] = []
            nodule_ids[pair[0]['info'][0]].append(pair[0]['info'][3])
    #nodule_ids.append()
    #nodule_ids.append(pair[1]['info'][0] + '_' + pair[1]['info'][3])

plt.figure()
plt.title('Freq of patients')
plt.hist([len(value) for key, value in nodule_ids.items()])

for key, value in nodule_ids.items():
    if len(value) >= 10:
        print(key)
        print(value)

# LIDC-IDRI-0129
# ['6', '6', '6', '6', 'Nodule 016', '6', 'Nodule 016', 'Nodule 016', '6', 'Nodule 016', 'Nodule 016', '1', 'Nodule 016', 'Nodule 016', 'Nodule 016']
# LIDC-IDRI-0027
# ['IL057_130667', 'IL057_130667', 'IL057_130664', 'IL057_130667', 'MI014_16647', 'MI014_16647', 'IL057_130667', 'IL057_130664', 'MI014_16647', 'IL057_130667', 'MI014_16647']
# LIDC-IDRI-1008
# ['0', '3', '3', '0', '3', '3', '0', '3', '0', '3', '0']
# ^^ examples of benign missclassified as malignant

for pair in failed_diff:
    if (pair[0]['info'][0] == 'LIDC-IDRI-0129') or (pair[1]['info'][0] == 'LIDC-IDRI-0129'):
        if (pair[0]['info'][-1] == '6') or (pair[1]['info'][-1] == '6'):
            print('Pair: {}, {}: {}'.format(malig(pair[0]), malig(pair[1]), pair[2]))
            print(pair[1]['info'])
            plt.figure()
            plt.subplot(211)
            plt.imshow(pair[0]['patch'])
            plt.subplot(212)
            plt.imshow(pair[1]['patch'])

images, pred, meta, labels, masks = pickle.load(open('output/embed/embed_siam063-34_Valid.p', 'br'))

plt.figure()
for p,m in zip(pred, meta):
    if m[0] == 'LIDC-IDRI-0686':
        if m[-1] == '9352':
            plt.plot(p)

'''

'''
distances = np.squeeze(np.array([a[2] for a in failed_same]))
malignancy = np.array([malig(a[0], a[1]) for a in failed_same])
size = np.array([factor(a[0]['size'], a[1]['size']) for a in failed_same])

plt.figure('failed_same')
plt.subplot(211)
plt.title('Malignancy')
plt.hist2d(distances, malignancy)
plt.subplot(212)
plt.title('Size')
plt.hist2d(distances, size)
'''