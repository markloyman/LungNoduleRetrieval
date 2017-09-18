import pickle

import matplotlib.pyplot as plt

from LIDC.lidcUtils import CheckPatch

plt.interactive(True)

dataset = pickle.load(open('LIDC/NodulePatchesClique.p', 'br'))

CheckPatch(dataset[111])

