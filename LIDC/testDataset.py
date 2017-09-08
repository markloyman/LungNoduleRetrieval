import pickle

import matplotlib.pyplot as plt

from LIDC.lidcUtils import CheckPatch

plt.interactive(False)

dataset = pickle.load(open('LIDC/NodulePatches.p','br'))

CheckPatch(dataset[111])

