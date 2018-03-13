import matplotlib.pyplot as plt
import numpy as np
import random
import LIDC
random.seed(1337)   # for reproducibility
np.random.seed(1337)


LIDC.check_nodule_intersections(patch_size=144, res=0.5)
plt.show()
