import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pickle
import sys
from timeit import default_timer as timer
import os
sys.path.insert(0, 'G:\LungNoduleRetrieval')
# for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1337)
random.seed(1337)
