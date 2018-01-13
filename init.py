import os
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
np.random.seed(1337)  # for reproducibility
import random
random.seed(1337)
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import FileManager