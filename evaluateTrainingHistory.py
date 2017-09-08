import pickle
import numpy as np

from analysis import history_summarize

subfolder = './history/'

file_list = [   #'history-003.p',
                'history-004.p']

for f in file_list:
    print(subfolder+f)
    history = pickle.load(open(subfolder+f, 'br'))
    history_summarize(history, f)