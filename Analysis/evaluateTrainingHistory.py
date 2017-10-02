import pickle
import matplotlib.pyplot as plt

from Analysis.analysis import history_summarize

subfolder = './history/'

file_list = [   #'history-siam00ZZ.p',
                #'history-siam00YY.p',
                'history-siam00YYY.p'
                #'history-siam000.p',
                #'history-siam006.p',
                #'history-siam009.p'
            ]

for f in file_list:
    print(subfolder+f)
    history = pickle.load(open(subfolder+f, 'br'))
    history_summarize(history, f)


plt.show()