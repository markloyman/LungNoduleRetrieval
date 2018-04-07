import numpy as np
import pickle
import matplotlib.pyplot as plt

#from Analysis.analysis import history_summarize


def combined_metric(run, targets, n_epochs, subfolder='./output/history/', n_groups=5):
    file_list = [subfolder + 'history-{}c{}.p'.format(run, config) for config in range(n_groups)]
    #metrics = dict.fromkeys(targets, np.zeros(n_epochs))
    metrics = {key: np.zeros(n_epochs) for key in targets}
    n_entries = 0
    for filepath in file_list:
        try:
            history = pickle.load(open(filepath, 'br'))
            for target in targets:
                key = 'val_' + target
                metrics[target] += np.array(history[key][:n_epochs])
            n_entries += 1
        except:
            print("couldn't find {}".format(filepath))
    assert(n_entries > 0)
    metrics.update({k: (v / n_entries) for k, v in metrics.items()})
    return metrics

if __name__ == "__main__":

    targets = ['accuracy', 'f1']
    runs = ['dir200', 'dir201']
    last_epoch = [51, 51]

    plt.figure()
    plt_ = [None] * len(targets)
    for i, label in enumerate(targets):
        plt_[i] = plt.subplot(len(plt_), 1, i + 1)
        plt_[i].axes.yaxis.label.set_text(label)
        plt_[i].grid(which='both', axis='y')

    for run, epoch in zip(runs, last_epoch):
        metrics = combined_metric(run=run, targets=targets, n_epochs=epoch)
        for i, label in enumerate(targets):
            plt_[i].plot(metrics[label], '*-')

    plt_[0].legend(runs)
    plt_[-1].axes.xaxis.label.set_text('epochs')

    plt.show()
