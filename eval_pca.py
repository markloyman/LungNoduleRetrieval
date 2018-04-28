from init import *
from Analysis.RatingCorrelator import calc_distance_matrix
from Analysis.retrieval import Retriever
from Network import FileManager
from experiments import load_experiments
from scipy.signal import savgol_filter

dset = 'Valid'
metrics = ['l2']
n_groups = 5

runs, run_net_types, run_metrics, run_epochs, run_names = load_experiments('Repeat160')


for run, net_type, r, epochs in zip(runs, run_net_types, range(len(runs)), run_epochs):
    Embed = FileManager.Embed(net_type)
    embed_source = [Embed(run + 'c{}'.format(c), dset) for c in range(n_groups)]
    Ret = Retriever(title='{}'.format(run), dset=dset)
    embd, epoch_mask = Ret.load_embedding(embed_source, multi_epcch=True)

    plt.figure()
    for i, e in enumerate([1, 5, 20, 3]):
        plt_ = plt.subplot(2, 2, i+1)
        Ret.pca(epoch=e, plt_=plt_)

plt.show()