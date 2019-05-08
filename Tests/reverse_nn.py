import numpy as np
import matplotlib.pyplot as plt
from Network import FileManager
from Analysis import Retriever
from Analysis.metric_space_indexes import k_occurrences


net_type = 'dirD'
config = 0
dset = 'Valid'
K = 2

res = {}
for run, label in zip(['821', '822'], ['Pearson-loss', 'KL-loss']):
    print(run + ': ' + label + '\n' + '*' * 20)
    embed_source = FileManager.Embed(net_type)(run + 'c{}'.format(config), dset)

    Ret = Retriever(title='{}'.format(''), dset=dset)
    Ret.load_embedding(embed_source, multi_epcch=True)
    Ret.fit(metric='euclidean', epoch=60)
    indices, distances = Ret.ret_nbrs()

    # get Hubs
    k_occ = k_occurrences(indices, K)
    hubs_indices = np.argsort(k_occ)[-3:]
    res[run] = hubs_indices, indices
    print([(a, b) for a, b in zip(hubs_indices, k_occ[hubs_indices])])

for run, label in zip(['821', '822'], ['Pearson-loss', 'KL-loss']):
    print(run + ': ' + label + '\n' + '*' * 20)

    hubs_indices, indices = res[run]
    rev_nn = [(hub_id, np.where(np.any(hub_id == indices[:, :K], axis=1))) for hub_id in hubs_indices]
    [print('\t{} => {}'.format(run, rn)) for rn in rev_nn]

    for run2 in ['821', '822']:
        if run2 == run:
            continue
        _, indices2 = res[run2]
        rev_nn = [(hub_id, np.where(np.any(hub_id == indices2[:, :K], axis=1))) for hub_id in hubs_indices]
        [print('\t{} => {}'.format(run2, rn)) for rn in rev_nn]





plt.show()