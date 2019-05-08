import numpy as np
import matplotlib.pyplot as plt
from Network import FileManager
from Analysis import Retriever
from Analysis.metric_space_indexes import calc_hubness, k_occurrences


net_type = 'dirD'
config = 0
dset = 'Valid'

for run, label in zip(['821', '822'], ['Pearson-loss', 'KL-loss']):
    embed_source = FileManager.Embed(net_type)(run + 'c{}'.format(config), dset)

    Ret = Retriever(title='{}'.format(''), dset=dset)
    Ret.load_embedding(embed_source, multi_epcch=True)

    Ret.fit(metric='euclidean', epoch=60)
    indices, distances = Ret.ret_nbrs()

    # hubness
    h, _ = calc_hubness(indices, K=[2], verbose=True, label=label)

    print(run + ': ' + label + '\n' + '*'*20)

    k_occ = k_occurrences(indices, 2)
    print("\t{} orphan nodules".format(len(np.argwhere(k_occ == 0))))

    #hubs_indices = np.argsort(k_occ)[-5:]
    #print([(a, b) for a, b in zip(hubs_indices, k_occ[hubs_indices])])
    #
    #for hub_id in hubs_indices:
    #    qs = np.where(np.any(indices[:, :2] == hub_id, axis=1))
    #    print('\t{} => {}'.format(hub_id, qs))

plt.show()