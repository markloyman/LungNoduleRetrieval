import matplotlib.pyplot as plt
from Network import FileManager
from Analysis import Retriever
from Analysis.metric_space_indexes import calc_hubness


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
    h, k_occ = calc_hubness(indices, K=[5, 10, 20], verbose=True, label=label)

plt.show()