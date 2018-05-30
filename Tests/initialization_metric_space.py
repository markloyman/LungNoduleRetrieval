from init import *
from Network.model import miniXception_loader
from Network.Direct.directArch import DirectArch
from Network.data_loader import load_nodule_dataset
import Analysis.metric_space_indexes as index
from sklearn.neighbors import NearestNeighbors

# =====================
#       Setup
# =====================

data_size = 128
res = 0.5  # 'Legacy' #0.7 #0.5 #'0.5I'
sample = 'Normal'  # 'UniformNC' #'Normal' #'Uniform'
# model
model_size = 128
input_shape = (model_size, model_size, 1)
out_size = 128

pooling_options = ['max', 'avg', 'msrmac']
metric = 'l2'
repeatitions = 200  # 10 rep ~ 1:20 hour

merge_range = lambda rng1, rng2: (np.minimum(rng1[0], rng2[0]), np.maximum(rng1[1], rng2[1]))

# =====================
#       Load Data
# =====================

dataset = load_nodule_dataset(size=data_size, res=res, sample=sample, configuration=0)
dataset = reduce(lambda x, y: x + y, dataset)

N = len(dataset)

images, masks, classes, meta, size, rating = zip(*dataset[:N])
images = np.array(images)

# =====================
#       Evaluate
# =====================

H_lim, S_lim, Tau_lim, Conc_lim, Crst_lim = (1e6, 0), (1e6, 0), (1e6, 0), (1e6, 0), (1e6, 0)

for p, pooling in enumerate(pooling_options):
    print('Evaluating {} pooling'.format(pooling))
    h, s, tau, conc, crst = [], [], [], [], []
    plot_data_filename = './Plots//Data/init_{}.p'.format(pooling)
    try:
        h, s, tau, conc, crst = pickle.load(open(plot_data_filename, 'br'))
        print('loaded cached data for ' + pooling)
    except:
        start = timer()
        for i in range(repeatitions):
            print('\tRep # {}'.format(i))
            model = DirectArch(miniXception_loader, input_shape, objective="malignancy", pooling=pooling,
                               output_size=out_size, normalize=True)
            core = model.extract_core()
            embed = core.predict(np.expand_dims(images, axis=-1), batch_size=32)

            nbrs = NearestNeighbors(n_neighbors=N, algorithm='auto', metric=metric).fit(embed)
            distances, indices = nbrs.kneighbors(embed)

            distances, indices = distances[:, 1:], indices[:, 1:]

            h += [index.calc_hubness(indices)[0]]
            s += [index.calc_symmetry(indices)[0]]
            tau += [index.kumar(distances, res=0.01)[0]]
            conc += [index.concentration(distances)]
            crst += [index.relative_contrast_imp(distances)]

        pickle.dump((h, s, tau, conc, crst), open(plot_data_filename, 'bw'))
        print('evaluated (and cached) {} in {:.1f} minutes '.format(pooling, (timer() - start) / 60))

    plt.subplot(len(pooling_options), 5, 1 + p*5)
    plt.hist(np.array(h).flatten(), bins=20)
    plt.ylabel('hubness')
    plt.title(pooling)
    h_lim = plt.gca().get_xlim()
    H_lim = merge_range(H_lim, h_lim)

    plt.subplot(len(pooling_options), 5, 2 + p*5)
    plt.hist(np.array(s).flatten(), bins=20)
    plt.ylabel('symmetry')
    plt.title(pooling)
    s_lim = plt.gca().get_xlim()
    S_lim = merge_range(S_lim, s_lim)

    plt.subplot(len(pooling_options), 5, 3 + p*5)
    plt.hist(np.array(tau).flatten(), bins=20)
    plt.ylabel('tau (kumari)')
    plt.title(pooling)
    tau_lim = plt.gca().get_xlim()
    Tau_lim = merge_range(Tau_lim, tau_lim)

    plt.subplot(len(pooling_options), 5, 4 + p*5)
    plt.hist(np.array(conc).flatten(), bins=20)
    plt.ylabel('concentration')
    plt.title(pooling)
    conc_lim = plt.gca().get_xlim()
    Conc_lim = merge_range(Conc_lim, conc_lim)

    plt.subplot(len(pooling_options), 5, 5 + p*5)
    plt.hist(np.array(crst).flatten(), bins=20)
    plt.ylabel('conrast')
    plt.title(pooling)
    crst_lim = plt.gca().get_xlim()
    Crst_lim = merge_range(Crst_lim, crst_lim)

for p, pooling in enumerate(pooling_options):
    plt.subplot(len(pooling_options), 5, 1 + p * 5)
    plt.xlim(H_lim)

    plt.subplot(len(pooling_options), 5, 2 + p * 5)
    plt.xlim(S_lim)

    plt.subplot(len(pooling_options), 5, 3 + p * 5)
    plt.xlim(Tau_lim)

    plt.subplot(len(pooling_options), 5, 4 + p * 5)
    plt.xlim(Conc_lim)

    plt.subplot(len(pooling_options), 5, 5 + p * 5)
    plt.xlim(Crst_lim)

print('DONE!')
plt.show()
