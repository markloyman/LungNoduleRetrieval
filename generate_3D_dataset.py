from init import *
from Network.FileManager import Embed, Dataset3d
from Network.dataUtils import crop_center

n_groups = 5

# base dataset
dataset_type = 'Primary'
size = 160
input_size = 128
res = 0.5
sample = 'Normal'

# embedding
net_type = 'dirR'
run = '251'
epoch = 60


def process_data(data):
    embed, epochs, meta, images, classes, labels, masks, z = data
    embed = np.squeeze(embed[np.array(epochs) == epoch])
    print('Loaded embedding: {} for config #{}'.format(embed.shape, c))

    nodule_ids = [reduce(lambda x, y: x + y, [m[0]] + m[-1]) for m in meta]
    unique_ids, id_map = np.unique(nodule_ids, return_inverse=True)

    dataset = []
    for i, idx in enumerate(unique_ids):
        curr_roi = np.array([int(idx) for idx in np.argwhere(id_map == i)])
        new_order = np.argsort(np.array([z[idx] for idx in curr_roi]).flatten())
        curr_roi = curr_roi[new_order]

        roi_volume = {}
        roi_volume['embed'] = np.moveaxis(np.array([embed[idx] for idx in curr_roi]), 0, 2)
        roi_volume['patch'] = np.array([images[idx] for idx in curr_roi]).swapaxes(0, -1).squeeze(axis=0)
        roi_volume['mask'] = np.array(
            [crop_center(image=None, mask=masks[idx], size=input_size)[1] for idx in curr_roi]).swapaxes(0, -1).squeeze(
            axis=0)

        roi_volume['rating'] = labels[curr_roi[0]]
        roi_volume['label'] = classes[curr_roi[0]]
        roi_volume['info'] = meta[curr_roi[0]]

        roi_volume['z'] = np.array([z[idx] for idx in curr_roi])
        roi_volume['size'] = np.count_nonzero(roi_volume['mask'])
        roi_volume['weights'] = None

        dataset.append(roi_volume)

    return dataset


for c in range(n_groups):

    # Valid -->> Train
    # ====================

    for d in ['Train', 'Valid', 'Test']:
        data = Embed('SP_' + net_type).load(run=run+'c{}'.format(c), dset=d)
        dataset = process_data(data)

        out_filename = Dataset3d(c).name(dset=d, net=net_type, run=run, epoch=epoch)
        pickle.dump(dataset, open(out_filename, 'bw'))
        print('Dumpted {} entries to: {}'.format(len(dataset), out_filename))

    # Test -->> Valid
    # ====================
    '''
    data = Embed('SP_' + net_type).load(run=run + 'c{}'.format(c), dset='Test')
    dataset = process_data(data)

    out_filename = Dataset3d(c).name(dset='Valid', net=net_type, run=run, epoch=epoch)
    pickle.dump(dataset, open(out_filename, 'bw'))
    print('Dumpted {} entries to: {}'.format(len(dataset), out_filename))
    '''
