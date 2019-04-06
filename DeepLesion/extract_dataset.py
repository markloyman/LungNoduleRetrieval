import numpy as np
from skimage.io import imread
from DeepLesion.load_metadata import load_meta, DeepLesionMetaEntry
import os.path


def valid_entry(x: DeepLesionMetaEntry) -> bool:
    valid = (x.LesionType == 5) and \
            (not x.Path['Noisy']) and \
            (x.get_bb_size_in_mm() <= 50.0)
            #(x.Window[0] >= -1000.) #and \
    return valid


def crop_roi(image: np.ndarray, center, roi_size):
    x0, y0 = center[0] - roi_size, center[1] - roi_size
    assert x0 >= 0
    assert y0 >= 0
    assert x0 + roi_size < image.shape[0]
    assert y0 + roi_size < image.shape[1]

    roi = image[x0:(x0 + roi_size),
                y0:(y0 + roi_size)]

    return roi


# ['patch', 'mask', 'weights',  'rating', 'z', 'label', 'ann_size', 'size', 'nod_ids', 'info']
def extract(data_path: str, meta_path: str, patch_size: int):

    full_metadata = load_meta(
        filename=os.path.join(meta_path, 'DL_info.csv'),
        index_field='File_name')

    filtered_metadata = filter(
            lambda x: valid_entry(x),
            full_metadata.values())

    dataset = list()
    for metadata in filtered_metadata:
        print('Processing {}...'.format(metadata.Path['FileName']))
        filepath = os.path.join(data_path, metadata.Path['FileName'])
        im = imread(filepath, as_grey=True)
        im -= 32768  # to HU
        patch = crop_roi(im, metadata.get_center(), patch_size)

        mask = np.zeros_like(im)

        entry = {
            'patch': patch.astype(np.int16),
            'info': (metadata.Path['Patient'], metadata.Path['Study'], metadata.Path['Series'], metadata.Slice['Key']),
            'nod_ids': metadata.Slice['Key'],
            'rating': np.zeros([1, 9]),
            'ann_size': np.array(metadata.get_bb_size_in_mm()),
            'weights': np.array(1.),
            'mask': mask.astype(np.bool),
            'z': metadata.Slice['Key'],
            'size': metadata.DiameterPx[0]
        }
        dataset.append(entry)

    print("Prepared {} entries".format(len(dataset)))

    return dataset
