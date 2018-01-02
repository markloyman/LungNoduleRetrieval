import numpy as np
from numpy.random import rand
from scipy.ndimage.interpolation import rotate
from sklearn.utils import class_weight
import matplotlib.pyplot as plt


def rating_normalize(rating, method):
    rating_mean = np.array(
        [3.75904203, 1.01148583, 5.50651678, 3.79985337, 3.96358749, 1.67269469, 1.56867058, 4.4591072, 2.55197133])
    rating_std = np.array(
        [1.09083287, 0.11373469, 1.02463593, 0.80119638, 1.04281277, 0.89359593, 0.89925905, 1.04813052, 1.12151403])
    rating_min = np.array(
        [1., 1., 1., 1., 1., 1., 1., 1., 1.])
    rating_max = np.array(
        [5., 4., 6., 5., 5., 5., 5., 5., 5.])

    if method == 'Norm' or method == 'Normal':
        #rating = rating.astype('float')
        rating_norm = (rating - rating_mean) / rating_std
    elif method == 'Scale':
        rating_norm = 2.0*(rating - rating_min)/(rating_max - rating_min)-1.0
    else:
        rating_norm = rating

    return rating_norm


def uncategorize(hot_labels):
    labels = np.argmax(hot_labels, axis=1)
    return labels


def get_class_weight(labels, method='balanced'):
# returns a dict of class weight
# CW[label] = weight
    CW = {}
    if labels.ndim > 1:
        labels = uncategorize(labels)
    if method is 'balanced':
        unique_labels = np.unique(labels)
        cw = class_weight.compute_class_weight( 'balanced',
                                                unique_labels,
                                                labels )
        for l, w in zip(unique_labels, cw):
            CW[l] = w
    elif method is 'count':
        # count
        d  = np.count_nonzero([l == 'D' for l in labels])
        sb = np.count_nonzero([l == 'SB' for l in labels])
        sm = np.count_nonzero([l == 'SM' for l in labels])
        n = d + sm + sb
        # calc class weights
        CW['D']  = n / (2.0 * d)
        CW['SB'] = n / (4.0 * sb)
        CW['SM'] = n / (4.0 * sm)
    elif method is 'dummy':
        CW['D'] = 1
        CW['SB'] = 1
        CW['SM'] = 1
    else:
        print("Illegal class weight method: {}".format(method))
        assert(False)

    return CW


def get_sample_weight(labels, wD=None, wSB=None, wSM=None):
    if labels.ndim > 1:
        labels = uncategorize(labels)
    if (wD is None) or (wSB is None) or (wSM is None):
        # count occurrences
        d  = np.count_nonzero([l == 'D'  for l in labels])
        sb = np.count_nonzero([l == 'SB' for l in labels])
        sm = np.count_nonzero([l == 'SM' for l in labels])
        n =  d + sm + sb
        # calc class weights
        wD  = n / (2.0 * d)
        wSB = n / (4.0 * sb)
        wSM = n / (4.0 * sm)
    label_weights = { 'D':  wD,
                      'SB': wSB,
                      'SM': wSM
                    }
    # calc sample weights
    sample_weights = class_weight.compute_sample_weight(label_weights, labels)
    return sample_weights


def crop_center(image, mask, size=None):
    assert(size is not None)

    mask_y = np.where(np.sum(mask, axis=1))[0]
    mask_x = np.where(np.sum(mask, axis=0))[0]
    center = np.median(mask_y).astype('int'), np.median(mask_x).astype('int')

    cx, cy = max(center[0], size//2), max(center[1], size//2)
    cx, cy = min(cx, image.shape[0] - size//2), min(cy, image.shape[1] - size//2)

    x0 = np.round(cx - size//2).astype('int')
    y0 = np.round(cy - size//2).astype('int')
    size = np.round(size).astype('int')
    
    return image[y0:(y0+size), x0:(x0+size)], mask[y0:(y0+size), x0:(x0+size)]


def crop(image, mask, fix_size=None, min_size=0, ratio=1.0, stdev=0.15):
    # Size:
    #   first sampled as Uniform(mask-size, full-patch-size)
    #   than clipped by [min_size] to more weight to the nodule size
    # ratio:
    #   Each dimension size is sampled so as to conform to [1-ratio, 1+ratio] relation
    # the shift of the crop window is sampled with normal distribution to have a bias towards a centered crop
    mask_y = np.where(np.sum(mask, axis=1))[0]
    mask_x = np.where(np.sum(mask, axis=0))[0]
    if fix_size is None:
        # Determine Size boundries on Y axis
        max_size_y = image.shape[0]
        min_size_y = np.max(mask_y) - np.min(mask_y)
        #print('min_size_y, max_size_y={}, {}'.format(min_size_y, max_size_y))
        # Determine Size boundries on X axis
        max_size_x = image.shape[1]
        min_size_x = np.max(mask_x) - np.min(mask_x)
        #print('min_size_x, max_size_x={}, {}'.format(min_size_x, max_size_x))
        # correct for ratio
        min_size_y = np.maximum(min_size_y, (1-ratio)*min_size_x)
        max_size_y = np.minimum(max_size_y, (1+ratio)*max_size_x)
        #print('min_size_y, max_size_y={}, {}'.format(min_size_y, max_size_y))

        # Determine New Size on Y axis
        new_size_y = np.random.rand() * (max_size_y - min_size_y) + min_size_y
        new_size_y = np.maximum(new_size_y, min_size).astype('uint16')
        #print('new_size_y={}'.format(new_size_y))
        # correct for ratio (based on chosen size of Y
        min_size_x = np.maximum(min_size_x, (1 - ratio) * new_size_y)
        max_size_x = np.minimum(max_size_x, (1 + ratio) * new_size_y)
        #print('min_size_x, max_size_x={}, {}'.format(min_size_x, max_size_x))
        # Determine New Size on X axis
        new_size_x = np.random.rand() * (max_size_x - min_size_x) + min_size_x
        new_size_x = np.maximum(new_size_x, min_size).astype('uint16')
        #print('new_size_x={}'.format(new_size_x))
    else:
        new_size_x = fix_size
        new_size_y = fix_size

    # Determine crop translation on Y axis
    min_crop_start_y = np.maximum(0, np.max(mask_y) - new_size_y)
    max_crop_start_y = np.minimum(np.min(mask_y), image.shape[0] - new_size_y)
    rand_factor = np.maximum(0.0, np.minimum(1.0, stdev*np.random.normal()+0.5))
    crop_start_y     = min_crop_start_y + rand_factor * (max_crop_start_y - min_crop_start_y)
    crop_start_y = crop_start_y.astype('uint16')
    #print('min_crop_start_y, max_crop_start_y={}, {}'.format(min_crop_start_y, max_crop_start_y))
    #print('factor={}'.format(rand_factor))
    #print('crop_start_y={}'.format(crop_start_y))

    # Determine crop translation on X axis
    min_crop_start_x = np.maximum(0, np.max(mask_x) - new_size_x)
    max_crop_start_x = np.minimum(np.min(mask_x), image.shape[1] - new_size_x)
    rand_factor = np.maximum(0.0, np.minimum(1.0, stdev * np.random.normal() + 0.5))
    crop_start_x     = min_crop_start_x + rand_factor * (max_crop_start_x - min_crop_start_x)
    crop_start_x = crop_start_x.astype('uint16')
    #print('min_crop_start_x, max_crop_start_x={}, {}'.format(min_crop_start_x, max_crop_start_x))
    #print('factor={}'.format(rand_factor))
    #print('crop_start_x={}'.format(crop_start_x))

    assert (crop_start_x >= 0)
    assert (crop_start_y >= 0)
    assert ( (crop_start_y + new_size_y) <= image.shape[0])
    assert ( (crop_start_x + new_size_x) <= image.shape[1])
    new_image = image[crop_start_y:crop_start_y + new_size_y, crop_start_x:crop_start_x + new_size_x]
    new_mask  =  mask[crop_start_y:crop_start_y + new_size_y, crop_start_x:crop_start_x + new_size_x]

    return new_image, new_mask


def augment(image, mask, size=0, max_angle=0, flip_ratio=0.0, crop_ratio=1.0, crop_stdev=0.15):
    # randomize
    angle = 0.3 * max_angle * np.random.normal()
    angle = np.maximum(np.minimum(angle, max_angle), -max_angle)
    do_flip = rand() < flip_ratio

    # apply
    image_r = rotate(image, angle, reshape=False, mode='nearest')
    mask_r  = rotate(mask,  angle, reshape=False, mode='nearest')

    image_r, mask_r = crop(image_r, mask_r, fix_size=size, ratio=crop_ratio, stdev=crop_stdev)

    if do_flip:
        image_r, mask_r = np.fliplr(image_r), np.fliplr(mask_r)

    return image_r, mask_r


def test_augment(dataset):
    im_array = []
    for entry in dataset:
        image, mask = entry[0], entry[1]
        image, mask = augment(image, mask, min_size=128, max_angle=30, flip_ratio=0.3)
        im_array.append(image)

    values = np.concatenate([np.array(im).flatten() for im in im_array])
    print(values.shape)
    plt.hist(values, 100)
    plt.show()

if __name__ == "__main__":

    try:
        from Network.data import load_nodule_dataset
    except:
        from data import load_nodule_dataset
		
    data = load_nodule_dataset()
    test_augment(data[2])

