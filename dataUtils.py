import numpy as np
from sklearn.utils import class_weight

def uncategorize(hot_labels):
    labels = np.argmax(hot_labels, axis=1)
    return labels

def get_class_weight(labels):
    if labels.ndim > 1:
        labels = uncategorize(labels)
    cw = class_weight.compute_class_weight( 'balanced',
                                            np.unique(labels),
                                            labels )
    return cw