import numpy as np
import random
import itertools
#from skimage.transform import resize
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import dataUtils

def MalignancyConfusionMatrix(pred, true):
    cm = confusion_matrix(  dataUtils.uncategorize(true),
                            dataUtils.uncategorize(pred) )
    #          Pred
    #   True  TN  FP
    #         FN  TP
    #plt.figure()

    TN = cm[0,0]
    TP = cm[1,1]
    FP = cm[0,1]
    FN = cm[1,0]

    print("Accuracy: {}".format( (TP+TN) / (TP+TN+FN+FP) ))
    print("Sensitivity: {}".format(TP/(TP+FN)))
    print("Precision: {}".format(TP / (TP + FP)))
    print("Specificity: {}".format(TN / (TN + FP)))

    plt.matshow(cm, cmap=plt.cm.Blues)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.title('Maligancy Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True')
    plt.xlabel('Predicted')

def MalignancyBySize(pred, true, size):
    assert pred.shape[0] == true.shape[0]

    true = dataUtils.uncategorize(true)
    pred = dataUtils.uncategorize(pred)
    plt.figure()
    plt.title("Malignancy by Nodule Size")
    plt.scatter(size, true, c=(pred!=true).astype('uint'))
    plt.xlabel('Size [mm]')
    plt.ylabel('True Malignancy')

def history_summarize(history, label=''):
    if hasattr(history, 'history'):
        history = history.history

    print(history.keys())
    # dict_keys(['loss', 'val_sensitivity', 'val_categorical_accuracy', 'val_loss', 'val_specificity', 'val_precision', 'lr', 'precision', 'categorical_accuracy', 'sensitivity', 'specificity'])

    plt.figure()

    # summarize history for sensitivity
    plt.subplot(311)
    plt.plot(history['sensitivity'])
    plt.plot(history['val_sensitivity'])
    plt.title(label+': sensitivity')
    plt.ylabel('sensitivity')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for precision
    plt.subplot(312)
    plt.plot(history['precision'])
    plt.plot(history['val_precision'])
    plt.title(label+': precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss and lr
    plt.subplot(313)
    plt.plot(history['lr'])
    plt.plot(history['loss'])
    plt.title(label+': loss and lr')
    plt.ylabel('loss n lr')
    plt.xlabel('epoch')
    plt.legend(['lr', 'loss'], loc='upper left')
    plt.show()