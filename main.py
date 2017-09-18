import numpy as np
from data import load_nodule_dataset, prepare_data, prepare_data_siamese_overlap
from model import miniXception_loader

from Network.directArch import directArch
from Network.siameseArch import siamArch

## --------------------------------------- ##
## ------- General Setup ----------------- ##
## --------------------------------------- ##

#dataset = generate_nodule_dataset(0.2, 0.25)
dataset = load_nodule_dataset()
size = 128
input_shape = (size, size,1)

# DIR / SIAM
choose_model = "SIAM"

## --------------------------------------- ##
## ------- Run Direct Architecture ------- ##
## --------------------------------------- ##

if choose_model is "DIR":
    #run = '005': lr=0.01, reduce factor=0.5, epsilon=0.05, patience=10, decay=lr/100
    #run = '006': lr=0.005, reduce factor=0.5, epsilon=0.05, patience=10, decay=lr/100
    #run = '007': lr=0.005, added dropout
    #run = '008' # categorical_hinge - sucks
    run = '009'

    # prepare training data
    images_train, labels_train = prepare_data(dataset[2], classes=2, size=size)
    print("Trainings data ready: images({}), labels({})".format(images_train.shape, labels_train.shape))
    print("Range = [{},{}]".format(np.min(images_train), np.max(images_train)))

    # prepare validation data
    images_valid, labels_valid = prepare_data(dataset[1], classes=2, size=size)
    print("Validation data ready: images({}), labels({})".format(images_valid.shape, labels_valid.shape))
    print("Range = [{},{}]".format(np.min(images_valid), np.max(images_valid)))


    model = directArch(miniXception_loader, input_shape, 2)
    #model.summary()

    model.compile(learning_rate=0.005)
    model.load_data(    images_train,   labels_train,
                        images_valid,   labels_valid )
    model.train(run)



## --------------------------------------- ##
## ------- Run Siamese Architecture ------ ##
## --------------------------------------- ##

if choose_model is "SIAM":

    # run = 'siam000'   Adam(lr=0.005)
    # run = 'siam001' # RMSpop
    #run = 'siam002'  # Nadam
    #run = 'siam003'  # Nadam & reversed loss function
    #run = 'siam004'  # Nadam & reversed loss function & reversed labels
    #run = 'siam005'  # Adam & reversed loss function & reversed labels
    #run = 'siam006'   # reload data after 10 epochs
    #run = 'siam0065'  # reload data after 25 epochs
    #run = 'siam007'  # reload data after 25 epochs, increasebatch size to 64
    #run = 'siam008'  # reload after 25 epochs, batch size back to 32, margin = 5
    #run = 'siam009'  # overlapped data (reload after 25 epochs, batch size back to 32, margin = 5)
    run = 'siam010'  # corrented overlapped data (reload after 25 epochs, batch size back to 32, margin = 5)

    # data epoch - overlap (increases epoch size)
    # introduce DataGen
    # on first epochs, more similar examples are required

    # model

    model = siamArch(miniXception_loader, input_shape)
    model.model.summary()
    model.compile(learning_rate=0.005)

    epch_len = 25
    for epch0 in [0,1,2,3]:
        epch0 = epch_len * epch0
        print("Next Epoch: {}".format(epch0))

        # prepare training data
        images_train, labels_train = prepare_data_siamese_overlap(dataset[2], size=size)
        assert images_train[0].shape == images_train[1].shape
        print("Trainings data ready: image pairs({}), labels({})".format(images_train[0].shape, labels_train.shape))
        print("Range = [{},{}]".format(np.min(images_train), np.max(images_train)))
        print("Labels = [{},{}]".format(np.min(labels_train), np.max(labels_train)))

        # prepare validation data
        images_valid, labels_valid = prepare_data_siamese_overlap(dataset[1], size=size)
        assert images_valid[0].shape == images_valid[1].shape
        print("Validation data ready: image pairs({}), labels({})".format(images_valid[0].shape, labels_valid.shape))
        print("Range = [{},{}]".format(np.min(images_valid), np.max(images_valid)))
        print("Labels = [{},{}]".format(np.min(labels_valid), np.max(labels_valid)))


        model.load_data(images_train, labels_train,
                        images_valid, labels_valid)

        model.train(label=run, epoch=epch0, n_epoch=epch_len)