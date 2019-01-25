from glob import glob
import pickle
import numpy as np

class Weights(object):

    def __init__(self, pre, output_dir='./output'):
        print(output_dir)
        self.weightsTemplate = output_dir + '/Weights/w_{}{{}}_{{}}-{{}}-{{}}.h5'.format(pre)

    def read(self, run=None, epoch=None):
        match = self.weightsTemplate.format(run,epoch, '*', '*')
        return open(glob(match)[0], 'br')

    def write(self, run=None, epoch=None, loss=None, val_loss=None):
        match = self.weightsTemplate.format(run, epoch, loss, val_loss)
        return open(match, 'bw')

    def name(self, run=None, epoch=None, loss=None, val_loss=None, exact=True):
        if (run is None) or (epoch is None):
            return None

        name = None
        if (loss is None) or (val_loss is None):
            while (name is None) and (epoch >= 0):
                if isinstance(epoch, str):
                    match = self.weightsTemplate.format(run, epoch, '*', '*')
                    files = glob(match)
                    if len(files):
                        name = files[0]
                    else:
                        assert False
                else:
                    match = self.weightsTemplate.format(run, '{:02d}'.format(epoch), '*', '*')
                    files = glob(match)
                    if len(files):
                        name = files[0]
                        if len(files) > 1:
                            print('[Warning] Multiple matches for template "{}"'.format(match))
                    else:
                        if exact:
                            assert False
                        else:
                            epoch -= 1
        else:
            name = self.weightsTemplate.format(run, epoch, loss, val_loss)

        assert name is not None
        return name

    __call__ = name


class Embed(object):

    def __init__(self, pre, output_dir='./output'):
        self.weightsTemplate = output_dir + '/embed/embed_{}{{}}_{{}}.p'.format(pre)

    def read(self, run=None, dset=None):
        match = self.weightsTemplate.format(run, dset)
        filelist = glob(match)
        if len(filelist) == 0:
            print("Failed to find: {}".format(match))
            return None
        return open(filelist[0], 'br')

    def load(self, run=None, dset=None):
        data = pickle.load(self.read(run=run, dset=dset))
        if len(data) == 7:
            embed, epochs, meta, images, classes, labels, masks = data
            return embed, epochs, meta, images, classes, labels, masks
        elif len(data) == 8:
            embed, epochs, meta, images, classes, labels, masks, z = data
            return embed, epochs, meta, images, classes, labels, masks, z
        elif len(data) == 10:
            embed, epochs, meta, images, classes, labels, masks, conf, rating_weights, z = data
            return embed, epochs, meta, images, classes, labels, masks, conf, rating_weights, z
        else:
            assert False

    def write(self, run=None, epoch=None, dset=None):
        match = self.weightsTemplate.format(run, epoch, dset)
        return open(match, 'bw')

    def name(self, run=None, epoch=None,  dset=None):
        return self.weightsTemplate.format(run, epoch, dset)

    __call__ = name


class Pred(object):
    def __init__(self, type, pre, output_dir='./output'):
        self.manager = Embed('')
        if type == 'rating':
            self.manager.weightsTemplate = output_dir + '/embed/predR_{}{{}}_{{}}.p'.format(pre)
        elif type == 'malig':
            self.manager.weightsTemplate = output_dir + '/embed/pred_{}{{}}_{{}}.p'.format(pre)
        elif type == 'size':
            self.manager.weightsTemplate = output_dir + '/embed/predS_{}{{}}_{{}}.p'.format(pre)
        else:
            print("{} - illegal pred type".format(type))
            assert(False)

    def read(self, run=None, dset=None):
        return self.manager.read(run, dset)

    def load(self, run=None, dset=None):
        return self.manager.load(run, dset)

    def write(self, run=None, dset=None):
        return self.manager.write(run, dset)

    def name(self, run=None, dset=None):
        return self.manager.name(run, dset)

    __call__ = name


class Dataset(object):

    def __init__(self, data_type, conf, dir='/Dataset'):
        self.weightsTemplate = dir + '/Dataset{}CV{}_{{:.0f}}-{{}}-{{}}.p'.format(data_type, conf)

    def read(self, size, res, sample='Normal'):
        match = self.weightsTemplate.format(size, res, sample)
        filelist = glob(match)
        if len(filelist) == 0:
            print("Failed to find: {}".format(match))
            return None
        return open(filelist[0], 'br')

    def load(self, size, res, sample='Normal'):
        data = pickle.load(self.read(size, res, sample))
        return data

    def write(self, size, res, sample='Normal'):
        match = self.weightsTemplate.format(size, res, sample)
        return open(match, 'bw')

    def name(self, size, res, sample='Normal'):
        return self.weightsTemplate.format(size, res, sample)

    __call__ = name


class Dataset3d(object):

    def __init__(self, conf, dir='./Dataset'):
        data_type = '3d'
        self.weightsTemplate = dir + '/Dataset{}CV{}-{{}}_{{}}{{}}-{{}}.p'.format(data_type, conf)

    def read(self, dset, net, run, epoch):
        match = self.weightsTemplate.format(dset, net, run, epoch)
        filelist = glob(match)
        if len(filelist) == 0:
            print("Failed to find: {}".format(match))
            return None
        return open(filelist[0], 'br')

    def load(self, dset, net, run, epoch):
        data = pickle.load(self.read(dset, net, run, epoch))
        return data

    def write(self,  dset, net, run, epoch):
        match = self.weightsTemplate.format( dset, net, run, epoch)
        return open(match, 'bw')

    def name(self,  dset, net, run, epoch):
        return self.weightsTemplate.format( dset, net, run, epoch)

    __call__ = name


class DatasetFromPredication(object):

    def __init__(self, type='rating', pre='dirR', input_dir='./output'):  # data_type, conf
        # self.weightsTemplate = dir + '/Dataset{}CV{}_{{:.0f}}-{{}}-{{}}.p'.format(data_type, conf)
        if type == 'rating':
            self.weightsTemplate = input_dir + '/embed/predR_{}{{}}_{{}}.p'.format(pre)
        elif type == 'malig':
            assert False
            # self.weightsTemplate = input_dir + '/embed/pred_{}{{}}_{{}}.p'.format(pre)
        elif type == 'size':
            assert False
            # self.weightsTemplate = input_dir + '/embed/predS_{}{{}}_{{}}.p'.format(pre)
        else:
            print("{} - illegal pred type".format(type))
            assert False

    def read(self, run=None, dset=None):
        match = self.weightsTemplate.format(run, dset)
        filelist = glob(match)
        if len(filelist) == 0:
            print("Failed to find: {}".format(match))
            return None
        return open(filelist[0], 'br')

    def load(self, goal, run=None, epoch=None):
        dset = 'Valid' if goal is 'Train' else 'Test'
        data = pickle.load(self.read(run, dset))
        # convert to Dataset
        all_preds, epochs_done, meta, images, classes, labels, masks, conf, rating_weights, z = data

        #print(epoch)
        #print(epochs_done)

        epoch_idx = np.argwhere(epoch == np.array(epochs_done))[0][0]
        pred = all_preds[epoch_idx]

        Dataset = []
        for i in range(len(meta)):
            Entry = dict()
            Entry['patch'] = images[i]
            Entry['mask'] = masks[i]
            Entry['label'] = classes[i]
            Entry['info'] = meta
            Entry['size'] = None
            Entry['rating'] = np.expand_dims(pred[i], axis=0) if goal in ['Train', 'Valid'] else labels[i]
            Entry['weights'] = [1] if goal in ['Train', 'Valid'] else rating_weights[i]
            Entry['z'] = z[i]
            Dataset.append(Entry)

        return Dataset

    #def write(self, run=None, dset=None):
    #    match = self.weightsTemplate.format(run, dset)
    #    return open(match, 'bw')

    def name(self, goal, run=None):
        dset = 'Valid' if goal is 'Train' else 'Test'
        return self.weightsTemplate.format(run, dset)

    __call__ = name
