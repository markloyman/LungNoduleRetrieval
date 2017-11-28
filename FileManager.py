from glob import glob
import pickle
import numpy as np


class Weights(object):

    def __init__(self, pre):
        self.weightsTemplate = './output/Weights/w_{}{{}}_{{}}-{{}}-{{}}.h5'.format(pre)

    def read(self, run=None, epoch=None):
        match = self.weightsTemplate.format(run,epoch,'*','*')
        return open(glob(match)[0], 'br')

    def write(self, run=None, epoch=None, loss=None, val_loss=None):
        match = self.weightsTemplate.format(run, epoch, loss, val_loss)
        return open(match, 'bw')

    def name(self, run=None, epoch=None, loss=None, val_loss=None):
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
                    else:
                        epoch -= 1
        else:
            name = self.weightsTemplate.format(run, epoch, loss, val_loss)

        return name

    __call__ = name


class Embed(object):

    def __init__(self, pre):
        self.weightsTemplate = './output/embed/embed_{}{{}}-{{}}_{{}}.p'.format(pre)

    def read(self, run=None, epoch=None, dset=None):
        match = self.weightsTemplate.format(run, epoch, dset)
        return open(glob(match)[0], 'br')

    def load(self, run=None, epoch=None, dset=None):
        images, embed, meta, labels, masks = pickle.load(self.read(run=run, epoch=epoch, dset=dset))
        if np.max(images) > 5:
            images = images.astype('float') / 50.0
        if np.max(embed) > 1:
            images = images.astype('float') / 1000.0
        return images, embed, meta, labels, masks

    def write(self, run=None, epoch=None, dset=None):
        match = self.weightsTemplate.format(run, epoch, dset)
        return open(match, 'bw')

    def name(self, run=None, epoch=None,  dset=None):
        return self.weightsTemplate.format(run, epoch, dset)

    __call__ = name
