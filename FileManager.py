from glob import glob


class Weights(object):

    def __init__(self, pre):
        self.weightsTemplate = 'Weights/w_{}{{}}_{{}}-{{}}-{{}}.h5'.format(pre)

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
            while name is None:
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