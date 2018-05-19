import numpy as np
try:
    from Network.Common.dataGenBase import DataGeneratorBase, DataSequenceBase
    from Network.data_loader import load_nodule_dataset, prepare_data_siamese, prepare_data_siamese_simple, prepare_data
    from Network.dataUtils import augment_all, crop_center, get_sample_weight_for_similarity, get_class_weight
    from Network.Siamese.metrics import siamese_rating_factor
except:
    from Common.dataGenBase import DataGeneratorBase, DataSequenceBase
    from data_loader import load_nodule_dataset, prepare_data_siamese, prepare_data_siamese_simple, prepare_data
    from dataUtils import augment_all, crop_center, get_sample_weight, get_class_weight
    from Siamese.metrics import siamese_rating_factor


class DataGeneratorSiam(DataGeneratorBase):
    """docstring for DataGenerator"""

    def __init__(self,  data_size= 128, model_size=128, res='Legacy', sample='Normal', batch_size=32, objective="malignancy",
                        categorize=False, rating_scale='none',
                        do_augment=False, augment=None, use_class_weight=False, use_confidence=False, debug=False,
                        val_factor = 1, balanced=False, configuration=None):

        super().__init__(data_size=data_size, model_size=model_size, res=res, sample=sample, batch_size=batch_size,
                         objective=objective, rating_scale=rating_scale, categorize=categorize,
                         do_augment=do_augment, augment=augment,
                         use_class_weight=use_class_weight, use_confidence=use_confidence,
                         val_factor=val_factor, balanced=balanced, configuration=configuration,
                         debug=debug)

    def get_sequence(self):
        return DataSequenceSiam

    def get_data(self, dataset, is_training):
        if self.objective == "malignancy":
            data = prepare_data_siamese(dataset, balanced=(self.balanced and is_training),
                                        objective=self.objective, verbose=True, return_meta=True)
        elif self.objective == "rating":
            data = prepare_data_siamese_simple(dataset, balanced=(self.balanced and is_training),
                                               objective=self.objective, verbose=True, return_meta=True)
            data[1] *= siamese_rating_factor
        return data


class DataSequenceSiam(DataSequenceBase):
    def __init__(self, dataset, is_training=True, model_size=128, batch_size=32,
                 objective="malignancy", rating_scale='none', categorize=False,
                 do_augment=False, augment=None, use_class_weight=False, use_confidence=False, debug=False,
                 val_factor=1, balanced=False):

        assert use_confidence is False
        assert categorize is False

        if objective == 'rating':
            assert balanced is False
            assert use_class_weight is False

        super().__init__(dataset, is_training=is_training, model_size=model_size, batch_size=batch_size,
                         objective=objective, rating_scale=rating_scale, categorize=categorize,
                         do_augment=do_augment, augment=augment,
                         use_class_weight=use_class_weight, use_confidence=use_confidence,
                         balanced=balanced, val_factor=val_factor)

    def calc_N(self, val_factor):
        if self.objective == "malignancy":
            if self.is_training:
                classes = np.array([entry[2] for entry in self.dataset])
                Nb = np.count_nonzero(1 - classes)
                Nm = np.count_nonzero(classes)
                if self.balanced:
                    N = 4 * np.minimum(Nb, Nm) // self.batch_size
                else:
                    N = (2 * np.minimum(Nb, Nm) + len(self.dataset)) // self.batch_size
            else:
                N = val_factor * 2 * len(self.classes) // self.batch_size
        elif self.objective == "rating":
            if self.is_training:
                N = len(self.dataset) // self.batch_size
            else:
                N = val_factor * (len(self.dataset) // self.batch_size)

        return N

    def load_data(self):

        if self.objective == "malignancy":
            images, labels, masks, confidence = \
                prepare_data_siamese(self.dataset, balanced=(self.balanced and self.is_training),
                                     objective=self.objective, verbose=self.verbose)
        elif self.objective == "rating":
            images, labels, masks, confidence = \
                prepare_data_siamese_simple(self.dataset, balanced=(self.balanced and self.is_training),
                                            objective=self.objective, verbose=self.verbose, siamese_rating_factor=siamese_rating_factor)
            labels *= siamese_rating_factor

        if self.use_class_weight:
            class_weight = get_class_weight(confidence, method='balanced')
            sample_weight = get_sample_weight_for_similarity(confidence, wD=class_weight['D'], wSB=class_weight['SB'],
                                               wSM=class_weight['SM'])
        else:
            sample_weight = np.ones(labels.shape)

        return images, labels, [None]*len(labels), masks, sample_weight


