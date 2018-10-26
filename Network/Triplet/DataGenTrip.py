import numpy as np
try:
    from Network.Common.dataGenBase import DataGeneratorBase, DataSequenceBase
    from Network.data_loader import load_nodule_dataset
    from Network.Triplet.prepare_data import prepare_data_triplet, prepare_data
    from Network.dataUtils import augment_all, crop_center, get_sample_weight_for_similarity, get_class_weight
except:
    from Common.dataGenBase import DataGeneratorBase, DataSequenceBase
    from data_loader import load_nodule_dataset
    from Triplet.prepare_data import prepare_data_triplet, prepare_data
    from dataUtils import augment_all, crop_center, get_sample_weight, get_class_weight


class DataGeneratorTrip(DataGeneratorBase):
    """docstring for DataGenerator"""

    def __init__(self,  load_dataset_fn, data_size= 128, model_size=128, batch_size=32,
                        categorize=False, rating_scale='none', balanced=False,
                        do_augment=False, augment=None, use_class_weight=False, use_confidence=False,
                        val_factor=1, train_factor=1, objective="malignancy"):

        super().__init__(load_dataset_fn, data_size=data_size, model_size=model_size, batch_size=batch_size,
                         objective=objective, rating_scale=rating_scale, categorize=categorize,
                         do_augment=do_augment, augment=augment,
                         use_class_weight=use_class_weight, use_confidence=use_confidence,
                         val_factor=val_factor, train_factor=train_factor, balanced=balanced)

    def get_sequence(self):
        return DataSequenceTrip

    def get_data(self, dataset, is_training):
        ret_conf = self.class_weight_method if self.use_class_weight else None
        data = prepare_data_triplet(set, verbose=self.verbose, objective=self.objective, return_confidence=ret_conf)
        return data


class DataSequenceTrip(DataSequenceBase):
    def __init__(self, dataset, is_training=True, model_size=128, batch_size=32,
                 objective="malignancy", rating_scale='none', categorize=False,
                 do_augment=False, augment=None, use_class_weight=False, use_confidence=False, debug=False,
                 data_factor=1, balanced=False):

        assert use_class_weight is False
        assert use_confidence is False

        if objective == 'malignancy':
            assert categorize is True

        super().__init__(dataset, is_training=is_training, model_size=model_size, batch_size=batch_size,
                         objective=objective, rating_scale=rating_scale, categorize=categorize,
                         do_augment=do_augment, augment=augment,
                         use_class_weight=use_class_weight, use_confidence=use_confidence,
                         balanced=balanced, data_factor=data_factor)

    def calc_N(self, data_factor):
        if self.is_training:
            # make sure N is correct
            if self.balanced:
                classes = np.array([entry[2] for entry in self.dataset])
                Nb = np.count_nonzero(1 - classes)
                Nm = np.count_nonzero(classes)
                M = min(Nb, Nm)
                N = 2 * M // self.batch_size
                print("VERIFY N IS CORRECT!!!")
            else:
                N = len(self.dataset) // self.batch_size
                assert False
        else:
            N = len(self.dataset) // self.batch_size

        N *= data_factor

        return N

    def load_data(self):
        assert self.use_class_weight is False
        ret_conf = self.class_weight_method if self.use_class_weight else None
        images, labels, masks, confidence = \
            prepare_data_triplet(self.dataset, objective=self.objective, balanced=self.balanced,
                                 return_confidence=ret_conf, verbose=self.verbose)

        if self.use_class_weight:
            class_weight = get_class_weight(confidence, method='balanced')
            sample_weight = get_sample_weight_for_similarity(confidence, wD=class_weight['D'], wSB=class_weight['SB'],
                                               wSM=class_weight['SM'])
        else:
            sample_weight = np.ones(labels.shape)

        return images, labels, [None]*len(labels), masks, sample_weight
