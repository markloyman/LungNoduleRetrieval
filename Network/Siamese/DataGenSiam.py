import numpy as np
try:
    from Network.Common.dataGenBase import DataGeneratorBase, DataSequenceBase
    from Network.data_loader import load_nodule_dataset
    from Network.Siamese.prepare_data import prepare_data_siamese, prepare_data_siamese_simple, prepare_data
    from Network.dataUtils import augment_all, crop_center, get_sample_weight_for_similarity, get_class_weight
    from Network.Siamese.metrics import siamese_rating_factor
except:
    from Common.dataGenBase import DataGeneratorBase, DataSequenceBase
    from data_loader import load_nodule_dataset
    from Siamese.prepare_data import prepare_data_siamese, prepare_data_siamese_simple, prepare_data
    from dataUtils import augment_all, crop_center, get_sample_weight, get_class_weight
    from Siamese.metrics import siamese_rating_factor


class DataGeneratorSiam(DataGeneratorBase):
    """docstring for DataGenerator"""

    def __init__(self,  load_dataset_fn, data_size= 128, model_size=128, batch_size=32, objective="malignancy",
                        categorize=False, rating_scale='none', weighted_rating=False,
                        do_augment=False, augment=None, use_class_weight=False, use_confidence=False,
                        val_factor = 1, train_facotr = 1, balanced=False, seq_model=False):

        super().__init__(load_dataset_fn, data_size=data_size, model_size=model_size, batch_size=batch_size,
                         objective=objective, rating_scale=rating_scale, categorize=categorize,
                         do_augment=do_augment, augment=augment, weighted_rating=weighted_rating,
                         use_class_weight=use_class_weight, use_confidence=use_confidence,
                         val_factor=val_factor, balanced=balanced, train_factor=train_facotr,
                         seq_model=seq_model)

    def get_sequence(self):
        return DataSequenceSiam

    def get_data(self, dataset, is_training):
        if self.objective == "malignancy":
            data = prepare_data_siamese(dataset, balanced=(self.balanced and is_training),
                                        objective=self.objective, verbose=True)
        else:
            data = prepare_data_siamese_simple(dataset, rating_distance='weighted_clusters' if self.weighted_rating else 'clusters',
                                               objective=self.objective, verbose=True)
            data[1] *= siamese_rating_factor
        return data


class DataSequenceSiam(DataSequenceBase):
    def __init__(self, dataset, is_training=True, model_size=128, batch_size=32,
                 objective="malignancy", rating_scale='none', categorize=False, weighted_rating=False,
                 do_augment=False, augment=None, use_class_weight=False, use_confidence=False, debug=False,
                 data_factor=1, balanced=False, seq_model=False):

        #assert use_confidence is False
        assert categorize is False

        if objective == 'rating':
            assert balanced is False
            assert use_class_weight is False

        super().__init__(dataset, is_training=is_training, model_size=model_size, batch_size=batch_size,
                         objective=objective, rating_scale=rating_scale, categorize=categorize,
                         do_augment=do_augment, augment=augment, weighted_rating=weighted_rating,
                         use_class_weight=use_class_weight, use_confidence=use_confidence,
                         balanced=balanced, data_factor=data_factor, seq_model=False)

    def calc_N(self, data_factor=1):
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
                N = len(self.classes) // self.batch_size
        else:
            N = len(self.dataset) // self.batch_size

        N *= data_factor

        return N

    def load_data(self):

        if self.objective == "malignancy":
            images, labels, masks, confidence, meta = \
                prepare_data_siamese(self.dataset, balanced=(self.balanced and self.is_training),
                                     objective=self.objective, verbose=self.verbose)
            if self.use_class_weight:
                class_weight = get_class_weight(confidence, method='balanced')
                sample_weight = get_sample_weight_for_similarity(confidence, wD=class_weight['D'], wSB=class_weight['SB'],
                                                   wSM=class_weight['SM'])
            else:
                sample_weight = np.ones(labels.shape)
        # elif self.objective == "rating":
        else:
            images, labels, masks, confidence, meta = \
                prepare_data_siamese_simple(self.dataset, rating_distance='weighted_clusters' if self.weighted_rating else 'clusters',
                                            objective=self.objective, verbose=self.verbose, siamese_rating_factor=siamese_rating_factor)

            if self.use_confidence:
                sample_weight = confidence
            else:
                sample_weight = np.ones(labels[0].shape)

        # print('sample weights: {}'.format(sample_weight[:10]))

        return images, labels if type(labels) is tuple else tuple([labels]), [None]*len(labels), masks, sample_weight


