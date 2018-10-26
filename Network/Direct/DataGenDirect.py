import numpy as np
try:
    from Network.Common.dataGenBase import DataGeneratorBase, DataSequenceBase
    from Network.data_loader import load_nodule_dataset
    from Network.Direct.prepare_data import prepare_data, prepare_data_direct
    from Network.dataUtils import augment_all, crop_center, get_sample_weight, get_class_weight
    from Network.dataUtils import rating_clusters_distance_matrix
except:
    from Common.dataGenBase import DataGeneratorBase, DataSequenceBase
    from data_loader import load_nodule_dataset
    from Direct.prepare_data import prepare_data, prepare_data_direct
    from dataUtils import augment_all, crop_center, get_sample_weight, get_class_weight
    from dataUtils import rating_clusters_distance_matrix


def select_balanced(some_set, labels, N, permutation):
    b = some_set[0 == labels][:N]
    m = some_set[1 == labels][:N]
    merged = np.concatenate([b, m], axis=0)
    reshuff = merged[permutation]
    return reshuff


class DataGeneratorDir(DataGeneratorBase):
    """docstring for DataGenerator"""

    def __init__(self,  load_dataset_fn, data_size= 128, model_size=128, batch_size=32,
                        objective='malignancy', rating_scale='none', categorize=False,
                        weighted_rating=False,
                        do_augment=False, augment=None,
                        use_class_weight=False, use_confidence=False,
                        val_factor = 0, train_factor=1, balanced=False,
                        seq_model=False):

        assert categorize is False

        super().__init__(load_dataset_fn, data_size= data_size, model_size=model_size, batch_size=batch_size,
                        objective=objective, rating_scale=rating_scale, categorize=categorize,
                        weighted_rating=weighted_rating,
                        do_augment=do_augment, augment=augment,
                        use_class_weight=use_class_weight, use_confidence=use_confidence,
                        val_factor = val_factor, balanced=balanced, train_factor=train_factor,
                        seq_model=seq_model)

    def get_sequence(self):
        return DataSequenceDir

    def get_data(self, dataset, is_training):
        return prepare_data_direct(dataset, objective=self.objective, reshuffle=False,
                                   rating_scale=self.rating_scale, num_of_classes=2, verbose=True)


class DataSequenceDir(DataSequenceBase):

    def __init__(self, dataset, is_training=True, model_size=128, batch_size=32,
                 objective='malignancy', rating_scale='none', categorize=False,
                 do_augment=False, augment=None, weighted_rating=False,
                 use_class_weight=False, use_confidence=False,
                 balanced=False, data_factor=1, seq_model=False):

        assert (categorize is False)

        if objective == 'rating':
            assert balanced is False

        super().__init__(dataset, is_training=is_training, model_size=model_size, batch_size=batch_size,
                         objective=objective, rating_scale=rating_scale, categorize=categorize,
                         do_augment=do_augment, augment=augment, weighted_rating=weighted_rating,
                         use_class_weight=use_class_weight, use_confidence=use_confidence,
                         balanced=balanced, data_factor=data_factor, seq_model=seq_model)

    def calc_N(self, data_factor):

        if self.objective == 'malignancy':
            labels = np.array([entry[2] for entry in self.dataset])
            Nb = np.count_nonzero(1 - labels)
            Nm = np.count_nonzero(labels)

            if self.is_training:
                if self.balanced:
                    N = 2 * np.minimum(Nb, Nm) // self.batch_size
                else:
                    N = (Nb + Nm) // self.batch_size
            else:
                N = len(self.dataset) // self.batch_size

        elif self.objective in ['rating', 'size', 'rating_size', 'distance-matrix']:
            N = len(self.dataset) // self.batch_size
            if self.balanced:
                print("WRN: objective rating does not support balanced")
                assert False
            self.balanced = False
            self.use_class_weight = self.use_class_weight

        N *= data_factor

        return N

    def load_data(self):

        rating_format = 'mean' if self.seq_model else 'w_mean'
        if rating_format == 'mean':
            print("WRN: rating_format is 'mean' instead of 'w_mean'");

        images, labels, classes, masks, meta, weights = \
            prepare_data_direct(self.dataset,
                                objective=self.objective, rating_scale=self.rating_scale, rating_format=rating_format,
                                num_of_classes=2, verbose=self.verbose, weighted_rating=self.weighted_rating)

        sample_weights = np.ones(len(classes))
        if self.use_class_weight:
            class_weights = get_class_weight(np.squeeze(classes), 'balanced')
            if len(class_weights) == 2:
                print("Class Weight -> Benign: {:.2f}, Malignant: {:.2f}".format(class_weights[0], class_weights[1]))
            elif len(class_weights) == 3:
                class_weights[2] = (class_weights[0] + class_weights[1]) / 6
                print("Class Weight -> Benign: {:.2f}, Malignant: {:.2f}, Unknown: {:.2f}".format(class_weights[0], class_weights[1], class_weights[2]))
            sample_weights *= get_sample_weight(classes, class_weights)
        if self.use_confidence:
            print('use confidence')
            sample_weights *= weights


        Nb = np.count_nonzero(1 - classes)
        Nm = np.count_nonzero(classes)
        N = np.minimum(Nb, Nm)
        if self.verbose:
            print("Benign: {}, Malignant: {}".format(Nb, Nm))
        if self.balanced and self.is_training:
            new_order = np.random.permutation(2 * N)
            labels_ = np.argmax(classes, axis=1)
            images = select_balanced(images, labels_, N, new_order)
            labels = select_balanced(labels, labels_, N, new_order)
            classes = select_balanced(classes, labels_, N, new_order)
            masks = select_balanced(masks, labels_, N, new_order)
            sample_weights = select_balanced(sample_weights, labels_, N, new_order)

            if self.verbose:
                Nb = np.count_nonzero(1 - np.argmax(classes, axis=1))
                Nm = np.count_nonzero(np.argmax(classes, axis=1))
                print("Balanced - Benign: {}, Malignant: {}".format(Nb, Nm))

        return (images,), labels if type(labels) is tuple else tuple([labels]), classes, (masks,), sample_weights

    def process_label_batch(self, labels_batch, weights_batch=None):
        if self.objective == 'distance-matrix':
            labels_batch = rating_clusters_distance_matrix(labels_batch, weights=weights_batch)
        return labels_batch