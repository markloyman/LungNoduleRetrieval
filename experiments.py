import itertools


class CrossValidationManager:
    def __init__(self, config_name):
        self.config = []
        self.n_groups = 5
        self.group_ids = range(self.n_groups)
        self.config_name = config_name

        target_set = [4, 3, 2, 1, 0, 0, 2, 4, 3, 1]

        predication_train_combinations = [a for a in itertools.combinations(self.group_ids, 2)]

        for predication_train, target in zip(predication_train_combinations, target_set):
            predication_eval = [g for g in self.group_ids if g not in predication_train]
            #predication_valid = predication_eval[0]
            retrieval_train_combinations = [a for a in itertools.combinations(predication_eval, 2) if target not in a]
            for retrieval_train in retrieval_train_combinations:
                self.add_to_config(predication_train, retrieval_train)

        #print(self.config)

    def add_to_config(self, predication_train, retrieval_train):
        cnf = dict()

        predication_eval = [g for g in self.group_ids if g not in predication_train]
        cnf['prediction_train'] = predication_train
        cnf['prediction_valid'] = retrieval_train
        cnf['prediction_eval'] = predication_eval

        target = [g for g in predication_eval if g not in retrieval_train]
        cnf['retrieval_train'] = retrieval_train
        cnf['target'] = target

        self.config.append(cnf)

        return cnf

    def get_prediction_train(self, index):
        cnf = self.config[index ]
        return cnf['prediction_train']

    def get_prediction_validation(self, index):
        cnf = self.config[index ]
        return cnf['prediction_valid']

    def get_prediction_eval(self, index):
        cnf = self.config[index ]
        return cnf['prediction_eval']

    def get_retrieval_train(self, index):
        cnf = self.config[index ]
        return cnf['retrieval_train']

    def get_target(self, index):
        cnf = self.config[index]
        return cnf['target']


def load_experiments(experiment):
    runs, run_net_types, run_metrics, run_epochs, run_names, run_ep_perf, run_ep_comb = [], [], [], [], [], [], []
    # ===========================
    #   Malignancy Objective
    # ===========================
    if experiment == 'MalignancyObjective':
        runs            = ['103', '100', '011XXX']
        run_net_types   = ['dir', 'siam', 'trip']
        run_metrics     = ['l2']*len(runs)
        run_epochs      = [ [5, 10, 15, 20, 25, 30, 35, 40, 45],
                            [5, 10, 15, 20, 25, 30, 35, 40, 45],
                            [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
                        ]
        run_names       = run_net_types
    # ===========================
    #   Triplet Compare to dirR reference
    # ===========================
    elif experiment == 'TripletToDirR':
        runs            = ['011X', '011XXX', '016XXXX', '023X']
        run_net_types   = ['dirR', 'trip', 'trip', 'trip']
        run_metrics     = ['l2']*len(runs)
        run_epochs      = [ [5, 15, 25, 35, 45, 55],
                            [5, 15, 25, 35, 45, 55],
                            [20, 40, 100, 150, 180],
                            [5, 15, 20, 25, 30, 35]
                        ]
        run_names       = ['dirR', 'malig-obj', 'trip', 'trip-finetuned']
    # ===========================
    #   Triplets
    # ===========================
    elif experiment == 'Triplets':
        runs = ['000']
        run_net_types   = ['tripR']*len(runs)
        run_metrics     = ['l2']*len(runs)
        run_epochs = [list(range(1, 161, 1))] * len(runs)
        run_names = ['rmac']
        # runs            = ['011XXX', '016XXXX', '027', '023X']
        # run_epochs      = [ [5, 15, 25, 35, 45, 55],
        #                    [20, 40, 100, 150, 180],
        #                    [5, 15, 25, 35, 40, 45, 50, 55, 60],
        #                    [5, 15, 20, 25, 30, 35]
        #                ]
        # run_names       = ['malig-obj', 'rating-obj', 'rating-obj', 'trip-finetuned']
    # ===========================
    #   Pooling
    # ===========================
    elif experiment == 'Pooling':
        runs            = ['251', '235', '252']  # '250', '900', '901', '999', '902',
        run_net_types   = ['dir']*len(runs)
        run_metrics     = ['l2']*len(runs)
        run_epochs      = [list(range(1, 61, 1))]*len(runs)
        run_names       = ['251-avg', '235-max', '252-rmac']  # '250-max2', '900-max', '901-avg', '999-rmac', '902-msrmac',
        run_ep_perf      = [[65], [19], [45]]
        run_ep_comb      = [[38], [33], [18]]
    # ===========================
    #   PoolingAug
    # ===========================
    elif experiment == 'PoolingAug':
        runs = ['253', '243b', '254']
        run_net_types = ['dir'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 71, 1))] * len(runs)
        run_names = ['253-avg', '243b-max', '254-rmac']
    # ===========================
    #   Output Size
    # ===========================
    elif experiment == 'OutputSize':
        runs = ['212', '210', '900', '211']
        run_net_types = ['dir'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 37, 1))] * len(runs)
        run_names = ['212-32', '210-64', '900-128', '211-256']
    # ===========================
    #   New Network Atchitectures
    # ===========================
    elif experiment == 'NewNetwork':
        runs = ['900', '220', '223', '224'] # '221', '222'
        run_net_types = ['dir'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 46, 1))] * len(runs)
        run_names = ['900-base', '220-w256', '223-seq-aug', '224-no-rot'] # '221-dataug', '222-data-aug'
    # ===========================
    #   Dropout
    # ===========================
    elif experiment == 'Dropout':
        runs = ['225', '226', '227', '228']
        run_net_types = ['dir'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 61, 1))] * len(runs)
        run_names = [ '225-drp.5', '226-drp.3', '227-drp.0', '228-drp.1']
    # ===========================
    #   Rotation
    # ===========================
    elif experiment == 'Rotation':
        runs = ['235', '236', '240', '240b', '242', '243b'] #['228', '224', '229', '230', '231', '232', '234', '233', '235']
        run_net_types = ['dir'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 71, 1))] * len(runs)
        run_names = ['235-baseline', '236-128', '240-rot0', '240b-ro0', '242-rot40', '243b-rot60'] #['228-rot0', '224-rot5', '229-rot10', '230-rot20', '231-rot30', '232-no-rot', '234-rot30', '233-no-rot', '235-baseline']
    # ===========================
    #   Repeat
    # ===========================
    elif experiment == 'Repeat160':
        runs = ['235', '235e', '235f', '235g', '236'] # '235b', '235c', '235d'
        run_net_types = ['dir'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 41, 1))] * len(runs)
        run_names = ['235-m', '235e-f', '235f-f', '235g-f', '236-128'] # '235b-f0', '235c-f0', '235d-f0'
    # ===========================
    #   SiamWTF
    # ===========================
    elif experiment == 'SiamWTF':
        runs = ['243b', '202', '203', '203b', '202b', '202c']
        run_net_types = ['dir', 'siam', 'siam', 'siam', 'siam', 'siam', 'siam']
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 71, 1))] * len(runs)
        run_names = ['243b-dir', '202-siam-net-w512', '203-data-old', '203b-eval-on-new', '202b-eval-on-old', '202c-full-config']
    # ===========================
    #   Siam
    # ===========================
    elif experiment == 'Siam':
        runs = ['207', '200b', '204', '208', '205', '206']
        #runs = ['200b', '204', '207', '210', '214']
        run_net_types = ['siam'] * len(runs)
        run_metrics = ['l2'] * 3 + ['l1'] * 3
        run_epochs = [list(range(1, 71, 1))] * len(runs)
        run_names = ['207-l2-avg', '200b-l2-max', '204-l2-rmac', '208-l1-avg', '205-siam-l1-max', '206-l1-rmac']
        #run_names = ['200b-l2-max', '204-l2-rmac', '207-l2-avg', '210-l2-max-alt-loss', '214-l2-rmac-alt-loss']
        run_ep_perf = []
        run_ep_comb = []
    # ===========================
    #   SiamAug
    # ===========================
    elif experiment == 'SiamAug':
        # runs = ['202c', '204', '207', '205', '206', '208', '210']
        runs = ['200b', '204', '220', '224']
        run_net_types = ['siam'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 71, 1))] * len(runs)
        run_names = ['200b-l2-max', '204-l2-rmac', '220-l2-max-aug', '224-l2-rmac-aug']
    # ===========================
    #   Siam
    # ===========================
    elif experiment == 'SiamAltLoss':
        runs = ['200b', '204', '210b', '214', '230', '234']
        run_net_types = ['siam'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 81, 1))] * len(runs)
        run_names = ['200b-l2-max', '204-l2-rmac', '210b-max-alt-loss', '214-rmac-alt-loss', '230-max-alt-aug', '234-rmac-alt-aug']
    # ===========================
    #   Siam
    # ===========================
    elif experiment == 'SiamAltLossMax':
        runs = ['200b', '220', '210b', '230']
        run_net_types = ['siam'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 81, 1))] * len(runs)
        run_names = ['200b-l2-max', '220-max-aug', '210b-max-alt-loss', '230-max-alt-aug']
    # ===========================
    #   Siam
    # ===========================
    elif experiment == 'SiamAltLossRmac':
        runs = ['204', '224', '214', '234']
        run_net_types = ['siam'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 81, 1))] * len(runs)
        run_names = ['204-l2-rmac', '224-rmac-aug', '214-rmac-alt-loss', '234-rmac-alt-aug']
    # ===========================
    #   Dir-Rating
    # ===========================
    elif experiment == 'DirRatingPooling': # objective: mean-rating
        runs = ['202c', '200b', '201']  # '202',
        run_net_types = ['dirR']*len(runs)
        run_metrics = ['l2']*len(runs)
        run_epochs = [list(range(1, 101, 1))] * len(runs)
        run_names = ['202c-avg', '200b-max', '201-rmac']  # '202-avg',
        run_ep_perf = [[35], [33], [100]]
        run_ep_comb = [[17], [29], [55]]
    # ===========================
    #   Dir-Rating-Max-Mean
    # ===========================
    elif experiment == 'DirRatingMax':
        runs = ['203', '213', '223', '253']
        run_net_types = ['dirR']*len(runs)
        run_metrics = ['l2']*len(runs)
        run_epochs = [list(range(1, 101, 1))] * len(runs)
        run_names = ['203-max', '213-aug', '223-aug-with_unknowns', '253-aug-with_unknowns-no_class_weight']
        #run_names = ['200b-max', '210-max-aug', '203-max-w_mean', '213-aug-w_mean']
        #run_names = ['200b-max', '210-max-aug', '220-max-aug-prm', '203-max-w_mean', '223-max-aug-prm']  # '202-avg',
        run_ep_perf = [[35], [33], [100]]
        run_ep_comb = [[17], [29], [55]]
    # ===========================
    #   Dir-Rating-Primary
    # ===========================
    elif experiment == 'DirRatingPrimary':
        runs = ['233', '243', '253', '251', '300']
        run_net_types = ['dirR'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 101, 1))] * len(runs)
        run_names = ['233-w_mean-class_weight-conf', '243-w_mean-conf', '253-w_mean', '251-msrmac', '300-sep']
        # '220-mean-bad_class_weight', '223-w_mean-bad_class_weight', '202-avg', '253-aug-prm-no-class-w', '243-aug-prm-conf-no-weight'
        run_ep_perf = [[35], [33], [100]]
        run_ep_comb = [[17], [29], [55]]
    # ===========================
    #   Dir-Rating-Reg
    # ===========================
    elif experiment == 'DirRatingReg':
        runs = ['270', '272', '275', '276', '277']  # '271', '260', '263', '273',
        run_net_types = ['dirR'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 101, 1))] * len(runs)
        run_names = ['270-base', '272-samp.1', '275-feat.1', '276-feat.schd1', '277-samp.schd1']  # '271-base128', '260-baseline', '263-l1.001', '273-samp.5'
        #run_ep_perf = [[35], [33], [100]]
        #run_ep_comb = [[17], [29], [55]]
    # ===========================
    #   Dir-Objectives
    # ===========================
    elif experiment == 'DirObj':
        runs = [ '251', '512c', '705', '709', '710', '720']  # '270', '412', '502', '503',
        run_net_types = ['dirR', 'dirRS', 'dirD', 'dirD', 'dirD', 'dirD'] # 'dirR', 'dirS', 'dirRS', 'dirRS',
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 161, 1))] * len(runs)
        run_names = [ 'R251-base-aug-primary', 'RS512c-aug-primary', 'D705-l2corr', 'D709-dm-label', 'D710-dm-lbl-pre:dirR251-20', 'D720-weighted-dm']  # '270-base', '412-size', '502-rating-size', '503-rating-size-02',
        # 'D706-l2corr-pre:dirR251-60', 'D707-lr-4', 'D708-lr-4-b64',
    # ===========================
    #   Dir-Full
    # ===========================
    elif experiment == 'DirFull':
        runs = [ '251', '601b', '602', '603', '604', '606']  # '270', '412', '502', '503',
        run_net_types = ['dirR'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 221, 1))] * len(runs)
        run_names = [ '251-base-aug-primary', '601b-full-conf', '602-full-conf-b64', '603-lr-4', '604-pre:dirR251', '606-pre:dirR251-lr-4']
    # ===========================
    #   Full-Dataset
    # ===========================
    elif experiment == 'FullDataset':
        runs = ['251', '601b']
        run_net_types = ['dirR'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 221, 1))] * len(runs)
        run_names = ['251-base-aug-primary', '601b-full-conf']
    # ===========================
    #   Siam-Rating-Reg
    # ===========================
    elif experiment == 'SiamRatingReg':
        runs = ['180', '181', '182', '200', '311']
        run_net_types = ['siamR'] *(len(runs)-1) + ['siamRS']
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 101, 1))] * len(runs)
        run_names = ['180-base', '181-feat.1', '182-samp.1', '200-pretrained', '311-rating-size']  # '271-base128', '260-baseline', '263-l1.001',
        # run_ep_perf = [[35], [33], [100]]
        # run_ep_comb = [[17], [29], [55]]
    # ===========================
    #   Rating-Size
    # ===========================
    elif experiment == 'RatingSize':
        runs = ['251', '512c', '132', '311']
        run_net_types = ['dirR', 'dirRS', 'siamR', 'siamRS']
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 121, 1))] * len(runs)
        run_names = ['251-dirR', '512c-dirRS', '132-siamR', '311-siamRS']
    # ===========================
    #   MaxPooling
    # ===========================
    elif experiment == 'MaxPooling':
        runs = ['235', '243b', '202c', '203', '213']  # '202', '253', '243'
        run_net_types = ['dir', 'dir', 'siam', 'dirR', 'dirR']
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 101, 1))] * len(runs)
        run_names = ['235-dir', '243b-dir-aug', '202c-siam-l2', '203-dirR', '213-dirR-aug']  # '202-avg', '253-aug-prm-no-class-w', '243-aug-prm-conf-no-weight'
        run_ep_perf = [[35], [33], [100]]
        run_ep_comb = [[17], [29], [55]]
    # ===========================
    #   PreTrain
    # ===========================
    elif experiment == 'PreTrain':
        runs = ['251', '601b', '132', '200']
        run_net_types = ['dirR', 'dirR', 'siamR', 'siamR']
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 161, 1))] * len(runs)
        run_names = ['251-dirR', '601b-dirR-full', '132-siamR', '200-siamR-pretrained']  # '202-avg', '253-aug-prm-no-class-w', '243-aug-prm-conf-no-weight'
    # ===========================
    #   Summary
    # ===========================
    elif experiment == 'Summary':
        runs = ['243b', '202c', '223', '132', '135']  # '202', '253', '243'
        run_net_types = ['dir', 'siam', 'dirR', 'siamR', 'siamR']
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 101, 1))] * len(runs)
        run_names = ['243b-dir-aug', '202c-siam-l2', '223-dirR-aug-primary', '132-siamrR-aug-primary', '135']  # '202-avg', '253-aug-prm-no-class-w', '243-aug-prm-conf-no-weight'
    # ===========================
    #   SummaryAlt
    # ===========================
    elif experiment == 'SummaryAlt':
        runs = ['254', '234', '251', '132']  # '202', '253', '243'
        run_net_types = ['dir', 'siam', 'dirR', 'siamR']
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 121, 1))] * len(runs)
        run_names = ['254-dir-aug', '234-siam-aug-alt', '251-dirR-aug-primary', '132-siamrR-aug-primary']
        run_ep_perf = [[30], [75], [70], [70]]
    # ===========================
    #   Siam-Rating
    # ===========================
    elif experiment == 'SiamRating':
        runs = ['110', '112', '122', '132', '135', '180', '200']
        run_net_types = ['siamR']*len(runs)
        run_metrics = ['l2']*len(runs)
        run_epochs = [list(range(1, 201, 1))] * len(runs)
        #run_names = ['100c-x1', '101b-x2', '103-x1-mse', '110-max-x3', '112-rmac-x3', '122-rmac-x2-aug', '132-aug-primary']
        run_names = ['110-max', '112-rmac', '122-rmac-aug', '132-rmac-aug-with_unknowns', '135-rmac-aug-uknwn-conf', '180-baseline', '200-pretrained']
    # ===========================
    #   Siam-Rating-Output
    # ===========================
    elif experiment == 'SiamRatingOutput':
        runs = ['162', '152', '142', '112', '172']
        run_net_types = ['siamR'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 101, 1))] * len(runs)
        run_names = ['162-o8', '152-o32', '142-o64', '112-o128', '172-o256']
    # ===========================
    #   Dir-Seq
    # ===========================
    elif experiment == 'DirSeq':
        runs = [ '0005'] # '0001',
        run_net_types = ['dirR'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 100, 1))] * len(runs)
        run_names = ['0005-flat-train']  # '0001-flat-valid',
    # ===========================
    #   SPIE-Summary
    # ===========================
    elif experiment == 'SpieSummary':
        runs = ['302', '312', '813', '410', '411']
        run_net_types = ['dir', 'siam', 'dirR', 'siamR', 'siamR']
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 101, 1))] * len(runs)
        run_names = ['dir-rmac', 'siam-l2-rmac', 'dirR-max-primary', 'siamR-l2-rmac-primary', 'siamR-cosine-rmac-primary']
        run_ep_perf = [[35], [35], [50], [80], [80]]
    # ===========================
    #   SPIE-Direct-Pooling
    # ===========================
    elif experiment == 'SpieDirPool':
        runs = ['300', '301', '302']
        run_net_types = ['dir'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 61, 1))] * len(runs)
        run_names = ['dir-avg', 'dir-max', 'dir-rmac']
        run_ep_perf = []
    # ===========================
    #   SPIE-Siamese-L1-Pooling
    # ===========================
    elif experiment == 'SpieSiamL1Pool':
        runs = ['300', '301', '302']
        run_net_types = ['siam'] * len(runs)
        run_metrics = ['l1'] * len(runs)
        run_epochs = [list(range(1, 61, 1))] * len(runs)
        run_names = ['siam-l1-avg', 'siam-l1-max', 'siam-11-rmac']
        run_ep_perf = []
    # ===========================
    #   SPIE-Siamese-L2-Pooling
    # ===========================
    elif experiment == 'SpieSiamL2Pool':
        runs = ['310', '311', '312']  # '314'
        run_net_types = ['siam'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 61, 1))] * len(runs)
        run_names = ['siam-l2-avg', 'siam-l2-max', 'siam-12-rmac']  # 'siam-l2-rmac-marginal'
        run_ep_perf = []
    # ===========================
    #   SPIE-Siamese-Cosine-Pooling
    # ===========================
    elif experiment == 'SpieSiamCosinePool':
        runs = ['320', '321', '322b']  # '324'
        run_net_types = ['siam'] * len(runs)
        run_metrics = ['cosine'] * len(runs)
        run_epochs = [list(range(1, 101, 1))] * len(runs)
        run_names = ['siam-cosine-avg', 'siam-cosine-max', 'siam-cosine-rmac']  # 'siam-cosine-rmac-marginal'
        run_ep_perf = []
    # ===========================
    #   SPIE-Siamese-L2-AltLoss
    # ===========================
    #elif experiment == 'SpieSiamL2AltLoss':
    #    runs = ['311', '312', '313b', '314b']
    #    run_net_types = ['siam'] * len(runs)
    #    run_metrics = ['cosine'] * len(runs)
    #    run_epochs = [list(range(1, 101, 1))] * len(runs)
    #    run_names = ['siam-l2-max', 'siam-l2-rmac', 'siam-l2-max-marginal', 'siam-l2-rmac-marginal']
    # ===========================
    #   SPIE-Siamese-Cosine-AltLoss
    # ===========================
    #elif experiment == 'SpieSiamCosineAltLoss':
    #    runs = ['321', '322', '323b', '324b']
    #    run_net_types = ['siam'] * len(runs)
    #    run_metrics = ['cosine'] * len(runs)
    #    run_epochs = [list(range(1, 101, 1))] * len(runs)
    #   run_names = ['siam-cosine-max', 'siam-cosine-rmac', 'siam-cosine-max-marginal', 'siam-cosine-rmac-marginal']
    # ===========================
    #   SPIE-DirectRating
    # ===========================
    elif experiment == 'SpieDirRating':
        runs = ['803', '801', '813', '811']  # '800' '802' '810' '812'
        run_net_types = ['dirR'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 101, 1))] * len(runs)
        run_names = ['dirR-max-clean', 'dirR-rmac-clean', 'dirR-max-primary', 'dirR-rmac-primary']  # 'dirR-conf-size' 'dirR-conf-rating-std' 'dirR-conf-size-primary' 'dirR-conf-rating-std-primary'
    # ===========================
    #   SPIE-SiamRating-L2
    # ===========================
    elif experiment == 'SpieSiamRatingL2':
        runs = ['404', '400', '414', '410']  # '402', '403'
        run_net_types = ['siamR'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 101, 1))] * len(runs)
        run_names = ['siamR-l2-max-clean', 'siamR-l2-rmac-clean', 'siamR-l2-max-primary', 'siamR-l2-rmac-primary']  # , 'siamR-l2-conf', 'siamR-cosine-conf'
    # ===========================
    #   SPIE-SiamRating-Cosine
    # ===========================
    elif experiment == 'SpieSiamRatingCosine':
        runs = ['405', '401', '415', '411']  # '412', '413'
        run_net_types = ['siamR'] * len(runs)
        run_metrics = ['cosine'] * len(runs)
        run_epochs = [list(range(1, 101, 1))] * len(runs)
        run_names = ['siamR-cosine-max-clean', 'siamR-cosine-rmac-clean', 'siamR-cosine-max-primary', 'siamR-cosine-rmac-primary']  # 'siamR-l2-conf', 'siamR-cosine-conf'
    # ===========================
    #   DirectDistanceMax
    # ===========================
    elif experiment == 'DirectDistanceMax':
        runs = ['820', '821' , '822', '823', '824', '825', '826']  #
        run_net_types = ['dirD'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 121, 1))] * len(runs)
        run_names = ['logcosh-loss', 'pearson-loss', 'kl-loss', 'poisson-loss', 'entropy-loss', 'ranked-pearson', 'kl-norm-loss']  #
    # ===========================
    #   DirectDistanceRmac
    # ===========================
    elif experiment == 'DirectDistanceRmac':
        runs = ['830', '831', '832', '833', '834', '835', '836']  #
        run_net_types = ['dirD'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 121, 1))] * len(runs)
        run_names = ['logcosh-loss', 'pearson-loss', 'kl-loss', 'poisson-loss', 'entropy-loss', 'ranked-pearson', 'kl-norm-loss']  #
    # ===========================
    #   DirectDistancePreDirR813-50
    # ===========================
    elif experiment == 'DirectDistancePre813-50':
        runs = ['841', '842', '846', '851', '852', '856', '842b']  #
        run_net_types = ['dirD'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 121, 1))] * len(runs)
        run_names = ['max-pearson', 'max-kl', 'max-norm-kl', 'rmac-pearson', 'rmac-kl', 'rmac-norm-kl', 'max-kl-lr-4']  #
    # ===========================
    #   DirDistPreFreezeLayers
    # ===========================
    elif experiment == 'DirDistPreFreezeLayers':
        runs = ['842b', '860', '861', '862', '863']  #
        run_net_types = ['dirD'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 151, 1))] * len(runs)
        run_names = ['no-freeze', '1-block', '2-blocks', '3-blocks', '4-blocks']  #
    # ===========================
    #   ConvLSTM
    # ===========================
    elif experiment == 'ConvLSTM':
        runs = ['0012']  #
        run_net_types = ['dirR'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 151, 1))] * len(runs)
        run_names = ['0012']  #
    # ===========================
    #   Debug
    # ===========================
    elif experiment == 'Debug':
        runs = ['312', '322', '314c', '324c']
        run_net_types = ['siam'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 101, 1))] * len(runs)
        run_names = ['l2-rmac', 'cosine-rmac', 'l2-rmac-marg', 'cosine-rmac-marg']
    else:
        print("No Such Experiment: " + experiment)
        assert(False)

    n = len(runs)
    assert len(run_net_types) == n
    assert len(run_metrics) == n
    assert len(run_epochs) == n
    assert len(run_names) == n

    return runs, run_net_types, run_metrics, run_epochs, run_names, run_ep_perf, run_ep_comb


if __name__ == "__main__":
    manager = CrossValidationManager()

    for idx in range(10):
        print('\npred_train: \t', manager.get_prediction_train(idx))
        print('pred_valid: \t', manager.get_prediction_validation(idx))
        print('pred_eval: \t', manager.get_prediction_eval(idx))
        print('ret_train: \t', manager.get_retrieval_train(idx))
        print('test: \t', manager.get_target(idx))

