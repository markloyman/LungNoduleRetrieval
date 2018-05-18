def load_experiments(experiment):
    runs, run_net_types, run_metrics, run_epochs, run_names = [], [], [], [], []
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
        runs            = ['011XXX', '016XXXX', '027', '023X']
        run_net_types   = ['trip']*len(runs)
        run_metrics     = ['l2']*len(runs)
        run_epochs      = [ [5, 15, 25, 35, 45, 55],
                            [20, 40, 100, 150, 180],
                            [5, 15, 25, 35, 40, 45, 50, 55, 60],
                            [5, 15, 20, 25, 30, 35]
                        ]
        run_names       = ['malig-obj', 'rating-obj', 'rating-obj', 'trip-finetuned']
    # ===========================
    #   Pooling
    # ===========================
    elif experiment == 'Pooling':
        runs            = ['235', '252', '251']  # '250', '900', '901', '999', '902',
        run_net_types   = ['dir']*len(runs)
        run_metrics     = ['l2']*len(runs)
        run_epochs      = [list(range(1, 61, 1))]*len(runs)
        run_names       = ['235-max', '252-msrmac', '251-avg']  # '250-max2', '900-max', '901-avg', '999-rmac', '902-msrmac',
    # ===========================
    #   PoolingAug
    # ===========================
    elif experiment == 'PoolingAug':
        runs = ['243b', '253', '254']
        run_net_types = ['dir'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 71, 1))] * len(runs)
        run_names = ['243b-max', '253-avg', '254-msrmac']
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
        runs = ['202b', '204', '207', '205', '206', '208']
        run_net_types = ['siam'] * 6
        run_metrics = ['l2'] * 3 + ['l1'] * 3
        run_epochs = [list(range(1, 51, 1))] * len(runs)
        run_names = ['202b-siam-l2-max', '204-siam-l2-msrmac', '207-l2-avg', '205-siam-l1-max', '206-siam-l1-msrmac', '208-l1-avg']
    # ===========================
    #   Dir-Rating
    # ===========================
    elif experiment == 'DirRating':
        runs = ['200b', '201', '202', '202c']
        run_net_types = ['dirR']*len(runs)
        run_metrics = ['l2']*len(runs)
        run_epochs = [list(range(1, 101, 1))] * len(runs)
        run_names = ['200b-max', '201-msrmac', '202-avg', '202c-avg']
    # ===========================
    #   Debug
    # ===========================
    elif experiment == 'Debug':
        runs = ['235', '252', '251']
        run_net_types = ['dir'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [[40]]*len(runs)
        run_names = ['235-max', '252-msrmac', '251-avg']
    else:
        print("No Such Experiment")
        assert(False)

    n = len(runs)
    assert len(run_net_types) == n
    assert len(run_metrics) == n
    assert len(run_epochs) == n
    assert len(run_names) == n

    return runs, run_net_types, run_metrics, run_epochs, run_names


'''
    #runs = ['100', '016XXXX', '021', '023X']  #['064X', '078X', '026'] #['064X', '071' (is actually 071X), '078X', '081', '082']
    #run_net_types = ['siam', 'trip','trip', 'trip']  #, 'dir']
    runs            = ['021', '022XX', '023X', '025']
    run_names       = ['max-pool', 'rmac', 'categ', 'confidence+cat' ]
    run_net_types   = ['trip']*len(runs)
    run_metrics     = ['l2']*len(runs)
    #rating_normalizaion = 'Scale' # 'None', 'Normal', 'Scale'

    run_epochs = [ [5, 15, 25, 35],
                   [5, 15, 25, 35, 45, 55],
                   [5, 15, 25, 35],
                   [5, 15, 25, 35, 45, 55]
               ]
'''