def load_experiments(experiment):
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
        runs            = ['900', '901', '999', '902']
        run_net_types   = ['dir']*len(runs)
        run_metrics     = ['l2']*len(runs)
        run_epochs      = [list(range(1, 37, 6))]*len(runs)
        run_names       = ['900-max', '901-avg', '999-rmac', '902-msrmac']
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
        runs = ['900', '220', '221', '222']
        run_net_types = ['dir'] * len(runs)
        run_metrics = ['l2'] * len(runs)
        run_epochs = [list(range(1, 46, 1))] * len(runs)
        run_names = ['900-base', '220-w256', '221-dataug', '222-data-aug']
    # ===========================
    else:
        print("No Such Experiment")
        assert(False)

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