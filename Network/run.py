import gc
#from keras import backend as K
try:
    from Network.train import run as run_training
    from Network.train3d import run as run_training_3d
    from Network.embed import Embeder
except:
    from train import run as run_training
    from train3d import run as run_training_3d
    from embed import Embeder

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Lung Nodule Retrieval NN")
    parser.add_argument("-e", "--epochs", type=int, help="epochs", default=0)
    parser.add_argument("-c", "--config", type=int, help="configuration", default=-1)
    parser.add_argument("-m", "--embed", action="store_true", default=False, help="generate embeddings")
    parser.add_argument("-p", "--predict", action="store_true", default=False, help="generate predictions")
    parser.add_argument("-s", "--seq", action="store_true", default=False, help="run 3d setup")
    parser.add_argument("--spatial", action="store_true", default=False, help="spatial embedding")
    args = parser.parse_args()

    epochs = args.epochs if (args.epochs != 0) else 81
    config_list = [args.config] if (args.config != -1) else list(range(5))
    test = args.embed or args.predict

    if len(config_list) > 1:
        print("Perform Full Cross-Validation Run")

    # DIR / SIAM / DIR_RATING / SIAM_RATING / TRIPLET
    net_type = 'DIR_RATING'
    embed_data_type = 'Test'

    for config in config_list:
        if args.seq:
            model = run_training_3d(net_type, epochs=epochs, config=config, skip_validation=False, no_training=test)
        else:
            model = run_training(net_type, epochs=epochs, config=config, skip_validation=True, no_training=test)
        if test:
            epoch0 = 1
            delta_epoch = 1
            if args.spatial:
                epochs_ = [30, 35, 40]
                model.embed_spatial(epochs=epochs_, data=embed_data_type)
            else:
                epochs_ = list(range(epoch0, model.last_epoch + 1, delta_epoch)) if delta_epoch > 0 else [epoch0]
                # epochs_ = [75, 80, 85]
                model.embed(epochs=epochs_, data=embed_data_type, use_core=True is args.embed, seq_model=args.seq)


    #K.clear_session()
    gc.collect()