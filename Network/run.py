import gc
#from keras import backend as K
try:
    from Network.train import run as run_training
    from Network.embed import Embeder
except:
    from train import run as run_training
    from embed import Embeder

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Lung Nodule Retrieval NN")
    parser.add_argument("-e", "--epochs", type=int, help="epochs", default=0)
    parser.add_argument("-c", "--config", type=int, help="configuration", default=-1)
    parser.add_argument("-t", "--test", action="store_true", default=False, help="generate embeddings")
    args = parser.parse_args()

    epochs = args.epochs if (args.epochs != 0) else 81
    config_list = [args.config] if (args.config != -1) else list(range(5))
    test = args.test

    if len(config_list) > 1:
        print("Perform Full Cross-Validation Run")

    # DIR / SIAM / DIR_RATING / SIAM_RATING / TRIPLET
    net_type = 'DIR_RATING'

    for config in config_list:
        model = run_training(net_type, epochs=epochs, config=config, skip_validation=True, no_training=test)
        if test:
            epoch0 = 1
            delta_epoch = 1
            epochs_ = list(range(epoch0, model.last_epoch + 1, delta_epoch)) if delta_epoch > 0 else [epoch0]
            #epochs_ = [30, 70, 100]
            model.embed(epochs=epochs_, data='Test', use_core=False)

    #K.clear_session()
    gc.collect()