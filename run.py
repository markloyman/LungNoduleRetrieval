import gc
import argparse
from Network.train import run as run_training
from Network.train3d import run as run_training_3d
import config
config.set_folders()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Lung Nodule Retrieval NN")
    parser.add_argument("-e", "--epochs", type=int, help="epochs", default=0)
    parser.add_argument("--confname", type=str, default='LEGACY', help="configuration name")
    parser.add_argument("-c", "--config", type=int, help="configuration", default=-1)
    parser.add_argument("-m", "--embed", action="store_true", default=False, help="generate embeddings")
    parser.add_argument("-p", "--predict", action="store_true", default=False, help="generate predictions")
    parser.add_argument("-s", "--seq", action="store_true", default=False, help="run 3d setup")
    parser.add_argument("--spatial", action="store_true", default=False, help="spatial embedding")
    parser.add_argument("--dataFromPredictions", action="store_true", default=False, help="load dataset from predications")
    args = parser.parse_args()

    epochs = args.epochs if (args.epochs != 0) else 81
    test = args.embed or args.predict

    config_list = []
    if args.config == -1:
        config_list = list(range(5))
    elif args.config == 10:
        config_list = range(args.config)
    else:
        config_list = [args.config]

    # DIR / SIAM / DIR_RATING / SIAM_RATING / TRIPLET
    net_type = 'DIR_RATING'
    embed_data_type = 'Test'

    for config in config_list:
        if args.seq:
            model = run_training_3d(net_type, epochs=epochs, config=config, skip_validation=False, no_training=test)
        else:
            model = run_training(net_type, epochs=epochs, config=config, config_name=args.confname,
                                 skip_validation=False, no_training=test, load_data_from_predications=args.dataFromPredictions)
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

    gc.collect()