from init import *
try:
    from Network.train import run as run_training
    from Network.embed import Embeder
except:
    from train import run
    from Network.embed import Embeder

import argparse
parser = argparse.ArgumentParser(description="Train Lung Nodule Retrieval NN")
parser.add_argument("-e", "--epochs", type=int, help="epochs", default=0)
parser.add_argument("-c", "--config", type=int, help="configuration", default=-1)
args = parser.parse_args()

epochs = args.epochs if (args.epochs != 0) else 2
config_list = [args.config] if (args.config != -1) else [2] #list(range(5))

if len(config_list) > 1:
    print("Perform Full Cross-Validation Run")

# DIR / SIAM / DIR_RATING / SIAM_RATING / TRIPLET
net_type = 'DIR'

#model = run_training(net_type, epochs=0, config=0)
#print("Begin Embedding")
#model.embed(model.net_type, model.run, epoch=10)

for config in config_list:
    model = run_training(net_type, epochs=epochs, config=config, skip_validation=True)

    model.embed(epoch0=1, delta_epoch=1)

