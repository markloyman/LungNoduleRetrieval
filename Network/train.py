from Network.data import load_nodule_dataset

from Network.DataGenSiam import DataGenerator
from Network.model import miniXception_loader
from Network.siameseArch import siamArch

## --------------------------------------- ##
## ------- General Setup ----------------- ##
## --------------------------------------- ##

#dataset = generate_nodule_dataset(0.2, 0.25)
dataset = load_nodule_dataset()
size = 128
input_shape = (size, size, 1)

# DIR / SIAM
choose_model = "SIAM"

## --------------------------------------- ##
## ------- Run Direct Architecture ------- ##
## --------------------------------------- ##

if choose_model is "DIR":
    run = 'dir000'

## --------------------------------------- ##
## ------- Run Siamese Architecture ------ ##
## --------------------------------------- ##

if choose_model is "SIAM":

    #run = 'siam000'   # chained, margin = 5, batch = 64,
    #run = 'siam001'    # base, margin = 5, batch = 64
    #run = 'siam002' # overlapped
    #run = 'siam000b'  # CHAINED, x2 sets
    #run = 'siam003'  # CHAINED, decrease learning rate to 1e-4
    #run = 'siam005'  # CHAINED, Sample-Weight
    run = 'siam00Y'  # junk


    # model

    model = siamArch(miniXception_loader, input_shape)
    model.model.summary()
    model.compile(learning_rate=1e-3)

    # methods = 'base', 'overlapped', 'chained'
    model.load_generator(DataGenerator(size, batch_sz=64, method='chained', use_class_weight=True))

    model.train(label=run, n_epoch=15, gen=True)