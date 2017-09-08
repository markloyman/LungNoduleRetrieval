import numpy as np
import pickle
from timeit import default_timer as timer

from data import generate_nodule_dataset, load_nodule_dataset, prepare_data, prepare_data_siamese, prepare_data_siamese_overlap
from DataGenSiam import  DataGenerator
import dataUtils
from model import miniXception_loader
from directArch import directArch
from siameseArch import  siamArch
from analysis import history_summarize



## --------------------------------------- ##
## ------- General Setup ----------------- ##
## --------------------------------------- ##

#dataset = generate_nodule_dataset(0.2, 0.25)
dataset = load_nodule_dataset()
size = 128
input_shape = (size, size,1)

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
    run = 'siam004XX'  # CHAINED, lr=1e-3, 1e-4, 1e-5


    # model

    model = siamArch(miniXception_loader, input_shape)
    model.model.summary()
    model.compile(learning_rate=1e-3)

    # methods = 'base', 'overlapped', 'chained'
    model.load_generator(DataGenerator(size, batch_sz=64, method='chained'))

    model.train(label=run, n_epoch=3, gen=True)
    #model.train(label=run, epoch=11, n_epoch=15, gen=True, new_lr=1e-8)
    #model.train(label=run, epoch=26, n_epoch=15, gen=True, new_lr=1e-5)