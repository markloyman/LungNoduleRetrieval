from init import *
from Network import embed

## ===================== ##
## ======= Setup ======= ##
## ===================== ##

network = 'trip'

wRuns = ['027']
wEpchs= [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

pooling = 'rmac'

# 0     Test
# 1     Validation
# 2     Training
DataSubSet = 0

def data_label(idx):
    if   idx == 0:
        return "Test"
    elif idx == 1:
        return "Valid"
    elif idx == 2:
        return "Train"
    else:
        assert False

## ========================= ##
## ======= Evaluate  ======= ##
## ========================= ##

start = timer()
try:

    Embd = embed.Embeder(network, pooling=pooling, categorize=True)

    DataSubSet = 0
    Embd.run(runs=wRuns, post=data_label(DataSubSet), data_subset_id=DataSubSet, epochs=wEpchs)

    DataSubSet = 1
    Embd.run(runs=wRuns, post=data_label(DataSubSet), data_subset_id=DataSubSet, epochs=wEpchs)

    DataSubSet = 2
    Embd.run(runs=wRuns, post=data_label(DataSubSet), data_subset_id=DataSubSet, epochs=wEpchs)

finally:
    plt.show()

    total_time = (timer() - start) / 60 / 60
    print("Total runtime is {:.1f} hours".format(total_time))


'''
            if do_eval:
                print(pred.shape)
                calc_embedding_statistics(pred, title=filename)

                plt.figure()
                #plt.subplot(211)
                plt.plot(np.transpose( np.squeeze(pred[np.argwhere(np.squeeze(labels==0)),
                                       np.squeeze(np.argwhere(np.std(pred, axis=0) > 0.0))])), 'blue', alpha=0.3)
                #plt.subplot(212)
                plt.plot(np.transpose( np.squeeze(pred[np.argwhere(np.squeeze(labels == 1)),
                                       np.squeeze(np.argwhere(np.std(pred, axis=0) > 0.0))])), 'red', alpha=0.2)
'''