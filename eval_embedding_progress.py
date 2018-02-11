from init import *
#from Network import embed
#from Analysis import RatingCorrelator
from evalulate import  dir_rating_delta_rmse, l2

## ===================== ##
## ======= Setup ======= ##
## ===================== ##

network = 'dirR'

wRuns = ['011X'] # '011XXX'
wEpchs= [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

pooling = 'rmac'

# 0     Test
# 1     Validation
# 2     Training
DataSubSet = 2

if   DataSubSet == 0:
    post = "Test"
elif DataSubSet == 1:
    post = "Valid"
elif DataSubSet == 2:
    post = "Train"
else:
    assert False
print("{} Set Analysis".format(post))
print('='*15)

## ========================= ##
## ======= Evaluate  ======= ##
## ========================= ##

start = timer()
try:
    e0, e1 = 35, 65
    delta = dir_rating_delta_rmse(wRuns[0], post, e0, e1)
    image_id = np.argmax(delta)
    print("selected #{} with delta-rmse {}".format(image_id, delta[image_id]))

    embd = [FileManager.Embed(network).load(run=wRuns[0], epoch=e, dset=post)[1][image_id] for e in wEpchs]
    diff = [l2(embd[i], embd[i+1]) for i in range(len(wEpchs)-1)]

    plt.figure()
    plt.title('nod#{}'.format(image_id))
    plt.plot(wEpchs[1:], diff,'-*')
    plt.xlabel('epochs')
    plt.ylabel('delta-rmse')

    #Embd = embed.Embeder(network, pooling=pooling)
    #embedding = Embd.generate_timeline_embedding(runs=wRuns, post=post, data_subset_id=DataSubSet, epochs=wEpchs)
    #
    #W = FileManager.Embed(network)(wRuns[0], e0, post)
    # correlation plot
    #Reg = RatingCorrelator(W)
    #Reg.evaluate_embed_distance_matrix(method='l2')
    #Reg.evaluate_rating_space(norm='Scale')
    #Reg.evaluate_rating_distance_matrix(method='l2')
    #p, s, k = Reg.correlate_retrieval('embed', 'rating')
    #delta = embedding[:, :, e1] - embedding[:, :, e0]

finally:
    plt.show()

    total_time = (timer() - start) / 60 / 60
    print("Total runtime is {:.1f} hours".format(total_time))