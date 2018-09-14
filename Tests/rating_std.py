import numpy as np
import pickle

data = []
for c in range(5):
    filename = './Dataset/Dataset{}CV{}_{:.0f}-{}-{}.p'.format('Primary', c, 160, 0.5, 'Normal')
    data_group = pickle.load(open(filename, 'br'))
    print("Loaded {} entries from {}".format(len(data_group), filename))
    data += data_group

ratings = [d['rating'] for d in data]

filtered_ratings = [r for r in ratings if len(r) >= 4]
stdev = np.array([np.std(r, axis=0) for r in filtered_ratings])
stdev = np.mean(stdev, axis=0)

rating_property = ['Subtlety', 'Internalstructure', 'Calcification', 'Sphericity', 'Margin',
                       'Lobulation', 'Spiculation', 'Texture', 'Malignancy']

res = dict(zip(rating_property, stdev))
[print(x + ": " + str(round(y*100)/100)) for x, y in res.items()]