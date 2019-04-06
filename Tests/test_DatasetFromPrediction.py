from Network import FileManager
from experiments import CrossValidationManager
import numpy as np

run = '888'

DataGroups = [FileManager.Dataset('Primary', i, './Dataset').load(size=160, res=0.5) for i in range(5)]
expected_datast_size = [len(d) for d in DataGroups]
label_stats = [np.bincount([element['label'] for element in DataGroups[i]]) for i in range(5)]
[print('group id {} => total:{}, benign:{}, malig:{}, unknown:{}'.
       format(i, expected_datast_size[i], label_stats[i][0], label_stats[i][1], label_stats[i][2])) for i in range(5)]


cv = CrossValidationManager('RET')

for i in range(10):  # conf in conf_names:
    conf = cv.get_run_id(i)
    #dataset_size = len(FileManager.DatasetFromPredication().load(run='{}c{}'.format(run, conf), goal='Test', epoch=70))
    dataset_size = len(FileManager.Embed(pre='dirRD').load(run='{}c{}'.format(run, conf), dset='Valid'))
    group_id = cv.get_test(i)
    print('#{} ({})- expected: {}, actual: {} (group id = {})'.format(i, conf, expected_datast_size[group_id[0]], dataset_size, group_id))