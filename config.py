import os
import errno


if 'LOCAL' in os.environ.keys():
    local = True
    dataset_dir = './Dataset'
    input_dir = './output'
    output_dir = './output'
    pred_as_dataset_dir = './output'
else:
    local = False
    dataset_dir = '/Dataset'
    input_dir = '/input'
    output_dir = '/output'
    pred_as_dataset_dir = '/Dataset'

def set_folders():
    print("Setting up output folders")
    try:
        os.makedirs('/output/Weights/')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    try:
        os.makedirs('/output/logs/')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    try:
        os.makedirs('/output/history/')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    try:
        os.makedirs('/output/embed/')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # [print(f) for f in os.listdir()]


# set_local:
#   os.environ["LOCAL"] = "True"