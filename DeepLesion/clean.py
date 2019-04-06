import os
data_folder = 'DeepLesion/data'


def move_files_with_filter(filter_filename, label):
    os.makedirs(data_folder + "/" + label, exist_ok=True)
    counter = 0
    with open(filter_filename, 'r') as fileFilter:
        for filename in tuple(fileFilter):
            filename = filename.strip()
            try:
                os.rename(data_folder + "/" + filename, data_folder + "/{}/".format(label) + filename)
                counter += 1
            except:
                print("\t{} failed".format(filename))
    print("Moved {} entries to \\{}".format(counter, label))


# Train: taken from "official" validation-set
move_files_with_filter(filter_filename='DeepLesion/lung-training.filter', label='Train')

# Valid: taken from "official" test-set (half)
move_files_with_filter(filter_filename='DeepLesion/lung-validation.filter', label='Valid')

# Test: taken from "official" test-set (half)
move_files_with_filter(filter_filename='DeepLesion/lung-test.filter', label='Test')

