import os
data_folder = r'G:\Library\Datasets\DeepLesion\Images_png'


def move_files_with_filter(filter_filename, label):
    os.makedirs(os.path.join(data_folder, label), exist_ok=True)
    counter = 0
    with open(filter_filename, 'r') as fileFilter:
        for filename in tuple(fileFilter):
            filename = filename.strip()
            last_underscore = filename.rfind('_')
            src_filename = os.path.join(filename[:last_underscore], filename[last_underscore+1:])
            try:
                full_src_path = os.path.join(data_folder, src_filename)
                full_dst_path = os.path.join(data_folder, label, filename)
                os.rename(full_src_path, full_dst_path)
                counter += 1
            except:
                print("\t{} failed".format(filename))
    print("Moved {} entries to \\{}".format(counter, label))


# Train: taken from "official" validation-set
move_files_with_filter(filter_filename='DeepLesion/lung-training.filter', label='F')

# Valid: taken from "official" test-set (half)
move_files_with_filter(filter_filename='DeepLesion/lung-validation.filter', label='F')

# Test: taken from "official" test-set (half)
move_files_with_filter(filter_filename='DeepLesion/lung-test.filter', label='F')

