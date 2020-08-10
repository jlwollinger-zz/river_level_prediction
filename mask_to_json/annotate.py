import os

from annotation import dataset
from annotation import mask2json


if __name__ == '__main__':

    main_dir = "./original_data"
    images_dir = os.path.join(main_dir, "images")
    masks_dir = os.path.join(main_dir, "silhouette")

    val_ratio = 0.2
    test_ratio = 0.1

    n_files, n_train, n_val, n_test = \
                        dataset.split(main_dir, images_dir, masks_dir, val_ratio, test_ratio)

    print('Total :      ', n_files)
    print('Training :   ', n_train)
    print('Validation : ', n_val)
    print('Testing :    ', n_test)


    data_dir = './dataset'
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test") 

    mask2json.create_annotations(train_dir)
    mask2json.create_annotations(val_dir)
    mask2json.create_annotations(test_dir)
    