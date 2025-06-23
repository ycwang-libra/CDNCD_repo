import os
from tqdm import tqdm
from image_corrupt_main import corrupt
import matplotlib.pyplot as plt
import numpy as np
import cv2

sever_select = 'server1' # your sever name
domains = ['Real_World'] # only corrupt real domain data
corruption_modes = ['gaussian_blur', 'jpeg_compression', 'impulse_noise']
corrupt_severity = 5 # 1 2 3 4 5
npy_reshape_size = 128
save_npy = True

if sever_select == 'server1':
    root = 'data_root_path/' # TODO fill your dataset path on server1
elif sever_select in ['server2','server3']:
    root = 'data_root_path/' # TODO fill your dataset path on server2 or server3
else:
    raise ValueError('Please set the correct server name')

data_path = root + '/OfficeHome'
train_img_file = os.path.join(data_path, 'image_list', domains[0] + '_train.txt')
test_img_file = os.path.join(data_path, 'image_list', domains[0] + '_test.txt')

for corrupt_mode in corruption_modes:
    back_words = '_corrupt_' + corrupt_mode + '_severity_' + str(corrupt_severity)
    # Create folders for different corrupt_mode and save different corrupt data respectively.
    new_path = os.path.join(data_path, domains[0] + back_words)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    train_file_name = open(train_img_file).readlines()
    test_file_name = open(test_img_file).readlines()
    train_list = [(val.split()[0], int(val.split()[1])) for val in train_file_name]
    test_list = [(val.split()[0], int(val.split()[1])) for val in test_file_name]
    corrupt_train_img_file = os.path.join(data_path, 'image_list', domains[0] + '_train' + back_words + '.txt')
    corrupt_test_img_file = os.path.join(data_path, 'image_list', domains[0] + '_test' + back_words + '.txt')
    ######################## train ####################
    print('train img process')
    if save_npy:
        tr_corrupt_npy = np.zeros([len(train_list), npy_reshape_size, npy_reshape_size, 3], dtype = 'uint8')
    with open(corrupt_train_img_file, 'w') as f:
        for tr_img_idx in tqdm(range(len(train_list))):
            tr_img_name = train_list[tr_img_idx][0]
            domain_name = tr_img_name.split('/')[0]
            class_name = tr_img_name.split('/')[1]
            img_name = tr_img_name.split('/')[2]
            # Change the image name and write it into a txt file.
            corrupted_img_name = img_name.split('.')[0] + back_words + '.jpg'
            f.write(domain_name + back_words + '/' + class_name + '/' + corrupted_img_name + ' ' + str(train_list[tr_img_idx][1]) + '\n')
            # Corrupt the image and save it.
            tr_img_path = os.path.join(data_path, tr_img_name)
            tr_img = plt.imread(tr_img_path)
            tr_img_corrupt = corrupt(tr_img, corrupt_severity, corrupt_mode)
            if os.path.exists(os.path.join(new_path, class_name)) == False:
                os.makedirs(os.path.join(new_path, class_name))
            corrupted_img_path = os.path.join(new_path, class_name, corrupted_img_name)
            plt.imsave(corrupted_img_path, tr_img_corrupt)
            # reshape and save as npy format
            if save_npy:
                tr_img_corrupt = cv2.resize(tr_img_corrupt, (npy_reshape_size, npy_reshape_size))
                tr_img_corrupt = np.reshape(tr_img_corrupt, (1, npy_reshape_size, npy_reshape_size, 3))
                tr_corrupt_npy[tr_img_idx,:,:,:] = tr_img_corrupt
    if save_npy:
        np.save(corrupt_train_img_file.split('.')[0] + '.npy', tr_corrupt_npy)
    ###################### test ########################
    print('test img process')
    if save_npy:
        ts_corrupt_npy = np.zeros([len(test_list), npy_reshape_size, npy_reshape_size, 3], dtype = 'uint8')
    with open(corrupt_test_img_file, 'w') as f:
        for ts_img_idx in tqdm(range(len(test_list))):
            ts_img_name = test_list[ts_img_idx][0]
            domain_name = ts_img_name.split('/')[0]
            class_name = ts_img_name.split('/')[1]
            img_name = ts_img_name.split('/')[2]
            # Change the image name and write it into a txt file.
            corrupted_img_name = img_name.split('.')[0] + back_words + '.jpg'
            f.write(domain_name + back_words + '/' + class_name + '/' + corrupted_img_name + ' ' + str(test_list[ts_img_idx][1]) + '\n')
            # Corrupt the image and save it.
            ts_img_path = os.path.join(data_path, ts_img_name)
            ts_img = plt.imread(ts_img_path)
            ts_img_corrupt = corrupt(ts_img, corrupt_severity, corrupt_mode)
            if os.path.exists(os.path.join(new_path, class_name)) == False:
                os.makedirs(os.path.join(new_path, class_name))
            corrupted_img_path = os.path.join(new_path, class_name, corrupted_img_name)
            plt.imsave(corrupted_img_path, ts_img_corrupt)
            # reshape and save as npy format
            if save_npy:
                ts_img_corrupt = cv2.resize(ts_img_corrupt, (npy_reshape_size, npy_reshape_size))
                ts_img_corrupt = np.reshape(ts_img_corrupt, (1, npy_reshape_size, npy_reshape_size, 3))
                ts_corrupt_npy[ts_img_idx,:,:,:] = ts_img_corrupt
    if save_npy:
        np.save(corrupt_test_img_file.split('.')[0] + '.npy', ts_corrupt_npy)
    print('The ' + corrupt_mode + ' corruption is done!')
