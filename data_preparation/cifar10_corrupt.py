import numpy as np
from image_corrupt_main import corrupt
import os
from tqdm import tqdm
import time
import datetime
import glob
import pickle

# use npy format cifar for saving time
origin_cifar10_path = 'data_root_path/cifar-10-batches-py/'
new_cifar10_path = 'data_root_path/cifar-10-numpy/' # TODO fill your dataset path
if (os.path.exists(new_cifar10_path) == False):
    os.makedirs(new_cifar10_path)

train_data_list = glob.glob(origin_cifar10_path + "data_batch_*") # 3,4,5,1,2
train_data_list = sorted(train_data_list, key=lambda name: int(name[-1]))
test_data_list = os.path.join(origin_cifar10_path, 'test_batch')

for data in train_data_list:
    data = pickle.load(open(data, 'rb'), encoding='bytes')
    labels, data, filenames = data[b'labels'], data[b'data'], data[b'filenames']
    labels, data = map(np.array, [labels, data])
    try:
        train_Data = np.r_[train_Data, data]
        train_Labels = np.r_[train_Labels, labels]
    except:
        train_Data = data
        train_Labels = labels

test_data = pickle.load(open(test_data_list, 'rb'), encoding='bytes')
test_labels, test_data, test_filenames = data[b'labels'], data[b'data'], data[b'filenames']
test_labels, test_data = map(np.array, [test_labels, test_data])

# 50000 * 3072 numpy.array data to 50000 * 3 * 32 * 32 numpy.array
train_Data = train_Data.reshape(50000, 3, 32, 32)
train_Data = train_Data.transpose(0,2,3,1) # 3 * 32 * 32 --> 32 * 32 * 3
# 10000 * 3072 numpy.array data to 10000 * 3 * 32 * 32 numpy.array
test_data = test_data.reshape(10000, 3, 32, 32)
test_data = test_data.transpose(0,2,3,1) # 3 * 32 * 32 --> 32 * 32 * 3
np.save(new_cifar10_path + "data.npy", train_Data)
np.save(new_cifar10_path + "label.npy", train_Labels)
np.save(new_cifar10_path + "test_data.npy", test_data)
np.save(new_cifar10_path + "test_label.npy", test_labels)

################  corrupt  ##############
train_data = np.load(new_cifar10_path + 'data.npy') # (50000,32,32,3)
test_data = np.load(new_cifar10_path + 'test_data.npy') # (10000,32,32,3)

aim_root_path = 'aim_dataset_path/CIFAR10/CIFAR-10_corruption/'# TODO aim_path
if (os.path.exists(aim_root_path + '/train') == False):
    os.makedirs(aim_root_path + '/train')
if (os.path.exists(aim_root_path + '/test') == False):
    os.makedirs(aim_root_path + '/test')

# all corruption_modes ['spatter', 'brightness', 'jpeg_compression', 'elastic_transform', 'motion_blur', 'zoom_blur', 'impulse_noise', 'speckle_noise', 'saturate', 'gaussian_noise', 'shot_noise', 'glass_blur', 'fog', 'gaussian_blur', 'pixelate', 'contrast', 'defocus_blur', 'snow', 'frost']
corruption_modes = ['gaussian_blur', 'jpeg_compression', 'impulse_noise']

num_corruption = len(corruption_modes)

num_train_data = train_data.shape[0]
num_test_data = test_data.shape[0]
# processing the data
start_time = time.time()
for s in range(1,6): # 5 corrupt severities
    print('The corruption severity is: ' + str(s) + '\n')
    for j in range(num_corruption):
        print('The corruption mode is: ' + corruption_modes[j] + '\n')
        train_corrupt_data = np.zeros(train_data.shape, dtype = 'uint8')
        test_corrupt_data = np.zeros(test_data.shape, dtype = 'uint8')
        for i in tqdm(range(num_train_data)): # 50000
            image = train_data[i,:,:,:]
            corrupt_name = corruption_modes[j]
            corrupt_img = corrupt(image, severity = s, corruption_name = corrupt_name)
            corrupt_root_path = aim_root_path + '/train/Severity' + str(s) + '/'
            if (os.path.exists(corrupt_root_path) == False):
                os.makedirs(corrupt_root_path)
            corrupt_data_path = corrupt_root_path + corrupt_name + '.npy'
            train_corrupt_data[i,:,:,:] = corrupt_img
        np.save(corrupt_data_path, train_corrupt_data)
        for i in tqdm(range(num_test_data)): # 10000
            image = test_data[i,:,:,:]
            corrupt_name = corruption_modes[j]
            corrupt_img = corrupt(image, severity = s, corruption_name = corrupt_name)
            corrupt_root_path = aim_root_path + '/test/Severity' + str(s) + '/'
            if (os.path.exists(corrupt_root_path) == False):
                os.makedirs(corrupt_root_path)
            corrupt_data_path = corrupt_root_path + corrupt_name + '.npy'
            test_corrupt_data[i,:,:,:] = corrupt_img
        np.save(corrupt_data_path, test_corrupt_data)
end_time = time.time()
used_time = end_time - start_time
et = str(datetime.timedelta(seconds = used_time))[:-7]
print('The used time is: ' + et)