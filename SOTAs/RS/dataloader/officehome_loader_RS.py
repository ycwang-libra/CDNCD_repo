# officehome data with cross domain or one domain

import torch.utils.data as data
import os
import numpy as np
from utils.util import TransformTwice, Solarize, Equalize
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as T
import random
import torch
from torch.utils.data.dataloader import default_collate
import torchnet as tnt

normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # for officehome dataset

def rotate_img(img, rot):
    if rot == 0: # 0 degrees rotation
        return img
    elif rot == 90: # 90 degrees rotation
        return np.flipud(np.transpose(img, (1,0,2)))
    elif rot == 180: # 90 degrees rotation
        return np.fliplr(np.flipud(img))
    elif rot == 270: # 270 degrees rotation / or -90
        return np.transpose(np.flipud(img), (1,0,2))
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

def rot_train_transform(resizecrop_size):
    return T.Compose([
        T.RandomResizedCrop(resizecrop_size, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip()
    ])

def rot_test_transform(resizecrop_size):
    return T.Compose([
        T.RandomResizedCrop(resizecrop_size, scale=(0.5, 1.0))
    ])

def train_transform(args): # use UNO's transform
    augmentation = [
        T.RandomResizedCrop(args.resizecrop_size, (0.5, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        normalize]
    transform = T.Compose(augmentation)
    return transform

def test_transform(args): # use UNO's transform
    augmentation = [
        T.Resize(args.resizecrop_size),
        T.CenterCrop(args.resizecrop_size),
        T.ToTensor(),
        normalize]
    transform = T.Compose(augmentation)
    return transform

class OFFICEHOMERotDataLoader(object):
    # After the data is imported, it is rotated at four angles and then concatenated. The concatenated tensor are output and the four angles of 0, 1, 2, and 3 be as labels.
    def __init__(self, args, dataset, batch_size=1, epoch_size=None, num_workers=0, shuffle=True):
        self.args = args
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # officehome dataset
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

    def get_iterator(self, epoch=0):
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)
        # if in unsupervised mode define a loader function that given the
        # index of an image it returns the 4 rotated copies of the image
        # plus the label of the rotation, i.e., 0 for 0 degrees rotation,
        # 1 for 90 degrees, 2 for 180 degrees, and 3 for 270 degrees.
        def _load_function(idx):
            idx = idx % len(self.dataset)
            img, label, _ = self.dataset[idx]
            rotated_imgs = [
                self.transform(img), # ToTensor Normalize
                self.transform(rotate_img(img,  90).copy()),
                self.transform(rotate_img(img, 180).copy()),
                self.transform(rotate_img(img, 270).copy()),                
            ]
            rotation_labels = torch.LongTensor([0, 1, 2, 3])
            if label < self.args.num_labeled_classes:
                domain_flag = torch.LongTensor([0, 0, 0, 0]) # source domain
            else:
                domain_flag = torch.LongTensor([1, 1, 1, 1]) # target domain
            return torch.stack(rotated_imgs, dim=0), rotation_labels, domain_flag # torch.Size([4, 3, 32, 32]) tensor([0,1,2,3])
        
        def _collate_fun(batch):
            batch = default_collate(batch)
            # assert(len(batch)==2)
            batch_size, rotations, channels, height, width = batch[0].size()
            batch[0] = batch[0].view([batch_size*rotations, channels, height, width]) # batch[0]:torch.Tensor[128,4,3,32,32]--> torch.Tensor[512,3,32,32]
            batch[1] = batch[1].view([batch_size*rotations]) # batch[1]:torch.Tensor[128,4]--> torch.Tensor[512]
            batch[2] = batch[2].view([batch_size*rotations]) # for domain_flag
            return batch

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
            load=_load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
            collate_fn=_collate_fun, num_workers=self.num_workers,
            shuffle=self.shuffle)
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size / self.batch_size

class OfficeHome(data.Dataset):
    # specific domain and specific class of data, label domain_label (0 for source domain and 1 for target domain)
    def __init__(self, args, lab_flag, domain_name, split='train + test', transform=None, label_range = range(5)):
        self.args = args
        self.data_root = args.dataset_root
        self.transform = transform
        self.loader = self._rgb_loader
        self.lab_flag = lab_flag
        self.domain_name = domain_name
        self.domain_flag = 0 if domain_name == self.args.source_domain else 1 # 0 for source domain and 1 for target domain
        if lab_flag == 'lab':
            txt_path = os.path.join(self.data_root, 'image_list', domain_name + '_' + split + '.txt')
        elif lab_flag == 'unlab':
            if args.unlab_corrupt_severity == 0: # no corrupt
                txt_path = os.path.join(self.data_root, 'image_list', domain_name + '_' + split + '.txt')
            else: # corrupt
                txt_path = os.path.join(self.data_root, 'image_list', domain_name + '_' + split + '_corrupt_' + args.corrupt_mode + '_severity_' + str(args.unlab_corrupt_severity) + '.txt')
        else:
            raise ValueError('lab_flag should be lab or unlab')
        images_pathwithlabel = self._make_dataset(txt_path)
        self.select_samples = self._sample_data(images_pathwithlabel, label_range)

    def _rgb_loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def _make_dataset(self, image_list_path):
        image_list = open(image_list_path).readlines()
        images_path_label = [(val.split()[0], int(val.split()[1])) for val in image_list]
        return images_path_label
    
    def _sample_data(self, pathwithlabel, label_range):
        # Input: Two list paths with labels: All the data in the dataset, each element including the image address and label; label_range: Specifies the sampled labels.
        # Output: Image addresses, labels, and domain flags of the specified labels.
        select_samples = []
        num_data = len(pathwithlabel)
        # If we do not use subclass data, but instead use (all data or the first x classes), then we use the original labels. There is no need to rewrite the labels, thus reducing the computational load.
        for img_idx in range(num_data):
            if pathwithlabel[img_idx][1] in label_range:
                select_samples.append([pathwithlabel[img_idx][0], pathwithlabel[img_idx][1], self.domain_flag]) # new define label
        return select_samples

    def __getitem__(self, index):
        img_name, label, domain_flag = self.select_samples[index]
        img_path = os.path.join(self.args.dataset_root, img_name) # image absolute path
        img = self.loader(img_path)

        if self.transform is not None:
            img = self.transform(img)
            
        return img, label, domain_flag
    
    def __len__(self):
        return len(self.select_samples)

def OFFICEHOMELoaderMix(args, split='train', shuffle=True, labeled_list=range(5), unlabeled_list=range(5, 10)):
    # for auto_novel mix training add domain flag and load seperately
    transform = TransformTwice(train_transform(args))
    # labeled data
    dataset_labeled = OfficeHome(args, lab_flag = 'lab', domain_name = args.source_domain, split=split,transform=transform, label_range=labeled_list)

    # unlabeled data
    dataset_unlabeled = OfficeHome(args, lab_flag = 'unlab', domain_name = args.target_domain, split=split, transform=transform, label_range=unlabeled_list)

    label_loader = DataLoader(dataset = dataset_labeled, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)
    unlabel_loader = DataLoader(dataset = dataset_unlabeled, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)
    return label_loader, unlabel_loader

def OfficeHomeLoader(args, lab_flag, domain, split='train', shuffle = None, label_range=range(5)):
    if split == 'train':
        transform = train_transform(args)
    elif split == 'test':
        transform = test_transform(args)
    else:
        raise ValueError('split should be train or test')
    dataset = OfficeHome(args, lab_flag, domain, split=split, transform = transform, label_range=label_range)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    return data_loader

def get_rot_officehome_loader(args):
    # for selfsupervised learning (labeled and unlabeled samples) seperately load two kinds of data
    # no need test data
    rot_tr_transform = rot_train_transform(resizecrop_size=args.resizecrop_size)
    rot_ts_transform = rot_test_transform(resizecrop_size=args.resizecrop_size)
    # labeled data

    train_lab_dataset = OfficeHome(args, lab_flag = 'lab', domain_name = args.source_domain, split='train', transform = rot_tr_transform, label_range = range(args.num_labeled_classes))
    val_lab_dataset = OfficeHome(args, lab_flag = 'lab', domain_name = args.source_domain,split='test', transform = rot_ts_transform, label_range = range(args.num_labeled_classes))

    # unlabeled data
    train_unlab_dataset = OfficeHome(args, lab_flag = 'unlab', domain_name = args.source_domain, split='train', transform = rot_tr_transform, label_range = range(args.num_labeled_classes, args.num_classes))
    val_unlab_dataset = OfficeHome(args, lab_flag = 'unlab', domain_name = args.source_domain, split='test', transform = rot_ts_transform, label_range = range(args.num_labeled_classes, args.num_classes))

    data_loader = dict()
    data_loader['train_lab_loader'] = OFFICEHOMERotDataLoader(args = args, dataset = train_lab_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    data_loader['train_unlab_loader'] = OFFICEHOMERotDataLoader(args = args, dataset = train_unlab_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    data_loader['test_lab_loader'] = OFFICEHOMERotDataLoader(args = args, dataset = val_lab_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    data_loader['test_unlab_loader'] = OFFICEHOMERotDataLoader(args = args, dataset = val_unlab_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False) 
    return data_loader

def get_sup_officehome_loader(args):
    # for supervised learning(labeled samples) unlabeled target sample are loaded for mixstyle
    data_loader = dict()
    # labeled data
    data_loader['train_lab_loader'] = OfficeHomeLoader(args, lab_flag = 'lab', domain = args.source_domain, split='train', shuffle=True, label_range = range(args.num_labeled_classes))
    data_loader['test_lab_loader'] = OfficeHomeLoader(args, lab_flag = 'lab', domain = args.source_domain, split='test', shuffle=False, label_range = range(args.num_labeled_classes))

    # unlabeled data
    data_loader['train_unlab_loader'] = OfficeHomeLoader(args, lab_flag = 'unlab', domain = args.target_domain, split='train', shuffle=True, label_range = range(args.num_labeled_classes, args.num_classes))
    data_loader['test_unlab_loader'] = OfficeHomeLoader(args, lab_flag = 'unlab', domain = args.target_domain, split='test', shuffle=False, label_range = range(args.num_labeled_classes, args.num_classes))

    return data_loader

def get_auto_officehome_loader(args):
    # for auto_novel training
    data_loader = dict()
    data_loader['label_loader'], data_loader['unlabel_loader'] = OFFICEHOMELoaderMix(args, split='train', shuffle=True, labeled_list=range(args.num_labeled_classes), unlabeled_list=range(args.num_labeled_classes, args.num_classes))
    data_loader['test_lab_loader'] = OfficeHomeLoader(args, lab_flag = 'lab', domain = args.source_domain, split='test', shuffle=False, label_range = range(args.num_labeled_classes))
    # unlabeled data
    data_loader['train_unlab_loader'] = OfficeHomeLoader(args, lab_flag = 'unlab', domain = args.target_domain, split='train', shuffle=True, label_range = range(args.num_labeled_classes, args.num_classes))
    data_loader['test_unlab_loader'] = OfficeHomeLoader(args, lab_flag = 'unlab', domain = args.target_domain, split='test', shuffle=False, label_range = range(args.num_labeled_classes, args.num_classes))
    return data_loader