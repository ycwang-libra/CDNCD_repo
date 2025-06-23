import torch.utils.data as data
import os
import numpy as np
import sys
import pickle
from utils.util import check_integrity, download_url, TransformTwice, Solarize, Equalize
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as T
import random
import torch
from torch.utils.data.dataloader import default_collate
import torchnet as tnt

cifar10_normalize = T.Normalize(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
cifar100_normalize = T.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

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
        T.RandomChoice([T.RandomCrop(args.resizecrop_size, padding=4),
                        T.RandomResizedCrop(args.resizecrop_size, (0.5, 1.0))]),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
        Solarize(p=0.1),
        Equalize(p=0.1),
        T.ToTensor(),
        cifar10_normalize if args.dataset_series == 'CIFAR10' else cifar100_normalize]
    transform = T.Compose(augmentation)
    return transform

def test_transform(args): # use UNO's transform
    augmentation = [
        T.CenterCrop(args.resizecrop_size),
        T.ToTensor(),
        cifar10_normalize if args.dataset_series == 'CIFAR10' else cifar100_normalize]
    transform = T.Compose(augmentation)
    return transform

class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, 
                args,
                split='train+test',
                transform=None, 
                target_transform=None,
                download=False, 
                label_range = range(5)):
        self.root = args.dataset_root
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        downloaded_list = []
        if split=='train':
            downloaded_list = self.train_list
        elif split=='test':
            downloaded_list = self.test_list
        elif split=='train+test':
            downloaded_list.extend(self.train_list)
            downloaded_list.extend(self.test_list)

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)  #  reshape (50000, 3, 32, 32) <class 'numpy.ndarray'>
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC  (50000, 32, 32, 3)
        self._load_meta()

        ind = [i for i in range(len(self.targets)) if self.targets[i] in label_range] # 2500 idx，first 5 classes idx recording

        self.data = self.data[ind]  # labeled first 5 classes data (25000, 32, 32, 3)
        self.targets = np.array(self.targets) # list --> numpy.ndarray (50000,)
        self.targets = self.targets[ind].tolist() # list 25000 first 5 classes label

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1') 
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index] # numpy.ndarray 32, 32, 3  list 1

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        domain_flag = 0 # always 0 for CIFAR10(only one domain)

        return img, target, domain_flag

    def __len__(self):
        return len(self.data) # 25000

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class CIFAR10C(data.Dataset):
    def __init__(self, args, split='train+test', transform=None, label_range = range(5)):
        self.args = args
        self.transform = transform
        self.cifar10c_data = np.load(self.args.dataset_root + '/CIFAR-10_corruption/' + split + '/Severity' + str(self.args.unlab_corrupt_severity) + '/' + self.args.corrupt_mode + '.npy')
        self.cifar10c_label = np.load(self.args.dataset_root + '/CIFAR-10_corruption/' + split + '_label.npy')

        ind = [i for i in range(len(self.cifar10c_label)) if self.cifar10c_label[i] in label_range]

        self.cifar10c_data = self.cifar10c_data[ind]  # labeled classes data (25000, 32, 32, 3)
        self.cifar10c_label = self.cifar10c_label[ind].tolist() # list 25000

    def __getitem__(self, index):

        img = self.cifar10c_data[index, :, :, :]  #  (50000, 32, 32, 3) --> (32, 32, 3)
        target = self.cifar10c_label[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)  # data turn to tensor np.array --> torch.Tensors

        domain_flag = 1 # always 1 for CIFAR10c(cifar10 only for target domain)
        return img, target, domain_flag

    def __len__(self):
        return len(self.cifar10c_data)

class CIFARRotDataLoader(object):
    # After the data is imported, it is rotated at four angles and then concatenated. The concatenated tensor are output and the four angles of 0, 1, 2, and 3 be as labels.
    def __init__(self, args, dataset, batch_size=1, epoch_size=None, num_workers=0, shuffle=True):
        self.args = args
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.num_workers = num_workers
        if args.dataset_series == 'CIFAR10':
            mean = [0.491, 0.482, 0.447]
            std = [0.202, 0.199, 0.201]
        elif args.dataset_series == 'CIFAR100':
            mean = [0.507, 0.487, 0.441]
            std = [0.267, 0.256, 0.276]
        else:
            raise NotImplementedError
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
            batch[0] = batch[0].view([batch_size*rotations, channels, height, width])
            batch[1] = batch[1].view([batch_size*rotations])
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

def CIFARLoader(args, split='train', shuffle = None, label_range=range(5)):
    if split == 'train':
        transform = train_transform(args)
    elif split == 'test':
        transform = test_transform(args)
    else:
        raise ValueError('split should be train or test')

    if args.dataset_series == 'CIFAR10':
        dataset = CIFAR10(args, split=split, transform = transform, label_range=label_range)
    else:
        raise NotImplementedError

    drop_last = True
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, drop_last=drop_last, pin_memory=True)
    return data_loader

def CIFAR10CLoader(args, split='train', shuffle = None, label_range=range(5)):
    if split == 'train':
        transform = train_transform(args)
    elif split == 'test':
        transform = test_transform(args)
    else:
        raise ValueError('split should be train or test')
            
    dataset = CIFAR10C(args, split=split, transform = transform, label_range=label_range)
    
    drop_last = True
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, drop_last=drop_last, pin_memory=True)
    return data_loader

def CIFARLoaderMix(args, split='train', shuffle=True, labeled_list=range(5), unlabeled_list=range(5, 10)):
    # for auto_novel mix training add domain flag and load seperately
    transform = TransformTwice(train_transform(args))
    if args.dataset_series == 'CIFAR10':
        # labeled data
        if args.dataset_subclass in ['CIFAR10CMix','new_select']:
            dataset_labeled = CIFAR10(args, split=split, transform=transform, label_range=labeled_list)
        elif args.dataset_subclass in ['CIFAR10CAll']:
            dataset_labeled = CIFAR10C(args, split=split, transform=transform, label_range=labeled_list)
        else:
            raise ValueError('dataset_subclass should be CIFAR10CMix or CIFAR10CAll')
        # unlabeled data
        if args.dataset_subclass in ['CIFAR10CMix','CIFAR10CAll']:
            dataset_unlabeled = CIFAR10C(args, split=split, transform=transform, label_range=unlabeled_list)
        elif args.dataset_subclass in ['new_select']:
            dataset_unlabeled = CIFAR10(args, split=split, transform=transform, label_range=unlabeled_list)
        else:
            raise ValueError('dataset_subclass should be CIFAR10CMix or CIFAR10CAll')
    else:
        raise ValueError('dataset_series should be CIFAR10 or CIFAR100')
    label_loader = DataLoader(dataset = dataset_labeled, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)
    unlabel_loader = DataLoader(dataset = dataset_unlabeled, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)
    return label_loader, unlabel_loader

def get_rot_cifar_loader(args):
    # for selfsupervised learning (labeled and unlabeled samples) seperately load two kinds of data
    # no need test data
    tr_transform = rot_train_transform(resizecrop_size=args.resizecrop_size)
    ts_transform = rot_test_transform(resizecrop_size=args.resizecrop_size)
    if args.dataset_series == 'CIFAR10':
        # labeled data
        if args.dataset_subclass in ['CIFAR10CMix','new_select']:
            train_lab_dataset = CIFAR10(args, split='train', transform = tr_transform, label_range = range(args.num_labeled_classes))
            val_lab_dataset = CIFAR10(args, split='test', transform = ts_transform, label_range = range(args.num_labeled_classes))
        elif args.dataset_subclass in ['CIFAR10CAll']:
            train_lab_dataset = CIFAR10C(args, split='train', transform = tr_transform, label_range = range(args.num_labeled_classes))
            val_lab_dataset = CIFAR10C(args, split='test', transform = ts_transform, label_range = range(args.num_labeled_classes))
        else:
            raise ValueError('dataset_subclass should be CIFAR10CMix or CIFAR10CAll')
        # unlabeled data
        if args.dataset_subclass in ['CIFAR10CMix','CIFAR10CAll']:
            train_unlab_dataset = CIFAR10C(args, split='train', transform = tr_transform, label_range = range(args.num_labeled_classes, args.num_classes))
            val_unlab_dataset = CIFAR10C(args, split='test', transform = ts_transform, label_range = range(args.num_labeled_classes, args.num_classes))
        elif args.dataset_subclass in ['new_select']:
            train_unlab_dataset = CIFAR10(args, split='train', transform = tr_transform, label_range = range(args.num_labeled_classes, args.num_classes))
            val_unlab_dataset = CIFAR10(args, split='test', transform = ts_transform, label_range = range(args.num_labeled_classes, args.num_classes))
        else:
            raise ValueError('dataset_subclass should be CIFAR10CMix or CIFAR10CAll')
    else:
        raise NotImplementedError
    data_loader = dict()
    data_loader['train_lab_loader'] = CIFARRotDataLoader(args = args, dataset = train_lab_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    data_loader['train_unlab_loader'] = CIFARRotDataLoader(args = args, dataset = train_unlab_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    data_loader['test_lab_loader'] = CIFARRotDataLoader(args = args, dataset = val_lab_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    data_loader['test_unlab_loader'] = CIFARRotDataLoader(args = args, dataset = val_unlab_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False) 
    return data_loader

def get_sup_cifar_loader(args):
    # for supervised learning(labeled samples) unlabeled target sample are loaded for mixstyle
    data_loader = dict()
    # labeled data
    if args.dataset_subclass in ['CIFAR10CMix','new_select']:
        data_loader['train_lab_loader'] = CIFARLoader(args, split='train', shuffle=True, label_range = range(args.num_labeled_classes))
        data_loader['test_lab_loader'] = CIFARLoader(args, split='test', shuffle=False, label_range = range(args.num_labeled_classes))
    elif args.dataset_subclass in ['CIFAR10CAll']:
        data_loader['train_lab_loader'] = CIFAR10CLoader(args, split='train', shuffle=True, label_range = range(args.num_labeled_classes))
        data_loader['test_lab_loader'] = CIFAR10CLoader(args, split='test', shuffle=False, label_range = range(args.num_labeled_classes))
    else:
        raise ValueError('dataset_subclass should be CIFAR10CMix or CIFAR10CAll')
    # unlabeled data
    if args.dataset_subclass in ['CIFAR10CMix','CIFAR10CAll']:
        data_loader['train_unlab_loader'] = CIFAR10CLoader(args, split='train', shuffle=True, label_range = range(args.num_labeled_classes, args.num_classes))
        data_loader['test_unlab_loader'] = CIFAR10CLoader(args, split='test', shuffle=False, label_range = range(args.num_labeled_classes, args.num_classes))
    elif args.dataset_subclass in ['new_select']: # cifar10 or cifar100 unlabelled data
        data_loader['train_unlab_loader'] = CIFARLoader(args, split='train', shuffle=True, label_range = range(args.num_labeled_classes, args.num_classes))
        data_loader['test_unlab_loader'] = CIFARLoader(args, split='test', shuffle=False, label_range = range(args.num_labeled_classes, args.num_classes))
    else:
        raise ValueError('dataset_subclass should be CIFAR10CMix or CIFAR10CAll')
    return data_loader

def get_auto_cifar_loader(args):
    # for auto_novel training
    data_loader = dict()
    data_loader['label_loader'], data_loader['unlabel_loader'] = CIFARLoaderMix(args, split='train', shuffle=True, labeled_list=range(args.num_labeled_classes), unlabeled_list=range(args.num_labeled_classes, args.num_classes))
    if args.dataset_subclass in ['CIFAR10CMix','new_select']: # cifar10 or cifar100 labelled data
        data_loader['test_lab_loader'] = CIFARLoader(args, split='test', shuffle=False, label_range = range(args.num_labeled_classes))
    elif args.dataset_subclass in ['CIFAR10CAll']:
        data_loader['test_lab_loader'] = CIFAR10CLoader(args, split='test', shuffle=False, label_range = range(args.num_labeled_classes))
    else:
        raise ValueError('dataset_subclass should be CIFAR10CMix or CIFAR10CAll')
    # unlabeled data
    if args.dataset_subclass in ['CIFAR10CMix','CIFAR10CAll']:
        data_loader['train_unlab_loader'] = CIFAR10CLoader(args, split='train', shuffle=True, label_range = range(args.num_labeled_classes, args.num_classes))
        data_loader['test_unlab_loader'] = CIFAR10CLoader(args, split='test', shuffle=False, label_range = range(args.num_labeled_classes, args.num_classes))
    elif args.dataset_subclass in ['new_select']: # cifar10 or cifar100 unlabelled data
        data_loader['train_unlab_loader'] = CIFARLoader(args, split='train', shuffle=True, label_range = range(args.num_labeled_classes, args.num_classes))
        data_loader['test_unlab_loader'] = CIFARLoader(args, split='test', shuffle=False, label_range = range(args.num_labeled_classes, args.num_classes))
    else:
        raise ValueError('dataset_subclass should be CIFAR10CMix or CIFAR10CAll')
    return data_loader