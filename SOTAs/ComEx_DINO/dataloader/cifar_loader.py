import torch.utils.data as data
import os
import numpy as np
import pickle
from util.util import check_integrity, download_url
from PIL import Image
from util.transforms import get_transforms
from torch.utils.data import DataLoader

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
            file_path = os.path.join(self.root, self.base_folder, file_name) # './data/datasets/CIFAR/cifar-10-batches-py/data_batch_1'  data_batch_1,2,3,4,5
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

        self.data = self.data[ind]  # labeled classes data (25000, 32, 32, 3)
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

        img = self.cifar10c_data[index, :, :, :]
        target = self.cifar10c_label[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        domain_flag = 1 # always 1 for CIFAR10c(cifar10 only for target domain)
        return img, target, domain_flag  #torch.Tensor  np.uint8

    def __len__(self):
        return len(self.cifar10c_data)

def CIFARLoader(args, split='train', transform_mode = None, shuffle = None, label_range=range(5)):
    if args.stage == 'pretrain':
        if transform_mode == 'train':
            transform = get_transforms(args, "unsupervised", args.dataset_series, args.num_views)
        elif transform_mode == 'test':
            transform = get_transforms(args, "eval", args.dataset_series, args.num_views)
    elif args.stage == 'discover':
        if transform_mode == 'train':
            transform = get_transforms(args, "unsupervised", args.dataset_series, args.num_views)
        elif transform_mode == 'test':
            transform = get_transforms(args, "eval", args.dataset_series, args.num_views)
    elif args.stage == 'test':
        transform = get_transforms(args, "eval", args.dataset_series, args.num_views)
    else:
        raise ValueError('stage should be pretrain, discover or test')

    if args.dataset_series == 'CIFAR10':
        dataset = CIFAR10(args, split=split, transform = transform, label_range=label_range)
    else:
        raise NotImplementedError

    if transform_mode == 'train': # HACK
        drop_last = True
    else:
        drop_last = False
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, drop_last=drop_last, pin_memory=True)

    return data_loader

def CIFAR10CLoader(args, split='train', transform_mode = None, shuffle = None, label_range=range(5)):
    if args.stage == 'pretrain':
        if transform_mode == 'train':
            transform = get_transforms(args, "unsupervised", args.dataset_series, args.num_views)
        elif transform_mode == 'test':
            transform = get_transforms(args, "eval", args.dataset_series, args.num_views)
    elif args.stage == 'discover':
        if transform_mode == 'train':
            transform = get_transforms(args, "unsupervised", args.dataset_series, args.num_views)
        elif transform_mode == 'test':
            transform = get_transforms(args, "eval", args.dataset_series, args.num_views)
    elif args.stage == 'test':
        transform = get_transforms(args, "eval", args.dataset_series, args.num_views)
    else:
        raise ValueError('stage should be pretrain, discover or test')

    dataset = CIFAR10C(args, split=split, transform = transform, label_range=label_range)
    
    if transform_mode == 'train': # HACK
        drop_last = True
    else:
        drop_last = False
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, drop_last=drop_last, pin_memory=True)
    return data_loader

def get_CIFAR_loader(args):
    # process both cifar10 and cifar100
    # seperately load two part data (labeled and unlabeled)
    data_loader = dict()
    # labeled data
    if args.dataset_subclass in ['CIFAR10CMix','new_select']: # cifar10 or cifar100 labelled data
        data_loader['train_lab_loader'] = CIFARLoader(args, split='train', transform_mode='train', shuffle=True, label_range = range(args.num_base_classes))
        data_loader['test_lab_loader'] = CIFARLoader(args, split='test', transform_mode='test', shuffle=False, label_range = range(args.num_base_classes))
    elif args.dataset_subclass in ['CIFAR10CAll']:
        data_loader['train_lab_loader'] = CIFAR10CLoader(args, split='train', transform_mode='train', shuffle=True, label_range = range(args.num_base_classes))
        data_loader['test_lab_loader'] = CIFAR10CLoader(args, split='test', transform_mode='test', shuffle=False, label_range = range(args.num_base_classes))
    else:
        raise NotImplementedError
    # unlabeled data
    if args.dataset_subclass in ['CIFAR10CMix','CIFAR10CAll']:
        data_loader['train_unlab_loader'] = CIFAR10CLoader(args, split='train', transform_mode='train', shuffle=True, label_range = range(args.num_base_classes, args.num_classes))
        data_loader['test_unlab_loader'] = CIFAR10CLoader(args, split='test', transform_mode='test', shuffle=False, label_range = range(args.num_base_classes, args.num_classes))
        # HACK in discover stage use 'Evaluate on unlabeled classes (train split)' add transform_mode='test'
        data_loader['train_unlab_loader2'] = CIFAR10CLoader(args, split='train', transform_mode='test', shuffle=False, label_range = range(args.num_base_classes, args.num_classes))
    elif args.dataset_subclass in ['new_select']: # cifar10 or cifar100 unlabelled data
        data_loader['train_unlab_loader'] = CIFARLoader(args, split='train', transform_mode='train', shuffle=True, label_range = range(args.num_base_classes, args.num_classes))
        data_loader['test_unlab_loader'] = CIFARLoader(args, split='test', transform_mode='test', shuffle=False, label_range = range(args.num_base_classes, args.num_classes))
        # HACK in discover stage use 'Evaluate on unlabeled classes (train split)' add transform_mode='test'
        data_loader['train_unlab_loader2'] = CIFARLoader(args, split='train', transform_mode='test', shuffle=False, label_range = range(args.num_base_classes, args.num_classes))

    return data_loader