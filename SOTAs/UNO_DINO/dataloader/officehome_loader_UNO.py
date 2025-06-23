# officehome data with cross domain or one domain

from torch.utils.data import DataLoader
import torch.utils.data as data
from PIL import Image
import os
from util.transforms import get_transforms

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

def OfficeHomeLoader(args, lab_flag, domain, split='train', transform_mode = None, shuffle = None, label_range=range(5)):
    if args.stage == 'pretrain':
        if transform_mode == 'train':
            transform = get_transforms(args, "unsupervised", 'OfficeHome')
        elif transform_mode == 'test':
            transform = get_transforms(args, "eval", 'OfficeHome')
    elif args.stage == 'discover':
        if transform_mode == 'train':
            transform = get_transforms(args, "unsupervised", 'OfficeHome', 
                                       multicrop=args.multicrop, 
                                       num_large_crops=args.num_large_crops, 
                                       num_small_crops=args.num_small_crops)
        elif transform_mode == 'test':
            transform = get_transforms(args, "eval", 'OfficeHome')
    elif args.stage == 'test':
        transform = get_transforms(args, "eval", 'OfficeHome')
    else:
        raise ValueError('stage should be pretrain, discover or test')
    
    dataset = OfficeHome(args, lab_flag, domain, split=split, transform = transform, label_range=label_range)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    return data_loader

def get_OfficeHome_loader(args):
    # seperately load two domain data (lab and unlab)
    data_loader = dict()
    data_loader['train_lab_loader'] = OfficeHomeLoader(args, 'lab', args.source_domain, split='train', transform_mode='train', shuffle=True, label_range = range(args.num_labeled_classes))
    data_loader['train_unlab_loader'] = OfficeHomeLoader(args, 'unlab', args.target_domain, split='train', transform_mode='train', shuffle=True, label_range = range(args.num_labeled_classes, args.num_classes))
    data_loader['test_lab_loader'] = OfficeHomeLoader(args, 'lab', args.source_domain, split='test', transform_mode='test', shuffle=False, label_range = range(args.num_labeled_classes))
    data_loader['test_unlab_loader'] = OfficeHomeLoader(args, 'unlab', args.target_domain, split='test', transform_mode='test', shuffle=False, label_range = range(args.num_labeled_classes, args.num_classes))
    # HACK in discover stage use 'Evaluate on unlabeled classes (train split)' add transform_mode='test'
    data_loader['train_unlab_loader2'] = OfficeHomeLoader(args, 'unlab', args.target_domain, split='train', transform_mode='test', shuffle=False, label_range = range(args.num_labeled_classes, args.num_classes))
    return data_loader