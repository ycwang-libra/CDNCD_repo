import argparse
import os
import torch
import datetime
from dataloader.cifar_loader import get_rot_cifar_loader, get_sup_cifar_loader, get_auto_cifar_loader
from dataloader.officehome_loader_RS import get_rot_officehome_loader, get_sup_officehome_loader, get_auto_officehome_loader

from solver import Solver
import random
import numpy as np
import logging

def main(args, logger):
    # Data loader two domains
    data_loader = dict()
    if args.dataset_series == 'CIFAR10':
        data_loader['self_data_loader'] = get_rot_cifar_loader(args)
        data_loader['sup_data_loader'] = get_sup_cifar_loader(args)
        data_loader['auto_data_loader'] = get_auto_cifar_loader(args)
    elif args.dataset_series == 'OfficeHome':
        data_loader['self_data_loader'] = get_rot_officehome_loader(args)
        data_loader['sup_data_loader'] = get_sup_officehome_loader(args)
        data_loader['auto_data_loader'] = get_auto_officehome_loader(args)
    else:
        raise NotImplementedError
    
    # def solver for train test every thing
    solver = Solver(args, data_loader, logger)
    if args.stage == 'self':
        logger.info(args.__dict__)
        solver.selfsupervised_learning_train()
    elif args.stage == 'sup':
        logger.info(args.__dict__)
        solver.supervised_learning_train()
    elif args.stage == 'auto':
        logger.info(args.__dict__)
        solver.auto_novel_train()
    elif args.stage == 'test':
        solver.auto_novel_test()
    else:
        raise NotImplementedError('Please set the correct stage name, including self, sup, auto, test')

# parameters setting
parser = argparse.ArgumentParser(description='RS')
parser.add_argument('--Server_select', type=str, default='your server name',choices = ['server1','server2','server3'])
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use in your server.')
parser.add_argument('--log_level', type=str, default='info', choices=['debug', 'info', 'warning', 'error', 'critical'])

# fundamental parameters
parser.add_argument('--total_name', type=str, default='NCD_SOTA_RS') # total name for save the pretrained model
parser.add_argument('--current_name', type=str, default='RS')
parser.add_argument('--stage', type=str, default='self', choices=['self','sup','auto','test'])
parser.add_argument('--style_arch', default='resnet18', help='style model backbone architecture')
parser.add_argument('--changestlye_rate', type=float, default=0, choices=[0, 0.5, 1.0])

parser.add_argument('--batch_size', default=8, type=int, help='input batch size for training') # selfsupervised: 4(roteloader will * 4), supervised: 8, auto_novel: 128 
parser.add_argument('--num_workers', default=16, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')

# optimzer parameters
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD solver')
parser.add_argument('--weight_decay', type=float, default=1e-4) # for supervised learning
parser.add_argument('--step_size', default=10, type=int) # for supervised learning
parser.add_argument('--gamma', type=float, default=0.5) # for supervised learning

parser.add_argument('--rampup_length', default=50, type=int, choices=[50,150,80]) # cifar10:50, # for auto_novel
parser.add_argument('--rampup_coefficient', type=float, default=5.0, choices=[5.0,50]) # cifar10:5.0  # for auto_novel
parser.add_argument('--topk', default=5, type=int)  # for auto_novel

# data
parser.add_argument('--dataset_series', type=str, default='DomainNet',choices=['CIFAR10', 'CIFAR100', 'SVHN', 'DomainNet','ImageNet', 'OfficeHome'])
parser.add_argument('--dataset_subclass', type=str, default='new_select',choices=['all','first10','first100','mammal','mammal_tiny','select_tiny','new_select','CIFAR10CMix','CIFAR10CAll'])
parser.add_argument('--source_domain', type=str, default='real',choices=['sketch','clipart','painting','real','Real_World','Art', 'Clipart', 'Product'])
parser.add_argument('--target_domain', type=str, default='sketch',choices=['sketch','clipart','painting','real','Real_World','Art', 'Clipart', 'Product'])
parser.add_argument('--lab_corrupt_severity', default=0, type=int, choices = [0,1,2,3,4,5])
parser.add_argument('--unlab_corrupt_severity', default=0, type=int, choices = [0,1,2,3,4,5])
parser.add_argument('--corrupt_mode', type=str, default='gaussian_blur',choices=['gaussian_blur','jpeg_compression', 'impulse_noise'])
parser.add_argument('--num_labeled_classes', default=50, type=int)
parser.add_argument('--num_unlabeled_classes', default=50, type=int) 

parser.add_argument('--resizecrop_size', type=int, default=128)
parser.add_argument('--use_wandb', default=False, type=bool, help='use the wandb to visualize the training process',choices=[True,False])
parser.add_argument('--use_wandb_offline', default=False, type=bool, help='use the wandb offline mode',choices=[True,False])

# for test switch
parser.add_argument('--test_matrics', default=False, type=bool, help='use or not test matrics',choices=[True,False])
parser.add_argument('--use_tsne_visual', default=False, type=bool, help='use the t-SNE visualization',choices=[True,False])
parser.add_argument('--del_trained_model', type=bool, default=False, choices = [True,False], help='delate the trained model for saving disk space')

# for stye removing 
parser.add_argument("--proj_dim", default=256, type=int, help="projected dim")
parser.add_argument('--style_remove_function', type=str, default=None, choices=['orth', 'coco', 'cossimi']) # if None, no style remove loss
parser.add_argument('--style_remove_loss_w', default=0.01, type=float, help='weight for supervised style_remove_loss_w')

args = parser.parse_args()

args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
args.log_loss = '_RSloss'
if args.style_remove_function:
    args.log_loss += '_' + args.style_remove_function + str(args.orthogonalloss_w)

# set the server
if args.Server_select == 'server1':
    args.dataset_root = 'data_root_path/' + args.dataset_series # TODO fill your dataset path
    args.aim_root_path = 'aim_path/' # TODO all files produced by experiment will be saved here
    args.gpu = 0
elif args.Server_select in ['server2','server3']:
    args.dataset_root = 'data_root_path/' + args.dataset_subclass # TODO
    args.aim_root_path = 'aim_path/' # TODO
else:
    raise ValueError('Please set the correct server name')

# add log direction
experiment_time = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
if args.unlab_corrupt_severity == 0: # no corruption
    args.current_log = args.current_name + '_' + args.stage + '_' + args.dataset_series + '_' + args.dataset_subclass + '_' + args.source_domain + str(args.num_labeled_classes) + '-' + args.target_domain + str(args.num_unlabeled_classes) + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) + '_' + args.Server_select + '_seed_' + str(args.seed)
else:
    args.current_log = args.current_name + '_' + args.stage + '_' + args.dataset_series + '_' + args.dataset_subclass + '_' + args.source_domain + str(args.num_labeled_classes) + '-' + args.target_domain + str(args.num_unlabeled_classes) + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) + '_labcorrupt_' + str(args.lab_corrupt_severity) + '_unlabcorrupt_' + str(args.unlab_corrupt_severity) + '_corrupt_mode_' + str(args.corrupt_mode) + '_' + args.Server_select + '_seed_' + str(args.seed)

args.current_log += args.log_loss

args.current_log_withtime = args.current_log + '_' + experiment_time

args.trained_model_root = args.aim_root_path + 'trained_models/' + args.total_name + '/'
args.log_root = args.aim_root_path + 'logs/' + args.total_name + '/'
args.wandb_log_root = args.aim_root_path + 'wandb_logs/' + args.total_name + '/' + args.current_log_withtime + '/'
args.tsne_root = args.aim_root_path + 'tsne/' + args.total_name + '/'

if not os.path.exists(args.trained_model_root):
    os.makedirs(args.trained_model_root)
if not os.path.exists(args.log_root):
    os.makedirs(args.log_root)
if not os.path.exists(args.wandb_log_root):
    os.makedirs(args.wandb_log_root)
if not os.path.exists(args.tsne_root):
    os.makedirs(args.tsne_root)
if not os.path.exists(args.total_name):
    os.makedirs(args.total_name)

args.log_txt_dir_withtime = args.log_root + '/' + args.current_log_withtime + '.txt'
# important small files saved in this repository
args.test_log_txt_dir_withtime = args.total_name + '/' + args.current_log_withtime + '.txt'
args.test_tsne_fig = args.total_name + '/t-SNE_' + args.current_log_withtime + '.pdf'

# wandb config
args.wandb_project = args.total_name + '_' + args.stage
os.environ['WANDB_DIR'] = args.wandb_log_root
os.environ["WANDB_SILENT"] = "true"

if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

if __name__ == '__main__':
    # init logger
    if args.stage == 'test':
        log_dir = args.test_log_txt_dir_withtime
    else:
        log_dir = args.log_txt_dir_withtime
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_dir), logging.StreamHandler()])
    logger = logging.getLogger(args.total_name)
    if args.log_level == 'debug':
        logger.setLevel(logging.DEBUG)
    elif args.log_level == 'info':
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    
    main(args, logger)