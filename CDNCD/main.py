import argparse
import os
import torch
from solver import Solver
import random
import numpy as np
import logging
import datetime

def main(args, logger):
    # Data loaders
    if args.dataset_series == 'DomainNet':
        from dataloader.domainnet40_loader import get_DomainNet_loader
        data_loader = get_DomainNet_loader(args)
    elif args.dataset_series == 'CIFAR10':
        from dataloader.cifar_loader import get_CIFAR_loader
        data_loader = get_CIFAR_loader(args)
    elif args.dataset_series == 'OfficeHome':
        from dataloader.officehome_loader import get_OfficeHome_loader
        data_loader = get_OfficeHome_loader(args)
    else:
        raise NotImplementedError

    # def solver for train test every thing
    solver = Solver(args, data_loader, logger)
    
    # train or test
    if args.mode == 'train':
        logger.info(args.__dict__)
        solver.train()
    elif args.mode == 'test':
        solver.test()
    else:
        raise NotImplementedError
    
parser = argparse.ArgumentParser(description='CDNCD')
parser.add_argument('--Server_select', type=str, default='your server name',choices = ['server1','server2','server3'])
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use in your server.')
parser.add_argument('--log_level', type=str, default='info', choices=['debug', 'info', 'warning', 'error', 'critical'])

# fundamental parameters
parser.add_argument('--total_name', type=str, default='CDNCD') # total name for save the trained model
parser.add_argument('--current_name', type=str, default='Prototype_code') # prototype code name 
parser.add_argument('--mode', type=str, default='test', choices=['train','test'])
parser.add_argument('--arch', default='DINO', help='backbone architecture')
parser.add_argument('--style_arch', default='resnet18', help='style model backbone architecture')
parser.add_argument('--optimizer', type=str, default='sgd', choices = ['sgd','adam'])
parser.add_argument('--resume', type=bool, default=False, choices = [True,False]) # resume to train

parser.add_argument('--num_workers', default=16, type=int, help='number of data loading workers (default: 16)')
parser.add_argument('--resizecrop_size', type=int, default=32)
parser.add_argument('--changestyle_rate', type=float, default=0.0, choices=[0.0, 0.5, 1.0])
parser.add_argument('--batch_size', default=4, type=int, help='mini-batch size')
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')

parser.add_argument('--dataset_series', type=str, default='CIFAR10',choices=['CIFAR10', 'DomainNet','OfficeHome'])
parser.add_argument("--download", default=True, help="whether to download")
parser.add_argument('--dataset_subclass', type=str, default='CIFAR10CMix',choices=['all','new_select','CIFAR10CMix','CIFAR10CAll','DomainNet40'])
parser.add_argument('--source_domain', type=str, default='real',choices=['sketch','clipart','painting','real','Real_World','Art', 'Clipart', 'Product']) # for DomainNet, OfficeHome
parser.add_argument('--target_domain', type=str, default='real',choices=['sketch','clipart','painting','real','Real_World','Art', 'Clipart', 'Product']) # for DomainNet, OfficeHome
parser.add_argument('--lab_corrupt_severity', default=0, type=int, choices = [0,1,2,3,4,5])
parser.add_argument('--unlab_corrupt_severity', default=0, type=int, choices = [0,1,2,3,4,5])
parser.add_argument('--corrupt_mode', type=str, default='gaussian_blur',choices=['gaussian_blur','jpeg_compression', 'impulse_noise'])
parser.add_argument('--num_labeled_classes', default=5, type=int)
parser.add_argument('--num_unlabeled_classes', default=5, type=int) 
parser.add_argument('--use_wandb', default=False, type=bool, help='use the wandb to visualize the training process',choices=[True,False])
parser.add_argument('--use_wandb_offline', default=False, type=bool, help='use the wandb offline mode',choices=[True,False])

# SimGCD parameters
parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])
parser.add_argument('--warmup_model_dir', type=str, default=None)
parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
parser.add_argument('--prop_train_labels', type=float, default=0.5)
parser.add_argument('--use_ssb_splits', action='store_true', default=True)

parser.add_argument('--grad_from_block', type=int, default=11)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument('--transform', type=str, default='imagenet')
parser.add_argument('--sup_weight', type=float, default=0.35)
parser.add_argument('--n_views', default=2, type=int)
parser.add_argument('--memax_weight', type=float, default=1)
parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')
parser.add_argument('--fp16', action='store_true', default=False)

# style remove loss switch
parser.add_argument('--style_remove_function', type=str, default=None, choices=['orth', 'coco', 'cossimi']) # if None, no style remove loss

# loss weight
parser.add_argument('--style_remove_loss_w', default=0.01, type=float, help='weight for supervised style_remove_loss_w')

# for style encoder
parser.add_argument('--dim_style', default = 50, help='style feature dimension')

# for test switch
parser.add_argument('--test_matrics', default=False, type=bool, help='use or not test matrics',choices=[True,False])
parser.add_argument('--tsne_embedding', default=False, type=bool, help='use the t-SNE embedding',choices=[True,False])
parser.add_argument('--tsne_plot', default=False, type=bool, help='use the t-SNE visualization',choices=[True,False])
parser.add_argument('--use_cam_visual', default=False, type=bool, help='use the CAM visualization',choices=[True,False])
parser.add_argument('--delete_model', default=False, type=bool, help='delete the model after test',choices=[True,False])

args = parser.parse_args()

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

# add some args
args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes

# model and log direcion setting
args.log_loss = '_SimGCDloss'
if args.style_remove_function:
    args.log_loss += '_' + args.style_remove_function + str(args.orthogonalloss_w)

# add log direction
experiment_time = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
if args.unlab_corrupt_severity == 0: # no corruption
    args.current_log = args.current_name + '_' + args.dataset_series + '_' + args.dataset_subclass + '_' + args.source_domain + str(args.num_labeled_classes) + '-' + args.target_domain + str(args.num_unlabeled_classes)+ '_arch_' + str(args.arch) + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) + '_' + args.Server_select + '_seed_' + str(args.seed)
else: # corruption
    args.current_log = args.current_name + '_' + args.dataset_series + '_' + args.dataset_subclass + '_' + args.source_domain + str(args.num_labeled_classes) + '-' + args.target_domain + str(args.num_unlabeled_classes)+ '_arch_' + str(args.arch) + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) + '_labcorrupt_' + str(args.lab_corrupt_severity) + '_unlabcorrupt_' + str(args.unlab_corrupt_severity) + '_corrupt_mode_' + str(args.corrupt_mode) + '_' + args.Server_select + '_seed_' + str(args.seed)

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
# tsne
args.test_tsne_fig = args.total_name + '/t-SNE_' + args.current_log_withtime + '.pdf'
args.test_lab_tsne_fig = args.total_name + '/lab_t-SNE_' + args.current_log_withtime + '.pdf'
args.test_unlab_tsne_fig = args.total_name + '/unlab_t-SNE_' + args.current_log_withtime + '.pdf'

# wandb config
args.wandb_project = args.total_name

# SimGCD parameters
args.feat_dim = 768
args.mlp_out_dim = args.num_classes
args.num_mlp_layers = 3

os.environ['WANDB_DIR'] = args.wandb_log_root # change the wandb log directory
os.environ["WANDB_SILENT"] = "true" # turn off the wandb log in the console

if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

if __name__ == '__main__':
    # init logger
    if args.mode == 'train':
        log_dir = args.log_txt_dir_withtime
    else:
        log_dir = args.test_log_txt_dir_withtime
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_dir), logging.StreamHandler()])
    # logging.basicConfig(
    #     format="%(asctime)s [%(levelname)s] %(message)s",
    #     handlers=[logging.StreamHandler()]) # log only shown in the console not save
    logger = logging.getLogger(args.total_name)
    if args.log_level == 'debug':
        logger.setLevel(logging.DEBUG)
    elif args.log_level == 'info':
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    main(args, logger)