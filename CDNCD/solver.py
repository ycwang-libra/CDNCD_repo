import torch
import torch.nn as nn
from models import DINOHead, base_model
import CDNCD
from glob import glob
import os
import time

class Solver(object):
    '''Solver for training and testing Network.'''
    def __init__(self, args, data_loader, logger):
        self.args = args
        self.logger = logger
        self.data_loader = data_loader
        self.logger.info('Use GPU: {} for {} {} !'.format(args.gpu, args.mode, args.current_log))

    def build_model(self):
        backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        for m in backbone.parameters():
            m.requires_grad = False
        
        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in backbone.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= self.args.grad_from_block:
                    m.requires_grad = True
        
        self.projector = DINOHead(args = self.args)
        self.model = nn.Sequential(backbone, self.projector).cuda(self.args.gpu)
        self.style_model = base_model(args = self.args).cuda(self.args.gpu)
        self.print_network(self.model, 'DINO')
        self.print_network(self.style_model, 'style_model')
        return self.model, self.style_model

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        self.logger.debug(model)
        self.logger.info(name)
        self.logger.info("The number of parameters: {}".format(num_params))

    def restore_trained_for_test(self, model_save_dir):
        self.model, self.style_model = self.build_model()

        checkpoint = torch.load(model_save_dir, map_location=lambda storage, loc: storage.cuda(self.args.gpu))
        self.model.load_state_dict(checkpoint['model'])

        file_stat = os.stat(model_save_dir)
        mtime = file_stat.st_mtime
        saved_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
        self.logger.info("=> loaded pretrained model '{}', this model saved time is {} ".format(model_save_dir, saved_time))

        # delete the trained discover model for saving disk space
        if self.args.delete_model:
            used_space = file_stat.st_size
            os.remove(model_save_dir)
            self.logger.info("Delete trained discover model '{}' for saving {} MB disk space".format(model_save_dir, round(used_space/1024/1024,2)))

    def train(self):
        self.logger.info('########## Start to train stage ##########')
        self.build_model()
        CDNCD.train(self.model, self.style_model, self.data_loader, self.args, self.logger)

    def test(self):
        self.logger.info('########## Start to test stage ##########')
        model_save_dir = sorted(glob(self.args.trained_model_root + 'checkpoint_last_' + self.args.current_log + '_2024*'), key=os.path.getmtime)[-1]

        self.restore_trained_for_test(model_save_dir)

        # evaluate by matrics
        if self.args.test_matrics:
            self.logger.info('Evaluate on labeled classes (test split)')
            CDNCD.test(self.model, self.args, self.logger, self.data_loader['test_lab_loader'])

            self.logger.info('Evaluate on unlabeled classes (train split)')
            CDNCD.test(self.model, self.args, self.logger, self.data_loader['train_unlab_loader'])

            self.logger.info('Evaluate on unlabeled classes (test split)')
            CDNCD.test(self.model, self.args, self.logger, self.data_loader['test_unlab_loader'])

        # save tsne feature and label
        if self.args.tsne_embedding:
            CDNCD.test_tsne(self.model, self.args, self.logger, self.data_loader['test_lab_loader'], self.data_loader['test_unlab_loader'])
            self.logger.info('t-SNE embeding feature and label saved to {}'.format(self.args.tsne_root + '/feature and label_' + self.args.current_log_withtime + '.npy'))