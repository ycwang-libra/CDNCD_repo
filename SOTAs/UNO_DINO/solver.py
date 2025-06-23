import os
from models import MultiHeadModel, base_model
import torch
import pretrain
import discover
import time
from glob import glob

class Solver(object):
    '''Solver for training and testing Network.'''
    def __init__(self, args, data_loader, logger):
        self.args = args
        self.logger = logger
        ## Data loader.
        self.data_loader = data_loader
        self.logger.info('Use GPU: {} for {} {} !'.format(args.gpu, args.stage, args.current_log))

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        self.logger.debug(model)
        self.logger.info(name)
        self.logger.info("The number of parameters: {}".format(num_params))

    def build_pretrain_model(self):
        self.pretrain_model = MultiHeadModel(
            args = self.args,
            arch = self.args.arch,
            low_res="CIFAR" in self.args.dataset_series,
            num_labeled=self.args.num_labeled_classes,
            num_unlabeled=self.args.num_unlabeled_classes,
            num_heads=None,
        )
        self.pretrain_model.cuda(self.args.gpu)
        self.print_network(self.pretrain_model, 'UNO_{}'.format(self.args.arch))

    def build_discover_model(self):
        self.discover_model = MultiHeadModel(
            args = self.args,
            arch=self.args.arch,
            low_res="CIFAR" in self.args.dataset_series,
            num_labeled=self.args.num_labeled_classes,
            num_unlabeled=self.args.num_unlabeled_classes,
            proj_dim=self.args.proj_dim,
            hidden_dim=self.args.hidden_dim,
            overcluster_factor=self.args.overcluster_factor,
            num_heads=self.args.num_heads,
            num_hidden_layers=self.args.num_hidden_layers,
        )
        self.discover_model.cuda(self.args.gpu)
        self.style_model = base_model(args = self.args).cuda(self.args.gpu)
        self.print_network(self.style_model, 'style_model')
        self.print_network(self.discover_model, 'UNO_{}'.format(self.args.arch))

    def restore_pretrain_for_discover(self, model_saved_path):
        self.build_discover_model()
        # load pretrained model
        checkpoint = torch.load(model_saved_path, map_location=lambda storage, loc: storage.cuda(self.args.gpu))

        state_dict = {k: v for k, v in checkpoint.items() if ("unlab" not in k)}
        self.discover_model.load_state_dict(state_dict, strict=False)

        file_stat = os.stat(model_saved_path)
        mtime = file_stat.st_mtime
        saved_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
        self.logger.info("=> loaded pretrained model '{}', this model saved time is {} ".format(model_saved_path, saved_time))

    def restore_discover_for_test(self, model_saved_path):
        self.build_discover_model()
        checkpoint = torch.load(model_saved_path, map_location=lambda storage, loc: storage.cuda(self.args.gpu))
        self.discover_model.load_state_dict(checkpoint['model_state_dict'])

        file_stat = os.stat(model_saved_path)
        mtime = file_stat.st_mtime
        saved_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
        self.logger.info("=> loaded pretrained model '{}', this model saved time is {} ".format(model_saved_path, saved_time))

    def remove_trained_model(self, model_saved_path):
        # delete the trained discover model for saving disk space
        file_stat = os.stat(model_saved_path)
        used_space = file_stat.st_size
        os.remove(model_saved_path)
        self.logger.info("Delete trained discover model '{}' for saving {} MB disk space".format(model_saved_path, round(used_space/1024/1024,2)))
    
    def pretrain(self):
        self.logger.info('########## Start to pretrain stage ##########')
        self.build_pretrain_model()
        pretrain.train(self.pretrain_model, self.data_loader, self.args, self.logger)

    def discover(self):
        self.logger.info('########## Start to discover stage ##########')
        pretrained_model_name = 'checkpoint_last_' + self.args.current_name + '_pretrain_' + self.args.dataset_series + '_' + self.args.dataset_subclass + '_' + self.args.source_domain + str(self.args.num_labeled_classes) + '-' + self.args.target_domain + str(self.args.num_unlabeled_classes) + '_arch_' + str(self.args.arch) + '_lr'+ str(self.args.base_lr) + '_bs' + str(self.args.batch_size) + '_labcorrupt_' + str(self.args.lab_corrupt_severity) + '_unlabcorrupt_' + str(self.args.unlab_corrupt_severity) + '_corrupt_mode_' + str(self.args.corrupt_mode) + '_' + self.args.Server_select + '_seed_' + str(self.args.seed)

        pretrained_model_save_dir = sorted(glob(self.args.trained_model_root + '/' + pretrained_model_name + '*'), key=os.path.getmtime)[-1]
        
        self.restore_pretrain_for_discover(pretrained_model_save_dir)
        discover.train(self.discover_model, self.style_model, self.data_loader, self.args, self.logger)

        if self.args.del_trained_model:
            self.logger.info('After discovering, then delete the pretrained model for saving disk space')
            self.remove_trained_model(pretrained_model_save_dir)

    def test(self):
        self.logger.info('########## Start to test stage ##########')
        discover_model_name = 'checkpoint_last_' + self.args.current_name + '_discover_' + self.args.dataset_series + '_' + self.args.dataset_subclass + '_' + self.args.source_domain + str(self.args.num_labeled_classes) + '-' + self.args.target_domain + str(self.args.num_unlabeled_classes) + '_arch_' + str(self.args.arch) + '_lr' + str(self.args.base_lr) + '_bs' + str(self.args.batch_size) + '_labcorrupt_' + str(self.args.lab_corrupt_severity) + '_unlabcorrupt_' + str(self.args.unlab_corrupt_severity) + '_corrupt_mode_' + str(self.args.corrupt_mode) + '_' + self.args.Server_select + '_seed_' + str(self.args.seed)
        
        discover_model_save_dir = sorted(glob(self.args.trained_model_root + discover_model_name + '*'), key=os.path.getmtime)[-1]
        
        self.restore_discover_for_test(discover_model_save_dir)
        
        # evaluate the model with three metrics
        if self.args.test_matrics:
            self.logger.info('Evaluate on labeled classes (test split)')
            discover.test(self.discover_model, self.args, self.logger, 'lab/test', self.data_loader['test_lab_loader'])

            self.logger.info('Evaluate on unlabeled classes (train split)')
            discover.test(self.discover_model, self.args, self.logger, 'unlab/train', self.data_loader['train_unlab_loader2'])

            self.logger.info('Evaluate on unlabeled classes (test split)')
            discover.test(self.discover_model, self.args, self.logger, 'unlab/test', self.data_loader['test_unlab_loader'])

        # plot the tsne figure
        if self.args.save_tsne_feature:
            discover.test_tsne(self.discover_model, self.args, self.data_loader['test_lab_loader'], self.data_loader['test_unlab_loader'])
            self.logger.info('t-SNE feature saved to {}'.format(self.args.tsne_root))
        
        if self.args.del_trained_model:
            self.logger.info('After testing, then delete the trained discover model for saving disk space')
            self.remove_trained_model(discover_model_save_dir)
