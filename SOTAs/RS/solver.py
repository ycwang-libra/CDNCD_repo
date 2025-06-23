import os
import torch
from models import SelfsupervisedResNet, SupervisedResNet, BasicBlock, style_encoder
import selfsupervised_learning
import supervised_learning
import auto_novel
import time
from glob import glob

class Solver(object):
    '''Solver for training and testing Network.'''
    def __init__(self, args, data_loader, logger):
        self.args = args
        self.logger = logger
        self.self_data_loader = data_loader['self_data_loader']
        self.sup_data_loader = data_loader['sup_data_loader']
        self.auto_data_loader = data_loader['auto_data_loader']
        self.logger.info('Use GPU: {} for {} {} !'.format(args.gpu, args.stage, args.current_log))
    
    def build_selfsupervised_model(self):
        '''define the RS rotnet model'''
        self.model = SelfsupervisedResNet(self.args, BasicBlock, [2,2,2,2], num_classes=4)
        self.model.cuda(self.args.gpu)

    def build_supervised_model(self):
        '''define the RS resnet model'''
        self.model = SupervisedResNet(self.args, BasicBlock, [2,2,2,2], self.args.num_labeled_classes, self.args.num_unlabeled_classes)
        self.model.cuda(self.args.gpu)
        self.style_model = style_encoder(args = self.args).cuda(self.args.gpu)

    def restore_selfsupervised_model(self, model_saved_path):
        '''restore the RS rotnet model'''
        checkpoint = torch.load(model_saved_path, map_location=lambda storage, loc: storage.cuda(self.args.gpu))
        state_dict = checkpoint['model']
        del state_dict['linear.weight']
        del state_dict['linear.bias']
        self.model.load_state_dict(state_dict, strict=False)
        for name, param in self.model.named_parameters(): 
            if 'head' not in name and 'layer4' not in name:
                param.requires_grad = False # conv1 bn1 layer1 layer2 layer3 fixed
        
        file_stat = os.stat(model_saved_path)
        mtime = file_stat.st_mtime
        saved_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
        self.logger.info("=> loaded selfsupervised model '{}', this model saved time is {} ".format(model_saved_path, saved_time))

        # delete the trained supervised model for saving disk space
        if self.args.del_trained_model:
            used_space = file_stat.st_size
            os.remove(model_saved_path)
            self.logger.info("Delete trained supervised model '{}' for saving {} MB disk space".format(model_saved_path, round(used_space/1024/1024,2)))

        return self.model

    def restore_supervised_model(self, model_saved_path):
        '''restore the RS resnet model'''
        checkpoint = torch.load(model_saved_path, map_location=lambda storage, loc: storage.cuda(self.args.gpu))
        state_dict = checkpoint['model']
        self.model.load_state_dict(state_dict, strict=False)
        for name, param in self.model.named_parameters(): 
            if 'head' not in name and 'layer4' not in name and 'mlp' not in name:
                param.requires_grad = False # conv1.weight bn1.weight bn1.bias layer1 layer2 layer3 fixed layer4 head1 head2 trainable
        
        file_stat = os.stat(model_saved_path)
        mtime = file_stat.st_mtime
        saved_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
        self.logger.info("=> loaded supervised model '{}', this model saved time is {} ".format(model_saved_path, saved_time))

        # delete the trained supervised model for saving disk space
        if self.args.del_trained_model:
            used_space = file_stat.st_size
            os.remove(model_saved_path)
            self.logger.info("Delete trained supervised model '{}' for saving {} MB disk space".format(model_saved_path, round(used_space/1024/1024,2)))

        return self.model

    def restore_autonovel_model(self, model_saved_path):
        '''restore the RS resnet model'''
        checkpoint = torch.load(model_saved_path, map_location=lambda storage, loc: storage.cuda(self.args.gpu))
        state_dict = checkpoint['model']
        self.model.load_state_dict(state_dict, strict=False)

        file_stat = os.stat(model_saved_path)
        mtime = file_stat.st_mtime
        saved_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
        self.logger.info("=> loaded auto_novel model '{}', this model saved time is {} ".format(model_saved_path, saved_time))

        # delete the trained autonovel model for saving disk space
        if self.args.del_trained_model:
            used_space = file_stat.st_size
            os.remove(model_saved_path)
            self.logger.info("Delete trained autonovel model '{}' for saving {} MB disk space".format(model_saved_path, round(used_space/1024/1024,2)))

        return self.model

    #################  train part ######################
    def selfsupervised_learning_train(self):
        '''Train the RS rotnet model'''
        self.logger.info('########## Start to train selfsupervised learning stage ##########')
        self.build_selfsupervised_model()
        selfsupervised_learning.train(self.model, self.self_data_loader, self.args, self.logger) 

    def supervised_learning_train(self):
        '''Train the RS resnet model'''
        self.logger.info('########## Start to train supervised learning stage ##########')
        self.build_supervised_model()
        # load the pretrained selfsupervised model
        if self.args.unlab_corrupt_severity == 0: # no corruption
            path = os.path.join(self.args.trained_model_root, 'checkpoint_best_' + self.args.current_name + '_self_' + self.args.dataset_series + '_' + self.args.dataset_subclass + '_' + self.args.source_domain + str(self.args.num_labeled_classes) + '-' + self.args.target_domain + str(self.args.num_unlabeled_classes) + '_lr' + str(self.args.lr) + '_bs' + str(self.args.batch_size) + '_' + self.args.Server_select + '_seed_' + str(self.args.seed) + '*.pth.tar')
        else:
            path = os.path.join(self.args.trained_model_root, 'checkpoint_best_' + self.args.current_name + '_self_' + self.args.dataset_series + '_' + self.args.dataset_subclass + '_' + self.args.source_domain + str(self.args.num_labeled_classes) + '-' + self.args.target_domain + str(self.args.num_unlabeled_classes) + '_lr' + str(self.args.lr) + '_bs' + str(self.args.batch_size) + '_labcorrupt_' + str(self.args.lab_corrupt_severity) + '_unlabcorrupt_' + str(self.args.unlab_corrupt_severity) + '_corrupt_mode_' + str(self.args.corrupt_mode) + '_' + self.args.Server_select + '_seed_' + str(self.args.seed) + '*.pth.tar')

        selfsupervised_model_path = sorted(glob(path), key=os.path.getmtime)[-1]
        self.model = self.restore_selfsupervised_model(selfsupervised_model_path)
        supervised_learning.train(self.model, self.sup_data_loader, self.args, self.logger)

    def auto_novel_train(self):
        '''Train the RS resnet model'''
        self.logger.info('########## Start to train auto_novel learning stage ##########')
        self.build_supervised_model() # the model in auto_novel is same as supervised_learning
        # load the pretrained supervised model
        if self.args.unlab_corrupt_severity == 0: # no corruption
            path = os.path.join(self.args.trained_model_root, 'checkpoint_last_' + self.args.current_name + '_sup_' + self.args.dataset_series + '_' + self.args.dataset_subclass + '_' + self.args.source_domain + str(self.args.num_labeled_classes) + '-' + self.args.target_domain + str(self.args.num_unlabeled_classes) + '_lr' + str(self.args.lr) + '_bs' + str(self.args.batch_size) + '_' + self.args.Server_select + '_seed_' + str(self.args.seed) + '*.pth.tar')
        else:
            path = os.path.join(self.args.trained_model_root, 'checkpoint_last_' + self.args.current_name + '_sup_' + self.args.dataset_series + '_' + self.args.dataset_subclass + '_' + self.args.source_domain + str(self.args.num_labeled_classes) + '-' + self.args.target_domain + str(self.args.num_unlabeled_classes) + '_lr' + str(self.args.lr) + '_bs' + str(self.args.batch_size) + '_labcorrupt_' + str(self.args.lab_corrupt_severity) + '_unlabcorrupt_' + str(self.args.unlab_corrupt_severity) + '_corrupt_mode_' + str(self.args.corrupt_mode) + '_' + self.args.Server_select + '_seed_' + str(self.args.seed) + '*.pth.tar')
        supervised_model_path = sorted(glob(path), key=os.path.getmtime)[-1]
        self.model = self.restore_supervised_model(supervised_model_path)
        auto_novel.train(self.model, self.style_model, self.auto_data_loader, self.args, self.logger)

    #################  test part ######################
    
    def auto_novel_test(self):
        '''Test the RS resnet model'''
        self.build_supervised_model()
        self.logger.info('########## Start to test auto_novel stage ##########')
        # load the pretrained auto_novel model
        if self.args.unlab_corrupt_severity == 0: # no corruption
            path = os.path.join(self.args.trained_model_root, 'checkpoint_last_' + self.args.current_name + '_auto_' + self.args.dataset_series + '_' + self.args.dataset_subclass + '_' + self.args.source_domain + str(self.args.num_labeled_classes) + '-' + self.args.target_domain + str(self.args.num_unlabeled_classes) + '_lr' + str(self.args.lr) + '_bs' + str(self.args.batch_size) + '_' + self.args.Server_select + '_seed_' + str(self.args.seed) + '*.pth.tar')
        else:
            path = os.path.join(self.args.trained_model_root, 'checkpoint_last_' + self.args.current_name + '_auto_' + self.args.dataset_series + '_' + self.args.dataset_subclass + '_' + self.args.source_domain + str(self.args.num_labeled_classes) + '-' + self.args.target_domain + str(self.args.num_unlabeled_classes) + '_lr' + str(self.args.lr) + '_bs' + str(self.args.batch_size) + '_labcorrupt_' + str(self.args.lab_corrupt_severity) + '_unlabcorrupt_' + str(self.args.unlab_corrupt_severity) + '_corrupt_mode_' + str(self.args.corrupt_mode) + '_' + self.args.Server_select + '_seed_' + str(self.args.seed) + '*.pth.tar')

        autonovel_model_path = sorted(glob(path), key=os.path.getmtime)[0]
        self.model = self.restore_autonovel_model(autonovel_model_path)

        # evaluate by matrics
        if self.args.test_matrics:
            self.args.head = 'head1'
            self.logger.info('Evaluate on labeled classes (test split)')
            auto_novel.test(self.model, self.args, self.logger, self.auto_data_loader['test_lab_loader'], self.auto_data_loader['test_unlab_loader']) 
            
            self.args.head='head2'
            self.logger.info('Evaluate on unlabeled classes (train split)')
            auto_novel.test(self.model, self.args, self.logger, self.auto_data_loader['train_unlab_loader'])
            self.logger.info('Evaluate on unlabeled classes (test split)')
            auto_novel.test(self.model, self.args, self.logger, self.auto_data_loader['test_unlab_loader'])

        # plot the tsne figure
        if self.args.use_tsne_visual:
            auto_novel.test_tsne(self.model, self.args, self.auto_data_loader['test_lab_loader'], self.auto_data_loader['test_unlab_loader'])
            self.logger.info('t-SNE visualization saved to {}'.format(self.args.test_tsne_fig))