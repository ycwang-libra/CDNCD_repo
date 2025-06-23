import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from utils.util import BCE, AverageMeter, PairEnum, cluster_acc
from utils import ramps
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import numpy as np
from sklearn.manifold import TSNE
import wandb
from tqdm import tqdm
from losses import pearson_correlation

def train(model, style_model, data_loader, args, logger):
    label_loader = data_loader['label_loader']
    unlabel_loader = data_loader['unlabel_loader']
    # init wandb
    if args.use_wandb:
        #======================================================================
        wandb.init(project=args.wandb_project, config = args.__dict__, name = args.current_log, save_code=True, mode = 'offline' if args.use_wandb_offline else 'online')
        model.run_id = wandb.run.id
        #======================================================================

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = BCE()
    for epoch in range(args.epochs):
        logger.info('########## epoch: ' + str(epoch) + ' ##########')
        loss_record = AverageMeter()
        model.train()
        target_dataloader_iterator = iter(unlabel_loader)
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length) 
        for batch_idx,((source_img, source_img_bar), source_label, source_domain_flag) in enumerate(tqdm(label_loader)):
            try:
                (target_img, target_img_bar), target_label, target_domain_flag = next(target_dataloader_iterator)
            except:
                target_dataloader_iterator = iter(unlabel_loader)
                (target_img, target_img_bar), target_label, target_domain_flag = next(target_dataloader_iterator)

            img = torch.cat([source_img, target_img], dim = 0)
            img_bar = torch.cat([source_img_bar, target_img_bar], dim = 0)
            label = torch.cat([source_label, target_label], dim = 0)
            domain_flag = torch.cat([source_domain_flag, target_domain_flag], dim = 0)
            img = img.cuda(args.gpu, non_blocking=True)
            img_bar = img_bar.cuda(args.gpu, non_blocking=True)
            label = label.cuda(args.gpu, non_blocking=True)
            domain_flag = domain_flag.cuda(args.gpu, non_blocking=True)

            output1, output2, feat = model(img, domain_flag) # batchsize * num_class
            output1_bar, output2_bar, _ = model(img_bar, domain_flag)
            style_feature = style_model(img, domain_flag)
            
            prob1, prob1_bar, prob2, prob2_bar=F.softmax(output1, dim=1),  F.softmax(output1_bar, dim=1), F.softmax(output2, dim=1), F.softmax(output2_bar, dim=1)
            mask_lb = label<args.num_labeled_classes

            rank_feat = (feat[~mask_lb]).detach()
            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
            rank_idx1, rank_idx2= PairEnum(rank_idx)
            rank_idx1, rank_idx2=rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)

            rank_diff = rank_idx1 - rank_idx2
            rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
            target_ulb = torch.ones_like(rank_diff).float().cuda(args.gpu, non_blocking=True)
            target_ulb[rank_diff>0] = -1 

            prob1_ulb, _= PairEnum(prob2[~mask_lb]) 
            _, prob2_ulb = PairEnum(prob2_bar[~mask_lb])

            loss_ce = criterion1(output1[mask_lb], label[mask_lb])
            loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb)
            consistency_loss = F.mse_loss(prob1, prob1_bar) + F.mse_loss(prob2, prob2_bar)

            # proposed loss
            if args.style_remove_function:
                if args.style_remove_function == 'orth':
                    ################ orth -inf~inf  ################
                    loss_Style_remove = abs(torch.sum(style_feature * feat, dim=1).mean())

                elif args.style_remove_function == 'cossimi':
                    ################ cos similarity -1~1  ################
                    loss_Style_remove = abs(F.cosine_similarity(style_feature, feat, dim=1).mean())

                elif args.style_remove_function == 'coco':
                    ################ COCO -1~1 ################
                    loss_Style_remove = abs(pearson_correlation(style_feature, feat).mean())

                else:
                    raise NotImplementedError('style_remove_function not implemented')
            else:
                loss_Style_remove = torch.zeros(()).cuda(args.gpu, non_blocking=True)

            loss = loss_ce + loss_bce + w * consistency_loss + args.style_remove_loss_w * loss_Style_remove

            loss_record.update(loss.item(), img.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        exp_lr_scheduler.step()

        logger.info('Train Epoch {} finished. Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        logger.info('##### Start to Evaluate at {} epoch #####'.format(epoch))
        
        args.head = 'head1'
        logger.info('Evaluate on labeled classes (test split)')
        labelts_acc, labelts_nmi, labelts_ari = test(model, args, logger, data_loader['test_lab_loader'], data_loader['test_unlab_loader'])  
        
        args.head='head2'
        logger.info('Evaluate on unlabeled classes (train split)')
        unlabeltr_acc, unlabeltr_nmi, unlabeltr_ari = test(model, args, logger, data_loader['train_unlab_loader'])
        logger.info('Evaluate on unlabeled classes (test split)')
        unlabelts_acc, unlabelts_nmi, unlabelts_ari = test(model, args, logger, data_loader['test_unlab_loader'])

        if args.use_wandb:
            #======================================================================
            wandb.log({'autonovel_epoch':epoch, 
            'autonovel_labelts_acc_epoch':labelts_acc,
            'autonovel_labelts_nmi_epoch':labelts_nmi,
            'autonovel_labelts_ari_epoch':labelts_ari,
            'autonovel_unlabeltr_acc_epoch':unlabeltr_acc,
            'autonovel_unlabeltr_nmi_epoch':unlabeltr_nmi,
            'autonovel_unlabeltr_ari_epoch':unlabeltr_ari,
            'autonovel_unlabelts_acc_epoch':unlabelts_acc,
            'autonovel_unlabelts_nmi_epoch':unlabelts_nmi,
            'autonovel_unlabelts_ari_epoch':unlabelts_ari,
            'autonovel_lr_epoch': optimizer.param_groups[0]['lr'],
            'autonovel_loss_ce_epoch':loss_ce.item(),
            'autonovel_loss_bce_epoch':loss_bce.item(),
            'autonovel_loss_consistency_epoch':consistency_loss.item(),
            'autonovel_loss_Style_remove_epoch':loss_Style_remove.item(),
            'autonovel_loss_total_epoch':loss.item(),
            })
            #======================================================================
    
        # save the last autonovel trained model
        state = {
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict(), 
                'epoch': epoch}
        autonovel_last_checkpoint_path = os.path.join(args.trained_model_root, 'checkpoint_last_' + args.current_log_withtime + '.pth.tar')
        torch.save(state, autonovel_last_checkpoint_path)
        logger.info("Final model saved to {}.".format(autonovel_last_checkpoint_path))

def test(model, args, logger, *test_loader):
    model.eval()
    preds=np.array([])
    targets=np.array([])
    if len(test_loader) == 1:
        unlab_loader = test_loader[0]
        for batch_idx, (x, label, domain_flag) in enumerate(tqdm(unlab_loader)): # remove the tqdm
            x = x.cuda(args.gpu, non_blocking=True)
            label = label.cuda(args.gpu, non_blocking=True)
            domain_flag = domain_flag.cuda(args.gpu, non_blocking=True)
            output1, output2, feature = model(x, domain_flag) # feature: batchsize x featuresize
            if args.head=='head1':
                output = output1
            else:
                output = output2
            _, pred = output.max(1)
            targets=np.append(targets, label.cpu().numpy())
            preds=np.append(preds, pred.cpu().numpy())

        acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds)

    elif len(test_loader) == 2:
        lab_loader = test_loader[0]
        unlab_loader = test_loader[1]
        unlab_dataloader_iterator = iter(unlab_loader)
        for batch_idx, (lab_img, lab_label, lab_domain_flag) in enumerate(tqdm(lab_loader)):
            try:
                unlab_img, unlab_label, unlab_domain_flag = next(unlab_dataloader_iterator)
            except:
                unlab_dataloader_iterator = iter(unlab_loader)
                unlab_img, unlab_label, unlab_domain_flag = next(unlab_dataloader_iterator)
            img = torch.cat([lab_img, unlab_img], dim = 0)
            label = lab_label
            domain_flag = torch.cat([lab_domain_flag, unlab_domain_flag], dim = 0)
            img = img.cuda(args.gpu, non_blocking=True)
            label = label.cuda(args.gpu, non_blocking=True)
            domain_flag = domain_flag.cuda(args.gpu, non_blocking=True)
            output1, output2, feature = model(img, domain_flag)  # torch.Size([128,5]) torch.Size([128,5])
            output1,_ = output1.chunk(2) # just for labeled prediction
            output2,_ = output2.chunk(2) # just for labeled prediction
            if args.head=='head1':
                output = output1
            else:
                output = output2
            _, pred = output.max(1)
            targets=np.append(targets, label.cpu().numpy())
            preds=np.append(preds, pred.cpu().numpy())

        acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds) 

    logger.info('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    
    return acc, nmi, ari

def test_tsne(model, args, *test_loader):
    tsne_fig_path = args.test_tsne_fig
    feature_path = args.tsne_root + '/feature_' + args.current_log_withtime + '.npy'
    label_path = args.tsne_root + '/label_' + args.current_log_withtime + '.npy'
    model.eval()
    targets=np.array([])
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    Feature = []
    tsne_result = []

    lab_loader = test_loader[0]
    unlab_loader = test_loader[1]
    unlab_dataloader_iterator = iter(unlab_loader)
    for batch_idx, (lab_img, lab_label, lab_domain_flag) in enumerate(tqdm(lab_loader)):
        try:
            unlab_img, unlab_label, unlab_domain_flag = next(unlab_dataloader_iterator)
        except:
            unlab_dataloader_iterator = iter(unlab_loader)
            unlab_img, unlab_label, unlab_domain_flag = next(unlab_dataloader_iterator)
        img = torch.cat([lab_img, unlab_img], dim = 0)
        label = torch.cat([lab_label, unlab_label], dim = 0)
        domain_flag = torch.cat([lab_domain_flag, unlab_domain_flag], dim = 0)
        img = img.cuda(args.gpu, non_blocking=True)
        label = label.cuda(args.gpu, non_blocking=True)
        domain_flag = domain_flag.cuda(args.gpu, non_blocking=True)
        _, _, feature = model(img, domain_flag)  # torch.Size([128,5]) torch.Size([128,5])
        targets=np.append(targets, label.cpu().numpy())
        Feature.extend(feature.detach().cpu().numpy())
    
    tsne_allbatch_result = tsne.fit_transform(np.array(Feature))
    tsne_result = np.array(tsne_allbatch_result)
    
    # tsne plot
    # tsne_fig = plot_embedding(tsne_result, targets.astype(int), 't-SNE embedding of {} on {} feature'.format(args.current_name, args.dataset_series + args.dataset_subclass))
    # tsne_fig.savefig(tsne_fig_path)
    # save feature and target
    np.save(feature_path, np.array(Feature))
    np.save(label_path, targets.astype(int))
