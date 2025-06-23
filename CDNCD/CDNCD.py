import torch
import wandb
from util.util import get_params_groups, AverageMeter, info_nce_logits, cluster_acc
from torch.optim import SGD, lr_scheduler
from losses import DistillLoss, SupConLoss, pearson_correlation
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from tqdm import tqdm
import torch.nn as nn
import math
import os
import numpy as np
from sklearn.manifold import TSNE
from util.tsne_util import plot_embedding
import torch.nn.functional as F

def train(model,style_model, data_loader, args, loggers):
    train_lab_loader = data_loader['train_lab_loader']
    train_unlab_loader = data_loader['train_unlab_loader']
    # init wandb
    if args.use_wandb:
        #======================================================================
        wandb.init(project=args.wandb_project, config = args.__dict__, name = args.current_log_withtime, save_code=True, mode = 'offline' if args.use_wandb_offline else 'online')
        model.run_id = wandb.run.id
        #======================================================================
    
    params_groups = get_params_groups(model)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    style_optimizer = SGD(style_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)
    exp_lr_scheduler_style = lr_scheduler.CosineAnnealingLR(style_optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)
    cluster_criterion = DistillLoss(args.warmup_teacher_temp_epochs, args.epochs, args.n_views, args.warmup_teacher_temp, args.teacher_temp)

    for epoch in range(args.epochs):
        loggers.info('########## Start epoch: ' + str(epoch) + ' ##########')
        loss_record = AverageMeter()

        model.train()
        style_model.train()
        unlab_dataloader_iterator = iter(train_unlab_loader)
        for _, (lab_imgs, lab_label, lab_domain_flag) in enumerate(tqdm(train_lab_loader)):
            try:
                unlab_imgs, unlab_label, unlab_domain_flag = next(unlab_dataloader_iterator)
            except:
                unlab_dataloader_iterator = iter(train_unlab_loader)
                unlab_imgs, unlab_label, unlab_domain_flag = next(unlab_dataloader_iterator)
            
            q_lab, k_lab = lab_imgs # q_lab torch.Size([bs, 3, 128, 128])
            q_unlab, k_unlab = unlab_imgs # q_unlab torch.Size([bs, 3, 128, 128])

            im_q = torch.cat([q_lab, q_unlab], dim = 0) # torch.Size([2*bs])
            im_k = torch.cat([k_lab, k_unlab], dim = 0) # torch.Size([2*bs])
            labels = torch.cat([lab_label, unlab_label], dim = 0) # torch.Size([2*bs])
            domain_flag = torch.cat([lab_domain_flag, unlab_domain_flag], dim = 0) # torch.Size([2*bs])
            domain_flag = torch.cat([domain_flag, domain_flag], dim = 0) # torch.Size([4*bs])
            
            im_q = im_q.cuda(args.gpu, non_blocking=True)
            im_k = im_k.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True) # only lab data has label
            domain_flag = domain_flag.cuda(args.gpu, non_blocking=True)
            mask_lab = labels<args.num_labeled_classes
            mask_lab = mask_lab.cuda(args.gpu, non_blocking=True).bool()

            images = torch.cat([im_q, im_k], dim=0) # torch.Size([2*bs, 3, 128, 128])

            student_proj, student_out = model(images)
            style_feature = style_model(images, domain_flag)
            # images: torch.Size([4*bs, 3, 32, 32]) 4:lab unlab q k 
            # student_proj: torch.Size([4*bs, 256]) 
            # student_out: torch.Size([4*bs, 10])
            teacher_out = student_out.detach()

            q_lab_style,q_unlab_style,k_lab_style,k_unlab_style = torch.chunk(style_feature.clone(), 4, dim=0) # [4*bs,128] --> [bs,128] [bs,128] [bs,128] [bs,128]
            lab_style = torch.cat([q_lab_style, k_lab_style], dim = 0) # [2*bs,128]
            unlab_style = torch.cat([q_unlab_style, k_unlab_style], dim = 0) # [2*bs,128]

            # clustering, sup, feature before projection
            sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
            sup_labels = torch.cat([labels[mask_lab] for _ in range(2)], dim=0)
            cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

            # clustering, unsup, feature before projection
            cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
            avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
            me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
            cluster_loss += args.memax_weight * me_max_loss

            # represent learning, unsup, feature after projection
            contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj, args = args)
            unsupcon_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            # representation learning, sup, feature after projection
            student = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
            student = torch.nn.functional.normalize(student, dim=-1)
            supcon_labels = labels[mask_lab]
            supcon_loss = SupConLoss()(student, labels=supcon_labels)

            log_msg = "loss_cls: {:.4f}, loss_cluster: {:.4f}, loss_unsupcon: {:.4f}, loss_supcon: {:.4f}".format(cls_loss.item(), cluster_loss.item(), unsupcon_loss.item(), supcon_loss.item())

            loss = 0
            loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
            loss += (1 - args.sup_weight) * unsupcon_loss + args.sup_weight * supcon_loss

            # proposed loss
            if args.style_remove_function:
                if args.style_remove_function == 'orth':
                    ################ orth -inf~inf  ################
                    loss_Style_remove = abs(torch.sum(style_feature * student_proj, dim=1).mean())

                elif args.style_remove_function == 'cossimi':
                    ################ cos similarity -1~1  ################
                    loss_Style_remove = abs(F.cosine_similarity(style_feature, student_proj, dim=1).mean())

                elif args.style_remove_function == 'coco':
                    ################ COCO -1~1 ################
                    loss_Style_remove = abs(pearson_correlation(style_feature, student_proj).mean())

                else:
                    raise NotImplementedError('style_remove_function not implemented')

                loss += args.style_remove_loss_w * loss_Style_remove
                log_msg += ", loss_Style_remove: {:.4f}".format(loss_Style_remove.item())
            else:
                loss_Style_remove = torch.zeros(()).cuda(args.gpu, non_blocking=True)

            log_msg += ", loss_total: {:.4f}".format(loss.item())
            loggers.debug(log_msg)

            # Train acc
            loss_record.update(loss.item(), labels.size(0))
            optimizer.zero_grad()
            style_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            style_optimizer.step()

        # Step schedule
        exp_lr_scheduler.step()
        exp_lr_scheduler_style.step()
        
        loggers.info('Train Epoch: {} finished. Avg Loss: {:.4f} '.format(epoch + 1, loss_record.avg))

        loggers.info('##### Start to Evaluate at {} epoch #####'.format(epoch))
        loggers.info('Evaluate on '+ args.dataset_series + ' data!')

        loggers.info('Evaluate on labeled classes (test split)')
        test(model, args, loggers, data_loader['test_lab_loader'])

        loggers.info('Evaluate on unlabeled classes (train split)')
        test(model, args, loggers, data_loader['train_unlab_loader'])

        loggers.info('Evaluate on unlabeled classes (test split)')
        acc, nmi, ari = test(model, args, loggers, data_loader['test_unlab_loader']) # only record the acc, nmi, ari of eval_unlab_loader

        if args.use_wandb:
            #======================================================================
            wandb.log({'epoch':epoch, 
            'val_acc_epoch': acc, 
            'val_nmi_epoch': nmi,
            'val_ari_epoch': ari,
            'lr_epoch': optimizer.param_groups[0]['lr'],
            'loss_cls_epoch':cls_loss.item(), 
            'loss_cluster_epoch':cluster_loss.item(),
            'loss_supcon_epoch':supcon_loss.item(),
            'loss_unsupcon_epoch':unsupcon_loss.item(),
            'loss_Style_remove_epoch':loss_Style_remove,
            'loss_total':loss.item()
            })
            #====================================================================== 
        save_dict = {
            'epoch': epoch + 1,
            'lr': optimizer.param_groups[0]['lr'],
            'model': model.state_dict(),
            'style_model': style_model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        supervised_checkpoint_path = os.path.join(args.trained_model_root, 'checkpoint_last_' + args.current_log_withtime + '.pth.tar')
        torch.save(save_dict, supervised_checkpoint_path)
        loggers.info("==> Last checkpoint {} saved to {}.".format(str(epoch + 1), supervised_checkpoint_path))

    if args.use_wandb:
        #======================================================================
        wandb.finish()
        #======================================================================

def test(model, args, loggers, test_loader):
    """Evaluation for the model on the eval or test set."""
    model.eval()
    preds=np.array([])
    targets=np.array([])

    for _, ((im_q,_), label, domain_flag) in enumerate(tqdm(test_loader)):
        im_q = im_q.cuda(args.gpu, non_blocking=True)
        label = label.cuda(args.gpu, non_blocking=True)
        domain_flag = domain_flag.cuda(args.gpu, non_blocking=True)

        _, logits = model(im_q)
        pred = logits.argmax(1).cpu().numpy()
        targets=np.append(targets, label.cpu().numpy())
        preds=np.append(preds, pred)

    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds) 
    loggers.info('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    return acc, nmi, ari

def test_tsne(model, args, loggers, lab_loader, unlab_loader):
    tsne_fig_path = args.test_tsne_fig
    lab_tsne_fig_path = args.test_lab_tsne_fig
    unlab_tsne_fig_path = args.test_unlab_tsne_fig
    tsne_feature_path = args.tsne_root + '/tsne_feature_' + args.current_log_withtime + '.npy'
    label_path = args.tsne_root + '/label_' + args.current_log_withtime + '.npy'

    model.eval()
    targets_org=np.array([])
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    Feature = []
    tsne_result = []

    unlab_dataloader_iterator = iter(unlab_loader)
    for batch_idx, ((q_lab,_), lab_label, _) in enumerate(tqdm(lab_loader)):
        try:
            (q_unlab,_), unlab_label, _ = next(unlab_dataloader_iterator)
        except:
            unlab_dataloader_iterator = iter(unlab_loader)
            (q_unlab,_), unlab_label, _ = next(unlab_dataloader_iterator)

        im_q = torch.cat([q_lab, q_unlab], dim = 0)
        label = torch.cat([lab_label, unlab_label], dim = 0)
        
        im_q = im_q.cuda(args.gpu, non_blocking=True)
        label = label.cuda(args.gpu, non_blocking=True)

        feature, _ = model(im_q)

        targets_org=np.append(targets_org, label.cpu().numpy())
        Feature.extend(feature.detach().cpu().numpy())
    
    tsne_allbatch_result = tsne.fit_transform(np.array(Feature))
    tsne_result_org = np.array(tsne_allbatch_result)
    tsne_result = tsne_result_org.copy().reshape(-1, 2*args.batch_size, *tsne_result_org.shape[1:])
    lab_tsne_result = tsne_result[:, :args.batch_size, :].reshape(-1, *tsne_result.shape[2:])
    unlab_tsne_result = tsne_result[:, args.batch_size:, :].reshape(-1, *tsne_result.shape[2:])

    targets = targets_org.copy().reshape(-1, 2*args.batch_size)
    lab_targets = targets[:, :args.batch_size].reshape(-1)
    unlab_targets = targets[:, args.batch_size:].reshape(-1)
    
    # tsne plot visuallization
    if args.tsne_plot:
        tsne_fig = plot_embedding(tsne_result_org, targets_org.astype(int), 't-SNE embedding of {} on {} feature'.format(args.current_name, args.dataset_series + args.dataset_subclass))
        lab_tsne_fig = plot_embedding(lab_tsne_result, lab_targets.astype(int), 'lab t-SNE embedding of {} on {} feature'.format(args.current_name, args.dataset_subclass))
        unlab_tsne_fig = plot_embedding(unlab_tsne_result, unlab_targets.astype(int), 'unlab t-SNE embedding of {} on {} feature'.format(args.current_name, args.dataset_subclass))
        tsne_fig.savefig(tsne_fig_path)
        lab_tsne_fig.savefig(lab_tsne_fig_path)
        unlab_tsne_fig.savefig(unlab_tsne_fig_path)
        loggers.info('t-SNE visualization saved to {}'.format(tsne_fig_path))
        loggers.info('lab t-SNE visualization saved to {}'.format(lab_tsne_fig_path))
        loggers.info('unlab t-SNE visualization saved to {}'.format(unlab_tsne_fig_path))
    
    # save feature and target
    np.save(tsne_feature_path, tsne_result_org)
    np.save(label_path, targets_org.astype(int))

def reshape_transform(tensor, height=8, width=8):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension, like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result