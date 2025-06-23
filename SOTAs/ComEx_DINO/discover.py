import torch
import wandb
from util.util import LinearWarmupCosineAnnealingLR
from util.sinkhorn_knopp import SinkhornKnopp
from util.util import AverageMeter
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import os
from util.util import cluster_acc
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

from util.tsne_util import plot_embedding
from sklearn.manifold import TSNE

from losses import pearson_correlation

def normed(fea):
    return F.normalize(fea, dim=-1)

def neighbor_targets(args, feats, queue_feat, queue_tar):
    sim = torch.einsum("vhbd, vhqd -> vhbq", feats, queue_feat)  # similarity between online feats and queue feats
    sim = F.softmax(sim / args.softmax_temperature, dim=-1)
    return torch.einsum("vhbq, vhqt -> vhbt", sim, queue_tar)

def sharpen(args, prob):
    sharp_p = prob ** (1. / args.sharp)
    sharp_p /= torch.sum(sharp_p, dim=-1, keepdim=True)
    return sharp_p

def index_swap(args, i1, i2):
    index = torch.arange(args.num_heads)
    index[i1] = i2
    index[i2] = i1
    return index

def cross_entropy_loss(args, logits, targets):
    if args.magic:
        # HACK
        preds = F.log_softmax(logits / args.softmax_temperature, dim=1)
        return -torch.mean(torch.sum(targets * preds, dim=1))
    else:
        preds = F.log_softmax(logits / args.softmax_temperature, dim=-1)
        return -torch.mean(torch.sum(targets * preds, dim=-1))
        
def swapped_prediction(args, logits, targets):
    loss = 0.0
    for view in range(args.num_views):
        for other_view in np.delete(range(args.num_views), view):
            loss += cross_entropy_loss(args, logits[other_view], targets[view])
    return loss / (args.num_views * (args.num_views - 1))

def train(model, style_model, data_loader, args, loggers):
    train_lab_loader = data_loader['train_lab_loader']
    train_unlab_loader = data_loader['train_unlab_loader']
    # init wandb
    if args.use_wandb:
        #======================================================================
        wandb.init(project=args.wandb_project, config = args.__dict__, name = args.current_log, save_code=True, mode = 'offline' if args.use_wandb_offline else 'online')
        model.run_id = wandb.run.id
        #======================================================================
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum_opt, weight_decay=args.weight_decay_opt)
    lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer,warmup_epochs=args.warmup_epochs, total_epochs=args.epochs, eta_min=args.min_lr)
    # Sinkorn-Knopp
    sk = SinkhornKnopp(num_iters=args.num_iters_sk, epsilon=args.epsilon_sk)
    nbc = args.num_base_classes
    nac = args.num_base_classes + args.num_novel_classes

    for epoch in range(args.epochs):
        loggers.info('########## Start discover epoch: ' + str(epoch) + ' ##########')
        loss_record = AverageMeter('Loss', ':.4e')

        model.loss_per_head = torch.zeros_like(model.loss_per_head)
        if args.batch_head_multi_novel:
            model.loss_per_batch_head = torch.zeros_like(model.loss_per_batch_head)

        # normalize prototypes
        model.normalize_prototypes()

        unlab_dataloader_iterator = iter(train_unlab_loader)
        for _, (lab_imgs, lab_label, lab_domain_flag) in enumerate(tqdm(train_lab_loader)):
            try:
                unlab_imgs, unlab_label, unlab_domain_flag = next(unlab_dataloader_iterator)
            except:
                unlab_dataloader_iterator = iter(train_unlab_loader)
                unlab_imgs, unlab_label, unlab_domain_flag = next(unlab_dataloader_iterator)
            
            cuda_lab_imgs = [lab_img.cuda(args.gpu, non_blocking=True) for lab_img in lab_imgs]
            cuda_unlab_imgs = [unlab_img.cuda(args.gpu, non_blocking=True) for unlab_img in unlab_imgs]

            imgs = [torch.cat([lab_img, unlab_img], dim = 0) for lab_img, unlab_img in zip(cuda_lab_imgs, cuda_unlab_imgs)] # [lab_q, unlab_q], [lab_k, unlab_k]
            domain_flag = torch.cat([lab_domain_flag, unlab_domain_flag], dim = 0) # torch.Size([2*bs])
            domain_flag = torch.cat([domain_flag, domain_flag], dim = 0) # torch.Size([4*bs])
            labels = torch.cat([lab_label, unlab_label], dim = 0).cuda(args.gpu, non_blocking=True)
            domain_flag = domain_flag.cuda(args.gpu, non_blocking=True)
            mask_base = labels < args.num_base_classes

            outputs = model(imgs)

            if isinstance(imgs, list): # train stage
                imgs = torch.cat(imgs, dim=0)
            style_feature = style_model(imgs, domain_flag)
            proj_feature = torch.cat([outputs["feats"][0], outputs["feats"][1]], dim=0)

            # proposed loss
            if args.style_remove_function:
                if args.style_remove_function == 'orth':
                    ################ orth -inf~inf  ################
                    loss_Style_remove = abs(torch.sum(style_feature * proj_feature, dim=1).mean())

                elif args.style_remove_function == 'cossimi':
                    ################ cos similarity -1~1  ################
                    loss_Style_remove = abs(F.cosine_similarity(style_feature, proj_feature, dim=1).mean())

                elif args.style_remove_function == 'coco':
                    ################ COCO -1~1 ################
                    loss_Style_remove = abs(pearson_correlation(style_feature, proj_feature).mean())

                else:
                    raise NotImplementedError('style_remove_function not implemented')
        
            else:
                loss_Style_remove = torch.zeros(()).cuda(args.gpu, non_blocking=True)


            outputs["logits_base"] = outputs["logits_base"].unsqueeze(1).expand(-1, args.num_heads, -1, -1)
            logits = torch.cat([outputs["logits_base"], outputs["logits_novel"]], dim=-1)
            logits_over = torch.cat([outputs["logits_base"], outputs["logits_novel_over"]], dim=-1)

            targets = torch.zeros_like(logits)
            targets_over = torch.zeros_like(logits_over)

            if args.batch_head:
                outputs["logits_batch_base"] = outputs["logits_batch_base"].unsqueeze(1).expand(
                    -1, args.num_heads, -1, -1)
                if not args.batch_head_multi_novel:
                    outputs["logits_batch_novel"] = outputs["logits_batch_novel"].unsqueeze(1).expand(
                        -1, args.num_heads, -1, -1)
                    outputs["logits_batch_novel_over"] = outputs["logits_batch_novel_over"].unsqueeze(1).expand(
                        -1, args.num_heads, -1, -1)

                logits_batch_base = outputs["logits_batch_base"][:, :, mask_base, :]
                logits_batch_novel = outputs["logits_batch_novel"][:, :, ~mask_base, :]
                logits_batch_novel_over = outputs["logits_batch_novel_over"][:, :, ~mask_base, :]

                targets_batch_base = torch.zeros_like(logits_batch_base)
                targets_batch_novel = torch.zeros_like(logits_batch_novel)
                targets_batch_novel_over = torch.zeros_like(logits_batch_novel_over)

                logits_batch = torch.zeros_like(outputs["logits_batch_base"])
                logits_batch_over = torch.zeros_like(outputs["logits_batch_novel_over"])
                logits_batch[:, :, mask_base, :] = logits_batch_base
                logits_batch[:, :, ~mask_base, :] = logits_batch_novel
                logits_batch_over[:, :, mask_base, :nac] = logits_batch_base
                logits_batch_over[:, :, ~mask_base, :] = logits_batch_novel_over

                targets_batch = torch.zeros_like(logits_batch)
                targets_batch_over = torch.zeros_like(logits_batch_over)

            # now create targets for base and novel samples
            # targets_base: [base_img_num, base_class_num]
            targets_base = F.one_hot(labels[mask_base], num_classes=args.num_base_classes).float().cuda(args.gpu)

            # generate pseudo-labels with sinkhorn-knopp and fill novel targets
            for v in range(args.num_views):
                for h in range(args.num_heads):
                    targets[v, h, mask_base, :nbc] = targets_base.type_as(targets)
                    targets_over[v, h, mask_base, :nbc] = targets_base.type_as(targets)
                    targets[v, h, ~mask_base, nbc:] = sk(outputs["logits_novel"][v, h, ~mask_base]).type_as(targets) # outputs["logits_novel"]: [num_views(2), num_heads(4), 2*bs(512), novel_class_num(5)] torch.Size([256, 5]) --> torch.Size([256, 5])
                    targets_over[v, h, ~mask_base, nbc:] = sk(
                        outputs["logits_novel_over"][v, h, ~mask_base]).type_as(targets)
                    if args.batch_head:
                        targets_batch_base[v, h, :, :nbc] = targets_base.type_as(targets)
                        targets_batch_novel[v, h, :, nbc:] = sk(
                            outputs["logits_batch_novel"][v, h, ~mask_base, nbc:]).type_as(targets)
                        targets_batch_novel_over[v, h, :, nbc:] = sk(
                            outputs["logits_batch_novel_over"][v, h, ~mask_base, nbc:]).type_as(targets)
                        targets_batch_novel[v, h, :, nbc:] = (
                            targets_batch_novel[v, h, :, nbc:] + targets[v, h, ~mask_base, nbc:]) / 2
                        targets_batch_novel_over[v, h, :, nbc:] = (
                            targets_batch_novel_over[v, h, :, nbc:] + targets_over[v, h, ~mask_base, nbc:]) / 2

                        targets[v, h, ~mask_base, nbc:] = targets_batch_novel[v, h, :, nbc:]
                        targets_over[v, h, ~mask_base, nbc:] = targets_batch_novel_over[v, h, :, nbc:]

                        targets_batch[v, h, mask_base, :] = targets_batch_base[v, h, :, :]
                        targets_batch[v, h, ~mask_base, :] = targets_batch_novel[v, h, :, :]
                        targets_batch_over[v, h, mask_base, :nac] = targets_batch_base[v, h, :, :]
                        targets_batch_over[v, h, ~mask_base, :] = targets_batch_novel_over[v, h, :, :]

            # now queue time
            if args.queue_size:
                if args.batch_head and args.batch_head_multi_novel:
                    model.queuing(
                        normed(outputs["proj_feats_novel"][:, :, ~mask_base, :] +
                                    outputs["proj_feats_batch_novel"][:, :, ~mask_base, :]),
                        normed(outputs["proj_feats_novel_over"][:, :, ~mask_base, :] +
                                    outputs["proj_feats_batch_novel_over"][:, :, ~mask_base, :]),
                        targets[:, :, ~mask_base, nbc:],
                        targets_over[:, :, ~mask_base, nbc:],
                        int((~mask_base).sum())
                    )
                else:
                    model.queuing(
                        outputs["proj_feats_novel"][:, :, ~mask_base, :],
                        outputs["proj_feats_novel_over"][:, :, ~mask_base, :],
                        targets[:, :, ~mask_base, nbc:],
                        targets_over[:, :, ~mask_base, nbc:],
                        int((~mask_base).sum())
                    )

                if -1 not in model.queue_targets:  # make sure the queue is full
                    if args.batch_head and args.batch_head_multi_novel:
                        neighbor_tar = neighbor_targets(args, 
                            normed(outputs["proj_feats_novel"][:, :, ~mask_base, :] +
                                        outputs["proj_feats_batch_novel"][:, :, ~mask_base, :]),
                            model.queue_feats.clone().detach(),
                            model.queue_targets.clone().detach()
                        )
                        neighbor_tar_over = neighbor_targets(args, 
                            normed(outputs["proj_feats_novel_over"][:, :, ~mask_base, :] +
                                        outputs["proj_feats_batch_novel_over"][:, :, ~mask_base, :]),
                            model.queue_feats_over.clone().detach(),
                            model.queue_targets_over.clone().detach()
                        )
                    else:
                        neighbor_tar = neighbor_targets(args, 
                            outputs["proj_feats_novel"][:, :, ~mask_base, :],
                            model.queue_feats.clone().detach(),
                            model.queue_targets.clone().detach()
                        )
                        neighbor_tar_over = neighbor_targets(args, 
                            outputs["proj_feats_novel_over"][:, :, ~mask_base, :],
                            model.queue_feats_over.clone().detach(),
                            model.queue_targets_over.clone().detach()
                        )

                    targets[:, :, ~mask_base, nbc:] = sharpen(args, 
                        args.queue_alpha * targets[:, :, ~mask_base, nbc:].type_as(targets) +
                        (1 - args.queue_alpha) * neighbor_tar.type_as(targets)
                    ).type_as(targets)
                    targets_over[:, :, ~mask_base, nbc:] = sharpen(args, 
                        args.queue_alpha * targets_over[:, :, ~mask_base, nbc:].type_as(targets) +
                        (1 - args.queue_alpha) * neighbor_tar_over.type_as(targets)
                    ).type_as(targets)

                    if args.batch_head:
                        targets_batch_novel[:, :, :, nbc:] = targets[:, :, ~mask_base, nbc:]
                        targets_batch_novel_over[:, :, :, nbc:] = targets_over[:, :, ~mask_base, nbc:]
                        targets_batch[:, :, ~mask_base, :] = targets_batch_novel
                        targets_batch_over[:, :, ~mask_base, :] = targets_batch_novel_over

            # compute losses
            loss_cluster = swapped_prediction(args, logits, targets)
            loss_overcluster = swapped_prediction(args, logits_over, targets_over)
            if args.batch_head:
                loss_batch_cluster = swapped_prediction(args, logits_batch, targets_batch)
                loss_batch_overcluster = swapped_prediction(args, logits_batch_over, targets_batch_over)
                if args.batch_head_reg:
                    loss_batch_base_reg = torch.norm(logits_batch_base[:, :, :, nbc:], dim=None)
                    loss_batch_novel_reg = torch.norm(logits_batch_novel[:, :, :, :nbc], dim=None)
                    loss_batch_novel_over_reg = torch.norm(logits_batch_novel_over[:, :, :, :nbc], dim=None)

            # update best head tracker, note that head with the smallest loss is not always the best
            model.loss_per_head += loss_cluster.clone().detach()
            if args.batch_head_multi_novel:
                model.loss_per_batch_head += loss_batch_cluster.clone().detach()

            # total loss and log
            loss_cluster = loss_cluster.mean()
            loss_overcluster = loss_overcluster.mean()
            # loss = (loss_cluster + loss_overcluster) / 2  
            loss = (loss_cluster + loss_overcluster) / 2  + args.style_remove_loss_w * loss_Style_remove

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_record.update(loss.item(), targets.size(0))

        loggers.debug('loss_total: {:.4f}'.format(loss.item()))

        if args.batch_head:
            loss += (loss_batch_cluster + loss_batch_overcluster) / 2
            if args.use_wandb:
                #======================================================================
                wandb.log({
                    "loss_Style_remove": loss_Style_remove.item(),
                    "loss_batch_cluster": loss_batch_cluster.mean(),
                    "loss_batch_overcluster": loss_batch_overcluster.mean()
                })
                #======================================================================

            if args.batch_head_reg:
                loss += args.batch_head_reg * (
                            loss_batch_base_reg + loss_batch_novel_reg + loss_batch_novel_over_reg) / 3
                if args.use_wandb:
                    #======================================================================
                    wandb.log({
                        "loss_batch_base_reg": loss_batch_base_reg.mean(),
                        "loss_batch_novel_reg": loss_batch_novel_reg.mean(),
                        "loss_batch_novel_over_reg": loss_batch_novel_over_reg.mean()
                    })
                    #======================================================================
        
        lr_scheduler.step(epoch)

        # log record
        loggers.info('Train Epoch {} finished. Avg Loss: {:.4f}'.format(epoch, loss_record.avg))

        loggers.info('##### Start to Evaluate at {} epoch #####'.format(epoch)) # {0: "unlab/train", 1: "unlab/test", 2: "lab/test"}
        loggers.info('Evaluate on '+ args.dataset_series + '_' + args.dataset_subclass + ' data!')

        loggers.info('Evaluate on labeled classes (test split)')
        Test_results = test(model, args, loggers, 'base/test', data_loader['test_lab_loader'])

        lab_ts_acc, lab_ts_nmi, lab_ts_ari = Test_results['acc'], Test_results['nmi'], Test_results['ari']
        lab_ts_acc_inc, lab_ts_nmi_inc, lab_ts_ari_inc = Test_results['acc_inc'], Test_results['nmi_inc'], Test_results['ari_inc']

        loggers.info('Evaluate on unlabeled classes (train split)')
        Test_results = test(model, args, loggers, 'novel/train', data_loader['train_unlab_loader2'])

        mean_unlab_tr_acc, mean_unlab_tr_nmi, mean_unlab_tr_ari = Test_results['mean_acc'], Test_results['mean_nmi'], Test_results['mean_ari']
        mean_unlab_tr_acc_inc, mean_unlab_tr_nmi_inc, mean_unlab_tr_ari_inc = Test_results['mean_acc_inc'], Test_results['mean_nmi_inc'], Test_results['mean_ari_inc']
        best_unlab_tr_acc, best_unlab_tr_nmi, best_unlab_tr_ari = Test_results['best_acc'], Test_results['best_nmi'], Test_results['best_ari']
        best_unlab_tr_acc_inc, best_unlab_tr_nmi_inc, best_unlab_tr_ari_inc = Test_results['best_acc_inc'], Test_results['best_nmi_inc'], Test_results['best_ari_inc']

        loggers.info('Evaluate on unlabeled classes (test split)')
        Test_results = test(model, args, loggers, 'novel/test', data_loader['test_unlab_loader'])
        
        mean_unlab_ts_acc, mean_unlab_ts_nmi, mean_unlab_ts_ari = Test_results['mean_acc'], Test_results['mean_nmi'], Test_results['mean_ari']
        mean_unlab_ts_acc_inc, mean_unlab_ts_nmi_inc, mean_unlab_ts_ari_inc = Test_results['mean_acc_inc'], Test_results['mean_nmi_inc'], Test_results['mean_ari_inc']
        best_unlab_ts_acc, best_unlab_ts_nmi, best_unlab_ts_ari = Test_results['best_acc'], Test_results['best_nmi'], Test_results['best_ari']
        best_unlab_ts_acc_inc, best_unlab_ts_nmi_inc, best_unlab_ts_ari_inc = Test_results['best_acc_inc'], Test_results['best_nmi_inc'], Test_results['best_ari_inc']
        
        if args.use_wandb:
            #======================================================================
            wandb.log({
                "loss": loss.detach(),
                "loss_cluster": loss_cluster.mean(),
                "loss_overcluster": loss_overcluster.mean(),
                "lr": optimizer.param_groups[0]["lr"],
            })
            #======================================================================
            #======================================================================
            wandb.log({
                "val/lab_ts_acc": lab_ts_acc,
                "val/lab_ts_nmi": lab_ts_nmi,
                "val/lab_ts_ari": lab_ts_ari,
                "val/lab_ts_acc_inc": lab_ts_acc_inc,
                "val/lab_ts_nmi_inc": lab_ts_nmi_inc,
                "val/lab_ts_ari_inc": lab_ts_ari_inc,
                "val/mean_unlab_tr_acc": mean_unlab_tr_acc,
                "val/mean_unlab_tr_nmi": mean_unlab_tr_nmi,
                "val/mean_unlab_tr_ari": mean_unlab_tr_ari,
                "val/mean_unlab_tr_acc_inc": mean_unlab_tr_acc_inc,
                "val/mean_unlab_tr_nmi_inc": mean_unlab_tr_nmi_inc,
                "val/mean_unlab_tr_ari_inc": mean_unlab_tr_ari_inc,
                "val/best_unlab_tr_acc": best_unlab_tr_acc,
                "val/best_unlab_tr_nmi": best_unlab_tr_nmi,
                "val/best_unlab_tr_ari": best_unlab_tr_ari,
                "val/best_unlab_tr_acc_inc": best_unlab_tr_acc_inc,
                "val/best_unlab_tr_nmi_inc": best_unlab_tr_nmi_inc,
                "val/best_unlab_tr_ari_inc": best_unlab_tr_ari_inc,
                "val/mean_unlab_ts_acc": mean_unlab_ts_acc,
                "val/mean_unlab_ts_nmi": mean_unlab_ts_nmi,
                "val/mean_unlab_ts_ari": mean_unlab_ts_ari,
                "val/mean_unlab_ts_acc_inc": mean_unlab_ts_acc_inc,
                "val/mean_unlab_ts_nmi_inc": mean_unlab_ts_nmi_inc,
                "val/mean_unlab_ts_ari_inc": mean_unlab_ts_ari_inc,
                "val/best_unlab_ts_acc": best_unlab_ts_acc,
                "val/best_unlab_ts_nmi": best_unlab_ts_nmi,
                "val/best_unlab_ts_ari": best_unlab_ts_ari,
                "val/best_unlab_ts_acc_inc": best_unlab_ts_acc_inc,
                "val/best_unlab_ts_nmi_inc": best_unlab_ts_nmi_inc,
                "val/best_unlab_ts_ari_inc": best_unlab_ts_ari_inc
            })
            #======================================================================  

        # save the model in now epoch as the last checkpoint
        save_dict = {
                    'epoch': epoch,
                    'lr': optimizer.param_groups[0]['lr'],
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
        discover_checkpoint_path = os.path.join(args.trained_model_root, 'checkpoint_last_' + args.current_log_withtime + '.pth.tar')
        torch.save(save_dict, discover_checkpoint_path)
        loggers.info("==> Last checkpoint {} saved to {}.".format(str(epoch), discover_checkpoint_path))

    if args.use_wandb:
        #======================================================================
        wandb.finish()
        #======================================================================

def test(model, args, loggers, tag, *test_loader):
    Test_results = {}
    """Evaluation for the model on the eval."""
    model.eval()
    preds = []
    preds_inc = []
    targets = []
    nbc = args.num_base_classes
    if len(test_loader) == 1:
        novel_loader = test_loader[0]
        for batch_idx, (img, label, domain_flag) in enumerate(tqdm(novel_loader)):
            img = img.cuda(args.gpu, non_blocking=True)
            label = label.cuda(args.gpu, non_blocking=True)
            domain_flag = domain_flag.cuda(args.gpu, non_blocking=True)

            # forward
            outputs = model(img)

            if "novel" in tag:
                logit = outputs["logits_novel"] # torch.Size([4, bs, 5])
                if args.batch_head:
                    if args.batch_head_multi_novel:
                        head_swapped = index_swap(args, torch.argmin(model.loss_per_head),
                                                    torch.argmin(model.loss_per_batch_head))
                        logit += args.alpha * outputs["logits_batch_novel"][head_swapped][:, :, nbc:]
                    else:
                        logit += args.alpha * outputs["logits_batch_novel"].unsqueeze(0)[:, :, nbc:]
                logit_inc = torch.cat(
                    [outputs["logits_base"].unsqueeze(0).expand(args.num_heads, -1, -1), 
                    outputs["logits_novel"]],dim=-1) # torch.Size([4, bs, 10])
                if args.batch_head:
                    logit_inc += args.alpha * outputs["logits_batch_base"].unsqueeze(0).expand(
                        args.num_heads, -1, -1)
                    if args.batch_head_multi_novel:
                        head_swapped = index_swap(args, torch.argmin(model.loss_per_head),
                                                    torch.argmin(model.loss_per_batch_head))
                        logit_inc += args.alpha * outputs["logits_batch_novel"][head_swapped]
                    else:
                        logit_inc += args.alpha * outputs["logits_batch_novel"].unsqueeze(0).expand(
                            args.num_heads, -1, -1)
            else:  #  use supervised classifier
                logit = outputs["logits_base"] # torch.Size([bs, 5])
                if args.batch_head:
                    logit += args.alpha * outputs["logits_batch_base"][:, :nbc]
                best_head = torch.argmin(model.loss_per_head)
                logit_inc = torch.cat([outputs["logits_base"], 
                                       outputs["logits_novel"][best_head]], dim=-1) # torch.Size([bs, 10])
                if args.batch_head:
                    logit_inc += args.alpha * outputs["logits_batch_base"]
                    if args.batch_head_multi_novel:
                        best_batch_head = torch.argmin(model.loss_per_batch_head)
                        logit_inc += args.alpha * outputs["logits_batch_novel"][best_batch_head]
                    else:
                        logit_inc += args.alpha * outputs["logits_batch_novel"]

            pred = logit.max(dim=-1)[1] # lab: torch.Size([bs]) unlab: torch.Size([4, bs])
            pred_inc = logit_inc.max(dim=-1)[1] # lab: torch.Size([bs]) unlab: torch.Size([4, bs])

            preds.append(pred) # lab: torch.Size([bs]) unlab: torch.Size([4, bs])
            preds_inc.append(pred_inc)
            targets.append(label)
            
        preds = torch.cat(preds, dim=-1).cpu().numpy() # lab: torch.Size([bs * num_bs_per_epoch]) unlab: torch.Size([4, bs * num_bs_per_epoch])
        preds_inc = torch.cat(preds_inc, dim=-1).cpu().numpy()
        targets = torch.cat(targets, dim=-1).cpu().numpy()

        if "novel" in tag: # unlab: torch.Size([4, bs * num_bs_per_epoch])
            ACC = []
            NMI = []
            ARI = []
            ACC_INC = []
            NMI_INC = []
            ARI_INC = []
            for head in range(args.num_heads):
                ACC.append(cluster_acc(targets, preds[head]))
                NMI.append(nmi_score(targets, preds[head]))
                ARI.append(ari_score(targets, preds[head]))
                ACC_INC.append(cluster_acc(targets, preds_inc[head]))
                NMI_INC.append(nmi_score(targets, preds_inc[head]))
                ARI_INC.append(ari_score(targets, preds_inc[head]))
            ACC = np.array(ACC)
            NMI = np.array(NMI)
            ARI = np.array(ARI)
            ACC_INC = np.array(ACC_INC)
            NMI_INC = np.array(NMI_INC)
            ARI_INC = np.array(ARI_INC)
            Test_results['mean_acc'], Test_results['mean_nmi'], Test_results['mean_ari'] = np.mean(ACC), np.mean(NMI), np.mean(ARI)
            Test_results['mean_acc_inc'], Test_results['mean_nmi_inc'], Test_results['mean_ari_inc'] = np.mean(ACC_INC), np.mean(NMI_INC), np.mean(ARI_INC)
            min_idx = torch.argmin(model.loss_per_head)
            Test_results['best_acc'], Test_results['best_nmi'], Test_results['best_ari'] = ACC[min_idx], NMI[min_idx], ARI[min_idx]
            Test_results['best_acc_inc'], Test_results['best_nmi_inc'], Test_results['best_ari_inc'] = ACC_INC[min_idx], NMI_INC[min_idx], ARI_INC[min_idx]
        else: # lab: torch.Size([bs * num_bs_per_epoch])
            Test_results['acc'], Test_results['nmi'], Test_results['ari'] = cluster_acc(targets, preds), nmi_score(targets, preds), ari_score(targets, preds)
            Test_results['acc_inc'], Test_results['nmi_inc'], Test_results['ari_inc'] = cluster_acc(targets, preds_inc), nmi_score(targets, preds_inc), ari_score(targets, preds_inc)

    
    # HACK useless now
    elif len(test_loader) == 2: 
        base_loader = test_loader[0]
        novel_loader = test_loader[1]
        novel_dataloader_iterator = iter(novel_loader)
        for batch_idx, (base_img, base_label, base_domain_flag) in enumerate(tqdm(base_loader)):
            try:
                novel_img, novel_label, novel_domain_flag = next(novel_dataloader_iterator)
            except:
                novel_dataloader_iterator = iter(novel_loader)
                novel_img, novel_label, novel_domain_flag = next(novel_dataloader_iterator)

            imgs = torch.cat([base_img, novel_img], dim = 0)
            imgs = imgs.cuda(args.gpu, non_blocking=True)

            domain_flag = torch.cat([base_domain_flag, novel_domain_flag], dim = 0) # torch.Size([2*bs])
            labels = torch.cat([base_label, novel_label], dim = 0)
            mask_base = labels < args.num_base_classes

            outputs = model(imgs)
            logit = outputs["logits_lab"]
            best_head = torch.argmin(model.loss_per_head)
            logit_inc = torch.cat([outputs["logits_lab"], outputs["logits_unlab"][best_head]], dim=-1)
            pred = logit.max(dim=-1)[1]
            pred_inc = logit_inc.max(dim=-1)[1]

            targets=np.append(targets, labels.cpu().numpy())
            preds=np.append(preds, pred.cpu().numpy())
            preds_inc=np.append(preds_inc, pred_inc.cpu().numpy())

        Test_results['acc'], Test_results['nmi'], Test_results['ari'] = cluster_acc(targets, preds), nmi_score(targets, preds), ari_score(targets, preds)
        Test_results['acc_inc'], Test_results['nmi_inc'], Test_results['ari_inc'] = cluster_acc(targets, preds_inc), nmi_score(targets, preds_inc), ari_score(targets, preds_inc) 
    
    else:
        raise NotImplementedError
    
    formatted_results = {k: "{:.4f}".format(v) if isinstance(v, float) else v for k, v in Test_results.items()}
    
    str_results = str(formatted_results)
    str_results = str_results.replace("'", "").replace(":", "").replace(",", "")
    str_results = str_results[1:-1]

    loggers.info(str_results)
    
    return Test_results

def test_tsne(model, args, *test_loader):
    tsne_fig_path = args.test_tsne_fig
    feature_path = args.tsne_root + '/feature_' + args.current_log_withtime + '.npy'
    label_path = args.tsne_root + '/label_' + args.current_log_withtime + '.npy'
    model.eval()
    targets=np.array([])
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    Feature = []
    tsne_result = []

    base_loader = test_loader[0]
    novel_loader = test_loader[1]
    novel_dataloader_iterator = iter(novel_loader)
    for batch_idx, (base_img, base_label, base_domain_flag) in enumerate(tqdm(base_loader)):
        try:
            novel_img, novel_label, novel_domain_flag = next(novel_dataloader_iterator)
        except:
            novel_dataloader_iterator = iter(novel_loader)
            novel_img, novel_label, novel_domain_flag = next(novel_dataloader_iterator)

        imgs = torch.cat([base_img, novel_img], dim = 0)
        imgs = imgs.cuda(args.gpu, non_blocking=True)
        labels = torch.cat([base_label, novel_label], dim = 0)

        outputs = model(imgs)
        feature = outputs["feats"]
        targets=np.append(targets, labels.cpu().numpy())
        Feature.extend(feature.detach().cpu().numpy())
    
    if args.save_tsne_feature:
        # save feature and target
        np.save(feature_path, np.array(Feature))
        np.save(label_path, targets.astype(int))
    
    if args.use_tsne_visual:
        tsne_allbatch_result = tsne.fit_transform(np.array(Feature))
        tsne_result = np.array(tsne_allbatch_result)
        # tsne plot
        tsne_fig = plot_embedding(tsne_result, targets.astype(int), 't-SNE embedding of {} on {} feature'.format(args.current_name, args.dataset_series + args.dataset_subclass))
        tsne_fig.savefig(tsne_fig_path)
    