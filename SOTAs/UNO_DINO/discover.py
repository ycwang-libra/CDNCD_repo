import torch
import wandb
import torch.nn.functional as F
import numpy as np
import os
from util.util import LinearWarmupCosineAnnealingLR, cluster_acc
from util.sinkhorn_knopp import SinkhornKnopp
from util.util import AverageMeter
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from tqdm import tqdm
from util.tsne_util import plot_embedding
from sklearn.manifold import TSNE
from losses import pearson_correlation

def cross_entropy_loss(args, logits, targets):
    preds = F.log_softmax(logits / args.softmax_temperature, dim=-1)
    return torch.mean(-torch.sum(targets * preds, dim=-1), dim=-1)

def swapped_prediction(args, logits, targets):
    loss = 0
    for view in range(args.num_large_crops):
        for other_view in np.delete(range(args.num_crops), view):
            loss += cross_entropy_loss(args, logits[other_view], targets[view])
    return loss / (args.num_large_crops * (args.num_crops - 1))

def train(model, style_model, data_loader, args, loggers):
    train_lab_loader = data_loader['train_lab_loader']
    train_unlab_loader = data_loader['train_unlab_loader']
    # init wandb
    if args.use_wandb:
        #======================================================================
        wandb.init(project=args.wandb_project, config = args.__dict__, name = args.current_log, save_code=True, mode = 'offline' if args.use_wandb_offline else 'online')
        model.run_id = wandb.run.id
        #======================================================================
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer,warmup_epochs = args.warmup_epochs, total_epochs=args.epochs, eta_min=args.min_lr)
    # Sinkorn-Knopp
    sk = SinkhornKnopp(num_iters=args.num_iters_sk, epsilon=args.epsilon_sk)
    nlc = args.num_labeled_classes

    for epoch in range(args.epochs):
        loggers.info('########## Start discover epoch: ' + str(epoch) + ' ##########')
        loss_record = AverageMeter('Loss', ':.4e')
        
        # UNO pytorchlightning framework begain each epoch, set loss_per_head to zero
        model.loss_per_head = torch.zeros_like(model.loss_per_head)

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

            imgs = [torch.cat([lab_img, unlab_img], dim = 0) for lab_img, unlab_img in zip(cuda_lab_imgs, cuda_unlab_imgs)]
            domain_flag = torch.cat([lab_domain_flag, unlab_domain_flag], dim = 0) # torch.Size([2*bs])
            domain_flag = torch.cat([domain_flag, domain_flag], dim = 0) # torch.Size([4*bs])
            labels = torch.cat([lab_label, unlab_label], dim = 0).cuda(args.gpu, non_blocking=True)
            domain_flag = domain_flag.cuda(args.gpu, non_blocking=True)
            mask_lab = labels < args.num_labeled_classes

            outputs = model(imgs)
            if isinstance(imgs, list): # train stage
                imgs = torch.cat([imgs[0],imgs[1]], dim=0) # UNO

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


            outputs["logits_lab"] = (outputs["logits_lab"].unsqueeze(1).expand(-1, args.num_heads, -1, -1))
            logits = torch.cat([outputs["logits_lab"], outputs["logits_unlab"]], dim=-1)
            logits_over = torch.cat([outputs["logits_lab"], outputs["logits_unlab_over"]], dim=-1)
            # create targets
            targets_lab = (F.one_hot(labels[mask_lab], num_classes=args.num_labeled_classes).float().cuda(args.gpu))
            targets = torch.zeros_like(logits)
            targets_over = torch.zeros_like(logits_over)

            # generate pseudo-labels with sinkhorn-knopp and fill unlab targets
            for v in range(args.num_large_crops):
                for h in range(args.num_heads):
                    targets[v, h, mask_lab, :nlc] = targets_lab.type_as(targets)
                    targets_over[v, h, mask_lab, :nlc] = targets_lab.type_as(targets)
                    targets[v, h, ~mask_lab, nlc:] = sk(outputs["logits_unlab"][v, h, ~mask_lab]).type_as(targets)
                    targets_over[v, h, ~mask_lab, nlc:] = sk(outputs["logits_unlab_over"][v, h, ~mask_lab]).type_as(targets)
            # compute swapped prediction loss
            loss_cluster = swapped_prediction(args, logits, targets)
            loss_overcluster = swapped_prediction(args, logits_over, targets_over)

             # update best head tracker
            model.loss_per_head += loss_cluster.clone().detach()

            # total loss
            loss_cluster = loss_cluster.mean()
            loss_overcluster = loss_overcluster.mean()
            loss = (loss_cluster + loss_overcluster) / 2 + args.style_remove_loss_w * loss_Style_remove
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_record.update(loss.item(), targets.size(0))
        
        lr_scheduler.step(epoch)
        
        if args.use_wandb:
            #======================================================================
            wandb.log({
                "loss": loss.detach(),
                "loss_Style_remove": loss_Style_remove.item(),
                "loss_cluster": loss_cluster.mean(),
                "loss_overcluster": loss_overcluster.mean(),
                "lr": optimizer.param_groups[0]["lr"],
            })
            #======================================================================

        loggers.info('Train Epoch {} finished. Avg Loss: {:.4f}'.format(epoch, loss_record.avg))

        loggers.info('##### Start to Evaluate at {} epoch #####'.format(epoch))
        loggers.info('Evaluate on '+ args.dataset_series + '_' + args.dataset_subclass + ' data!')

        loggers.info('Evaluate on labeled classes (test split)')
        Test_results = test(model, args, loggers, 'lab/test', data_loader['test_lab_loader'])
        lab_ts_acc, lab_ts_nmi, lab_ts_ari = Test_results['acc'], Test_results['nmi'], Test_results['ari']
        lab_ts_acc_inc, lab_ts_nmi_inc, lab_ts_ari_inc = Test_results['acc_inc'], Test_results['nmi_inc'], Test_results['ari_inc']

        loggers.info('Evaluate on unlabeled classes (train split)')
        Test_results = test(model, args, loggers, 'unlab/train', data_loader['train_unlab_loader2'])
        mean_unlab_tr_acc, mean_unlab_tr_nmi, mean_unlab_tr_ari = Test_results['mean_acc'], Test_results['mean_nmi'], Test_results['mean_ari']
        mean_unlab_tr_acc_inc, mean_unlab_tr_nmi_inc, mean_unlab_tr_ari_inc = Test_results['mean_acc_inc'], Test_results['mean_nmi_inc'], Test_results['mean_ari_inc']
        best_unlab_tr_acc, best_unlab_tr_nmi, best_unlab_tr_ari = Test_results['best_acc'], Test_results['best_nmi'], Test_results['best_ari']
        best_unlab_tr_acc_inc, best_unlab_tr_nmi_inc, best_unlab_tr_ari_inc = Test_results['best_acc_inc'], Test_results['best_nmi_inc'], Test_results['best_ari_inc']

        loggers.info('Evaluate on unlabeled classes (test split)')
        Test_results = test(model, args, loggers, 'unlab/test', data_loader['test_unlab_loader'])
        mean_unlab_ts_acc, mean_unlab_ts_nmi, mean_unlab_ts_ari = Test_results['mean_acc'], Test_results['mean_nmi'], Test_results['mean_ari']
        mean_unlab_ts_acc_inc, mean_unlab_ts_nmi_inc, mean_unlab_ts_ari_inc = Test_results['mean_acc_inc'], Test_results['mean_nmi_inc'], Test_results['mean_ari_inc']
        best_unlab_ts_acc, best_unlab_ts_nmi, best_unlab_ts_ari = Test_results['best_acc'], Test_results['best_nmi'], Test_results['best_ari']
        best_unlab_ts_acc_inc, best_unlab_ts_nmi_inc, best_unlab_ts_ari_inc = Test_results['best_acc_inc'], Test_results['best_nmi_inc'], Test_results['best_ari_inc']
        
        if args.use_wandb:
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
    if len(test_loader) == 1:
        unlab_loader = test_loader[0]
        for batch_idx, (img, label, domain_flag) in enumerate(tqdm(unlab_loader)):
            img = img.cuda(args.gpu, non_blocking=True)
            label = label.cuda(args.gpu, non_blocking=True)
            domain_flag = domain_flag.cuda(args.gpu, non_blocking=True)

            # forward
            outputs = model(img)

            if "unlab" in tag:
                logit = outputs["logits_unlab"] # torch.Size([4, bs, 5])
                logit_inc = torch.cat(
                    [outputs["logits_lab"].unsqueeze(0).expand(args.num_heads, -1, -1), 
                    outputs["logits_unlab"]],dim=-1) # torch.Size([4, bs, 10])
            else:  #  use supervised classifier
                logit = outputs["logits_lab"] # torch.Size([bs, 5])
                best_head = torch.argmin(model.loss_per_head)
                logit_inc = torch.cat([outputs["logits_lab"], 
                                       outputs["logits_unlab"][best_head]], dim=-1) # torch.Size([bs, 10])

            pred = logit.max(dim=-1)[1] # lab: torch.Size([bs]) unlab: torch.Size([4, bs])
            pred_inc = logit_inc.max(dim=-1)[1] # lab: torch.Size([bs]) unlab: torch.Size([4, bs])

            preds.append(pred) # lab: torch.Size([bs]) unlab: torch.Size([4, bs])
            preds_inc.append(pred_inc)
            targets.append(label)
            
        preds = torch.cat(preds, dim=-1).cpu().numpy() # lab: torch.Size([bs * num_bs_per_epoch]) unlab: torch.Size([4, bs * num_bs_per_epoch])
        preds_inc = torch.cat(preds_inc, dim=-1).cpu().numpy()
        targets = torch.cat(targets, dim=-1).cpu().numpy()

        if "unlab" in tag: # unlab: torch.Size([4, bs * num_bs_per_epoch])
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
        lab_loader = test_loader[0]
        unlab_loader = test_loader[1]
        unlab_dataloader_iterator = iter(unlab_loader)
        for batch_idx, (lab_img, lab_label, lab_domain_flag) in enumerate(tqdm(lab_loader)): 
            try:
                unlab_img, unlab_label, unlab_domain_flag = next(unlab_dataloader_iterator)
            except:
                unlab_dataloader_iterator = iter(unlab_loader)
                unlab_img, unlab_label, unlab_domain_flag = next(unlab_dataloader_iterator)

            imgs = torch.cat([lab_img, unlab_img], dim = 0)
            imgs = imgs.cuda(args.gpu, non_blocking=True)

            domain_flag = torch.cat([lab_domain_flag, unlab_domain_flag], dim = 0) # torch.Size([2*bs])
            labels = torch.cat([lab_label, unlab_label], dim = 0)
            mask_lab = labels < args.num_labeled_classes

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

    lab_loader = test_loader[0]
    unlab_loader = test_loader[1]
    unlab_dataloader_iterator = iter(unlab_loader)
    for batch_idx, (lab_img, lab_label, lab_domain_flag) in enumerate(tqdm(lab_loader)): 
        try:
            unlab_img, unlab_label, unlab_domain_flag = next(unlab_dataloader_iterator)
        except:
            unlab_dataloader_iterator = iter(unlab_loader)
            unlab_img, unlab_label, unlab_domain_flag = next(unlab_dataloader_iterator)

        imgs = torch.cat([lab_img, unlab_img], dim = 0)
        imgs = imgs.cuda(args.gpu, non_blocking=True)
        labels = torch.cat([lab_label, unlab_label], dim = 0)

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

    