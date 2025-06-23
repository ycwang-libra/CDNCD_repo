import torch
import wandb
import torch.nn.functional as F
import numpy as np
import os
from util.util import LinearWarmupCosineAnnealingLR, cluster_acc
from util.util import AverageMeter
from tqdm import tqdm

def train(model, data_loader, args, loggers):
    train_lab_loader = data_loader['train_lab_loader']
    # init wandb
    if args.use_wandb:
        #======================================================================
        wandb.init(project=args.wandb_project, config = args.__dict__, name = args.current_log, save_code=True, mode = 'offline' if args.use_wandb_offline else 'online')
        model.run_id = wandb.run.id
        #======================================================================
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer,warmup_epochs = args.warmup_epochs, total_epochs=args.epochs, eta_min=args.min_lr)

    for epoch in range(args.epochs):
        loggers.info('########## Start supervised pretraining epoch: ' + str(epoch) + ' ##########')
        loss_record = AverageMeter('Loss', ':.4e')
        
        # normalize prototypes
        model.normalize_prototypes()
        
        for _, (labeled_imgs, labeled_label, _) in enumerate(tqdm(train_lab_loader)):

            imgs = [labeled_img.cuda(args.gpu, non_blocking=True) for labeled_img in labeled_imgs] # two transform produce two imgs to form a list
            label = labeled_label.cuda(args.gpu, non_blocking=True) # only source data has label
            
            # forward
            outputs = model(imgs)
            # supervised loss
            loss_supervised = torch.stack([F.cross_entropy(o / args.softmax_temperature, label) for o in outputs["logits_lab"]]).mean()

            optimizer.zero_grad()
            loss_supervised.backward()
            optimizer.step()

            loss_record.update(loss_supervised.item(), label.size(0))
        
        lr_scheduler.step(epoch)
        loggers.info('Train Epoch {} finished. Avg Loss: {:.4f}'.format(epoch, loss_record.avg))

        loggers.info('##### Start to Evaluate at {} epoch #####'.format(epoch))
        loggers.info('Evaluate on '+ args.dataset_series + '_' + args.dataset_subclass + ' labeled data! (test split)')
        acc = test(model, args, loggers, data_loader['test_lab_loader']) # no need changestyle in target
        loggers.info("Acc on labeled classes (test split): {:.4f}".format(acc))

        if args.use_wandb:
            #======================================================================
            wandb.log({'epoch':epoch, 
            'loss_supervised': loss_record.avg, 
            'lr': optimizer.param_groups[0]['lr']
            })
            #====================================================================== 

        # save the model in now epoch as the last checkpoint
        save_dict = {
                    'epoch': epoch,
                    'lr': optimizer.param_groups[0]['lr'],
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
        pretrain_checkpoint_path = os.path.join(args.trained_model_root, 'checkpoint_last_' + args.current_log_withtime + '.pth.tar')
        torch.save(save_dict, pretrain_checkpoint_path)
        loggers.info("==> Last checkpoint {} saved to {}.".format(str(epoch), pretrain_checkpoint_path))
    
    if args.use_wandb:
        #======================================================================
        wandb.finish()
        #======================================================================
        
def test(model, args, loggers, *test_loader):
    """Evaluation for the model on the eval."""
    model.eval()
    targets = np.array([])
    preds = np.array([])
    losses_supervised = np.array([])
    if len(test_loader) == 1:
        unlabeled_loader = test_loader[0]
        for batch_idx, (q_unlabeled, label, domain_flag) in enumerate(tqdm(unlabeled_loader)):
            im_q = q_unlabeled.cuda(args.gpu, non_blocking=True)
            label = label.cuda(args.gpu, non_blocking=True)
            domain_flag = domain_flag.cuda(args.gpu, non_blocking=True)

            # forward
            logit = model(im_q)["logits_lab"]
            _, pred = logit.max(dim=-1)
            
            # calculate batch loss and acc
            loss_supervised = F.cross_entropy(logit, label)
            losses_supervised = np.append(losses_supervised, loss_supervised.detach().cpu().numpy())

            targets=np.append(targets, label.cpu().numpy())
            preds=np.append(preds, pred.cpu().numpy())
        
        acc = cluster_acc(targets.astype(int), preds.astype(int))

        if args.use_wandb:
            #======================================================================
            wandb.log({
            'val/loss_supervised': losses_supervised.mean(), 
            'val/acc': acc
            })
            #======================================================================   
    
    elif len(test_loader) == 2: 
        labeled_loader = test_loader[0]
        unlabeled_loader = test_loader[1]
        unlabeled_dataloader_iterator = iter(unlabeled_loader)
        for batch_idx, (q_labeled, labeled_label, labeled_domain_flag) in enumerate(tqdm(labeled_loader)):
            try:
                q_unlabeled, unlabeled_label, unlabeled_domain_flag = next(unlabeled_dataloader_iterator)
            except:
                unlabeled_dataloader_iterator = iter(unlabeled_loader)
                q_unlabeled, unlabeled_label, unlabeled_domain_flag = next(unlabeled_dataloader_iterator)

            im_q = torch.cat([q_labeled, q_unlabeled], dim = 0)
            domain_flag = torch.cat([labeled_domain_flag, unlabeled_domain_flag], dim = 0)
            
            im_q = im_q.cuda(args.gpu, non_blocking=True)
            label = labeled_label.cuda(args.gpu, non_blocking=True)  # just for labeled prediction
            domain_flag = domain_flag.cuda(args.gpu, non_blocking=True)

            # forward
            logit = model(im_q)["logits_lab"]
            _, pred = logit.max(dim=-1)
            # calculate batch loss and accuracy
            loss_supervised = F.cross_entropy(pred, label)
            losses_supervised = np.append(losses_supervised, loss_supervised.cpu().numpy())

            targets=np.append(targets, label.cpu().numpy())
            preds=np.append(preds, pred.cpu().numpy())

        acc = cluster_acc(targets.astype(int), preds.astype(int))

        if args.use_wandb:
            #======================================================================
            wandb.log({
            'val/loss_supervised': losses_supervised.mean(),
            'val/acc': acc
            })
            #======================================================================   
    else:
        raise NotImplementedError
    return acc