import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import AverageMeter, cluster_acc
import wandb
from tqdm import tqdm

def train(model, data_loader, args, logger):
    train_lab_loader = data_loader['train_lab_loader']
    train_unlab_loader = data_loader['train_unlab_loader']

    # init wandb
    if args.use_wandb:
        #======================================================================
        wandb.init(project=args.wandb_project, config = args.__dict__, name = args.current_log, save_code=True, mode = 'offline' if args.use_wandb_offline else 'online')
        model.run_id = wandb.run.id
        #======================================================================

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma) # gamma = 0.5
    criterion1 = nn.CrossEntropyLoss().cuda(args.gpu)
    for epoch in range(args.epochs):
        logger.info('########## epoch: ' + str(epoch) + ' ##########')
        loss_record = AverageMeter()
        model.train()
        unlab_dataloader_iterator = iter(train_unlab_loader)
        for batch_idx, (lab_img, lab_label, lab_domain_flag) in enumerate(tqdm(train_lab_loader)):
            try:
                unlab_img, unlab_label, unlab_domain_flag = next(unlab_dataloader_iterator)
            except:
                unlab_dataloader_iterator = iter(train_unlab_loader)
                unlab_img, unlab_label, unlab_domain_flag = next(unlab_dataloader_iterator)
            img = torch.cat([lab_img, unlab_img], dim = 0)
            label = lab_label
            domain_flag = torch.cat([lab_domain_flag, unlab_domain_flag], dim = 0)
            img = img.cuda(args.gpu, non_blocking=True)
            label = label.cuda(args.gpu, non_blocking=True)
            domain_flag = domain_flag.cuda(args.gpu, non_blocking=True)
            
            output1, _, _ = model(img, domain_flag)
            output1, _ = output1.chunk(2) # just for labeled prediction
            loss = criterion1(output1, label)
            loss_record.update(loss.item(), img.size(0))
            
            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        exp_lr_scheduler.step()
        logger.info('Train Epoch {} finished. Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
            
        logger.info('##### Start to Evaluate at {} epoch #####'.format(epoch))
        logger.info('test on labeled classes')
        args.head = 'head1'
        acc, nmi, ari = test(model, data_loader, args, logger)

        if args.use_wandb:
            #======================================================================
            wandb.log({'supervised_epoch':epoch, 
            'supervised_eval_acc_epoch':acc,
            'supervised_eval_nmi_epoch':nmi,
            'supervised_eval_ari_epoch':ari,
            'supervised_lr_epoch': optimizer.param_groups[0]['lr'],
            'supervised_loss_epoch':loss.item()
            })
            #======================================================================

        # save the last supervised trained model
        state = {
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict(), 
                'epoch': epoch}
        supervised_checkpoint_path = os.path.join(args.trained_model_root, 'checkpoint_last_' + args.current_log_withtime + '.pth.tar')
        torch.save(state, supervised_checkpoint_path)
        logger.info("Final model saved to {}.".format(supervised_checkpoint_path))
    
    if args.use_wandb:
        #======================================================================
        wandb.finish()
        #======================================================================

def test(model, data_loader, args, logger):
    lab_loader = data_loader['test_lab_loader']
    unlab_loader = data_loader['test_unlab_loader']
    model.eval() 
    preds=np.array([])
    targets=np.array([])
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

        output1, output2, _ = model(img, domain_flag)  # torch.Size([128,5]) torch.Size([128,5])
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

    # recording
    logger.info('Test on '+ args.dataset_series + ' labeled data!')
    logger.info('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))

    return acc, nmi, ari
