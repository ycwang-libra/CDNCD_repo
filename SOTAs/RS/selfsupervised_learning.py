import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from utils.util import AverageMeter, accuracy
import datetime
import wandb
from tqdm import tqdm

def train(model, data_loader, args, logger):
    train_lab_loader = data_loader['train_lab_loader']
    train_unlab_loader = data_loader['train_unlab_loader']

    # init wandb
    if args.use_wandb:
        #======================================================================
        wandb.init(project=args.wandb_project, config = args.__dict__, name = args.current_log_withtime, save_code=True, mode = 'offline' if args.use_wandb_offline else 'online')
        model.run_id = wandb.run.id
        #======================================================================
        
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4, nesterov=True)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(args.epochs):
        logger.info('########## epoch: ' + str(epoch) + ' ##########')
        loss_record = AverageMeter()
        acc_record = AverageMeter()
        model.train()
        unlab_dataloader_iterator = iter(train_unlab_loader(epoch))
        for batch_idx, (lab_img, lab_label, lab_domain_flag) in enumerate(tqdm(train_lab_loader(epoch))):
            try:
                unlab_img, unlab_label, unlab_domain_flag = next(unlab_dataloader_iterator)
            except:
                unlab_dataloader_iterator = iter(train_unlab_loader(epoch))
                unlab_img, unlab_label, unlab_domain_flag = next(unlab_dataloader_iterator)
            img = torch.cat([lab_img, unlab_img], dim = 0)
            label = torch.cat([lab_label, unlab_label], dim = 0)
            domain_flag = torch.cat([lab_domain_flag, unlab_domain_flag], dim = 0)

            img = img.cuda(args.gpu, non_blocking=True)
            label = label.cuda(args.gpu, non_blocking=True)
            domain_flag = domain_flag.cuda(args.gpu, non_blocking=True)

            optimizer.zero_grad()
            output = model(img, domain_flag)
            loss = criterion(output, label)
        
            # measure accuracy and record loss
            acc = accuracy(output, label)
            acc_record.update(acc[0].item(), img.size(0))
            loss_record.update(loss.item(), img.size(0))

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if args.use_wandb:
            #======================================================================
            wandb.log({'epoch':epoch, 
            'acc_epoch':acc[0].item(), 
            'lr_epoch': optimizer.param_groups[0]['lr'],
            'loss_epoch':loss.item()
            })
            #======================================================================

        exp_lr_scheduler.step()
        now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info('Train Epoch {} finished. Avg Loss: {:.4f}  Avg Acc: {:.4f}'.format(epoch, loss_record.avg, acc_record.avg))

        logger.info('##### Start to evaluate at {} epoch #####'.format(epoch))
        acc_record = test(model, data_loader, args, logger)
        is_best = acc_record.avg > best_acc 
        best_acc = max(acc_record.avg, best_acc)
        if is_best:
            state = {
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict(), 
                'epoch': epoch}
            selfsupervised_best_checkpoint_path = os.path.join(args.trained_model_root, 'checkpoint_best_' + args.current_log_withtime + '.pth.tar')
            torch.save(state, selfsupervised_best_checkpoint_path)
            logger.info("model saved to {}.".format(selfsupervised_best_checkpoint_path))
    
    if args.use_wandb:
        #======================================================================
        wandb.finish()
        #======================================================================

def test(model, data_loader, args, logger):
    lab_loader = data_loader['test_lab_loader']
    unlab_loader = data_loader['test_unlab_loader']
    acc_record = AverageMeter()
    model.eval()
    unlab_dataloader_iterator = iter(unlab_loader())
    for batch_idx, (lab_img, lab_label, lab_domain_flag) in enumerate(tqdm(lab_loader())):
        try:
            unlab_img, unlab_label, unlab_domain_flag = next(unlab_dataloader_iterator)
        except:
            unlab_dataloader_iterator = iter(unlab_loader())
            unlab_img, unlab_label, unlab_domain_flag = next(unlab_dataloader_iterator)
        img = torch.cat([lab_img, unlab_img], dim = 0)
        label = torch.cat([lab_label, unlab_label], dim = 0)
        domain_flag = torch.cat([lab_domain_flag, unlab_domain_flag], dim = 0)

        img = img.cuda(args.gpu, non_blocking=True)
        label = label.cuda(args.gpu, non_blocking=True)
        output = model(img, domain_flag)
     
        # measure accuracy and record loss
        acc = accuracy(output, label)
        acc_record.update(acc[0].item(), img.size(0))

    logger.info('Test on ' + args.dataset_series + ' data!')
    logger.info('Test Acc: {:.4f}'.format(acc_record.avg))

    return acc_record