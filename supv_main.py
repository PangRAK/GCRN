import os
import time
import random
import json
from tqdm import tqdm
from collections import defaultdict
import h5py

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import numpy as np
from configs.opts import parser
from model.main_model import supv_main_model as main_model
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.Recorder import Recorder
from dataset.AVEDataset import AVEDataset
# from dataset.AVE_dataset import AVEDataset

import pdb

 # =================================  seed config ============================
SEED = 789
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_dtype(torch.float32)
# =============================================================================


def main():
    # utils variable
    global args, logger, writer, dataset_configs
    # statistics variable
    global best_accuracy, best_accuracy_epoch
    # configs
    dataset_configs = get_and_save_args(parser)
    parser.set_defaults(**dataset_configs)
    args = parser.parse_args()
    # select GPUs
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    '''Create snapshot_pred dir for copying code and saving models '''
    if not os.path.exists(args.snapshot_pref):
        os.makedirs(args.snapshot_pref)

    if os.path.isfile(args.resume):
        args.snapshot_pref = os.path.dirname(args.resume)
    
    logger = Prepare_logger(args, eval=args.evaluate)

    if not args.evaluate:
        logger.info(f'\nCreating folder: {args.snapshot_pref}')
        logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))
    else:
        logger.info(f'\nLog file will be save in a {args.snapshot_pref}/Eval.log.')

    
    
    '''Split setting'''
    data_root = 'D:/AVE_Dataset_feature/'
    train_splits_path = os.path.join(data_root, f'train_splits.h5')
    val_splits_path = os.path.join(data_root, f'val_splits.h5')
    test_splits_path = os.path.join(data_root, f'test_splits.h5')
    # train_splits_path = os.path.join(data_root, f'consistent_order.h5')
    # val_splits_path = os.path.join(data_root, f'val_splits.h5')
    # test_splits_path = os.path.join(data_root, f'inconsistent_order.h5')

    with h5py.File(train_splits_path, 'r') as f:
        train_splits = f['order'][:]
    with h5py.File(val_splits_path, 'r') as f:
        val_splits = f['order'][:]
    with h5py.File(test_splits_path, 'r') as f:
        test_splits = f['order'][:]
    
    best_acc_list = []
    k = 10
    for fold in range(k):
        best_accuracy, best_accuracy_epoch = 0, 0
        
        '''Dataset'''
        train_dataloader = DataLoader(
            AVEDataset(data_root, split='train'),
            batch_size=args.batch_size,
            sampler=train_splits[fold],
            num_workers=0,
            pin_memory=True
        )

        test_dataloader = DataLoader(
            AVEDataset(data_root,split='test'),
            batch_size=args.test_batch_size,
            sampler=test_splits[fold],
            num_workers=0,
            pin_memory=True
        )

        '''model setting'''
        mainModel = main_model()
        mainModel = nn.DataParallel(mainModel).cuda()
        learned_parameters = mainModel.parameters()
        optimizer = torch.optim.Adam(learned_parameters, lr=args.lr)
        # scheduler = StepLR(optimizer, step_size=40, gamma=0.2)
        scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
        criterion = nn.BCEWithLogitsLoss().cuda()
        criterion_event = nn.CrossEntropyLoss().cuda()

        '''Resume from a checkpoint'''
        if os.path.isfile(args.resume):
            logger.info(f"\nLoading Checkpoint: {args.resume}\n")
            mainModel.load_state_dict(torch.load(args.resume), strict=False)
        elif args.resume != "" and (not os.path.isfile(args.resume)):
            raise FileNotFoundError

        '''Only Evaluate'''
        if args.evaluate:
            logger.info(f"\nStart Evaluation..")
            
            # RAK
            test_splits_path = os.path.join(data_root, f'mismatch_order.h5')
            with h5py.File(test_splits_path, 'r') as f:
                test_splits = f['order'][:]
            test_dataloader = DataLoader(
                AVEDataset(data_root,split='test'),
                batch_size=args.test_batch_size,
                sampler=test_splits,
                num_workers=0,
                pin_memory=True
            )
            
            validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch=0, eval_only=True)
            return
        
        '''Tensorboard and Code backup'''
        writer = SummaryWriter(args.snapshot_pref)
        # recorder = Recorder(args.snapshot_pref, ignore_folder="Exps/")
        # recorder.writeopt(args)

        '''Training and Testing'''
        for epoch in range(args.n_epoch):
            loss = train_epoch(mainModel, train_dataloader, criterion, criterion_event, optimizer, epoch, fold)

            if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
                acc, pred, targets, file_names, feat, event_relevant_score, segment_similar, is_event_scores = validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch, fold=fold)

                if acc > best_accuracy:
                    best_accuracy = acc
                    best_accuracy_epoch = epoch
                    # save_checkpoint(
                    #     mainModel.state_dict(),
                    #     top1=best_accuracy,
                    #     task='Supervised',
                    #     epoch=epoch + 1,
                    # )
                    # save_data(pred, targets, file_names, feat, event_relevant_score, segment_similar, is_event_scores, fold, epoch )
                logger.info(
                    f'*********************************************************************************\n'
                    f'Fold: [{fold}] Epoch: [{epoch}] (best epoch : {best_accuracy_epoch}, best acc : {best_accuracy})\n'
                    f'*********************************************************************************\n'
                )
            scheduler.step()
        
        best_acc_list.append(best_accuracy)
    for x in best_acc_list:
        logger.info(x)


def train_epoch(model, train_dataloader, criterion, criterion_event, optimizer, epoch, fold):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    end_time = time.time()

    model.train()
    # Note: here we set the model to a double type precision, 
    # since the extracted features are in a double type. 
    # This will also lead to the size of the model double increases.
    model.float()
    optimizer.zero_grad()

    for n_iter, batch_data in enumerate(train_dataloader):
    
        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        visual_feature, audio_feature, labels, _ = batch_data
        # For a model in a float precision
        # visual_feature = visual_feature.float()
        # audio_feature = audio_feature.float()
        labels = labels.double().cuda()
        is_event_scores, event_scores, _, _, _ = model(visual_feature, audio_feature)
        is_event_scores = is_event_scores.transpose(1, 0).squeeze().contiguous()

        labels_foreground = labels[:, :, :-1]  # [32, 10, 28]
        labels_BCE, labels_evn = labels_foreground.max(-1)

        labels_event, _ = labels_evn.max(-1)
        loss_is_event = criterion(is_event_scores, labels_BCE.cuda()) # torch.Size([32, 10])
        if event_scores.dim() == 3:
            loss_event_class = criterion_event(event_scores, labels_foreground.cuda()) # torch.Size([32, 10, 28])
        else:
            loss_event_class = criterion_event(event_scores, labels_event.cuda())
        loss = (0.5*loss_is_event) + (0.5*loss_event_class)
        loss.backward()

        '''Compute Accuracy'''
        acc, _, _ = compute_accuracy_supervised(is_event_scores, event_scores, labels, epoch=epoch)
        train_acc.update(acc.item(), visual_feature.size(0) * 10)

        '''Clip Gradient'''
        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                logger.info(f'Clipping gradient: {total_norm} with coef {args.clip_gradient/total_norm}.')

        '''Update parameters'''
        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), visual_feature.size(0)*10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        '''Add loss of a iteration in Tensorboard'''
        writer.add_scalar('Train_data/loss', losses.val, epoch * len(train_dataloader) + n_iter + 1)

        '''Print logs in Terminal'''
        if n_iter % args.print_freq == 0:
            logger.info(
                f'Train Fold: [{fold}] Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {train_acc.val:.3f} ({train_acc.avg: .3f})'
            )

        '''Add loss of an epoch in Tensorboard'''
        writer.add_scalar('Train_epoch_data/epoch_loss', losses.avg, epoch)

    return losses.avg



@torch.no_grad()
def validate_epoch(model, test_dataloader, criterion, criterion_event, epoch, fold = 0, eval_only=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end_time = time.time()

    model.eval()
    model.float()

    for n_iter, batch_data in enumerate(test_dataloader):
        
        data_time.update(time.time() - end_time)

        '''Feed input to model'''
        visual_feature, audio_feature, labels, file_names = batch_data
        # For a model in a float type
        # visual_feature = visual_feature.float()
        # audio_feature = audio_feature.float()
        labels = labels.double().cuda()
        bs = visual_feature.size(0)
        is_event_scores, event_scores, feat, event_relevant_score, segment_similar = model(visual_feature, audio_feature, file_names, epoch)
        is_event_scores = is_event_scores.transpose(1, 0).squeeze()

        labels_foreground = labels[:, :, :-1]
        labels_BCE, labels_evn = labels_foreground.max(-1)
        labels_event, _ = labels_evn.max(-1)
        loss_is_event = criterion(is_event_scores, labels_BCE.cuda())
        if event_scores.dim() == 3:
            loss_event_class = criterion_event(event_scores, labels_foreground.cuda()) # torch.Size([32, 10, 28])
        else:
            loss_event_class = criterion_event(event_scores, labels_event.cuda())
        loss = (0.3*loss_is_event) + (0.7*loss_event_class)

        acc, pred, targets = compute_accuracy_supervised(is_event_scores, event_scores, labels)
        accuracy.update(acc.item(), bs * 10)

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        losses.update(loss.item(), bs * 10)

        '''Print logs in Terminal'''
        if n_iter % args.print_freq == 0:
            logger.info(
                f'Test Fold: [{fold}] Epoch: [{epoch}][{n_iter}/{len(test_dataloader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {accuracy.val:.3f} ({accuracy.avg:.3f})'
            )
        
        if n_iter == 0:
            all_pred = pred
            all_target = targets
            all_file_names = file_names
            all_feat = feat.squeeze()
            all_event_relevant_score = event_relevant_score.squeeze()
            all_segment_similar = segment_similar.squeeze()
            all_event_relevant_score = event_relevant_score.squeeze()
            all_is_event_scores = is_event_scores.squeeze()
        else:
            all_pred = torch.cat((all_pred, pred), dim=0)
            all_target = torch.cat((all_target, targets), dim=0)
            all_file_names = all_file_names + file_names
            all_feat = torch.cat((all_feat, feat.squeeze()), dim=0)
            all_event_relevant_score = torch.cat((all_event_relevant_score, event_relevant_score.squeeze()), dim=0)
            all_segment_similar = torch.cat((all_segment_similar, segment_similar.squeeze()), dim=0)
            all_is_event_scores = torch.cat((all_is_event_scores, is_event_scores.squeeze()), dim=0)
            

    '''Add loss in an epoch to Tensorboard'''
    if not eval_only:
        writer.add_scalar('Val_epoch_data/epoch_loss', losses.avg, epoch)
        writer.add_scalar('Val_epoch/Accuracy', accuracy.avg, epoch)

    logger.info(
        f"\tEvaluation results (acc): {accuracy.avg:.4f}%."
    )
    return accuracy.avg, all_pred, all_target, all_file_names, all_feat, \
        all_event_relevant_score, all_segment_similar, all_is_event_scores

def post_process(is_event, window_length=3):
    batch = is_event.shape[0]
    sequence_length = is_event.shape[-1]
    res = is_event.clone()

    for b in range(batch):
        count = 0
        start = -1
        end = 0
        for i in range(sequence_length):
            current_pred = is_event[b, i]
            
            if current_pred == True:
                if count == 0:
                    start = i
                count += 1
            else:
                if count > 0 and count < window_length:
                    end = i
                    res[b, start:end] = False
                start = i
                count = 0

    return res

def compute_accuracy_supervised(is_event_scores, event_scores, labels, epoch=None):
    # if epoch is not None and epoch > 48:
    #     pdb.set_trace()
    if event_scores.dim() == 3:
        # # labels : torch.Size([32, 10, 29])
        # _, targets = labels.max(-1) # torch.Size([32, 10])
        # # pos pred
        # scores_pos_ind = is_event_scores > 0.5
        # scores_mask = scores_pos_ind == 0
        # _, event_class = event_scores.max(-1) # foreground classification
        # pred = scores_pos_ind.long() # torch.Size([32, 10])
        
        # pred *= event_class
        
        # # add mask
        # pred[scores_mask] = 28 # 28 denotes bg
        # correct = pred.eq(targets)
        # correct_num = correct.sum().double()
        # acc = correct_num * (100. / correct.numel())
        pass
    else:
        # labels : torch.Size([32, 10, 29])
        _, targets = labels.max(-1) # torch.Size([32, 10])
        # pos pred
        # is_event_scores = is_event_scores.sigmoid()
        scores_pos_ind = is_event_scores > 0.5
        scores_pos_ind = post_process(scores_pos_ind, window_length=3)
        scores_mask = scores_pos_ind == 0
        _, event_class = event_scores.max(-1) # foreground classification
        pred = scores_pos_ind.long() # torch.Size([32, 10])
        
        pred *= event_class[:, None]
        
        # add mask
        pred[scores_mask] = 28 # 28 denotes bg
        correct = pred.eq(targets)
        correct_num = correct.sum().double()
        acc = correct_num * (100. / correct.numel())


    return acc, pred, targets

def save_checkpoint(state_dict, top1, task, epoch):
    model_name = f'{args.snapshot_pref}/model_epoch_{epoch}_top1_{top1:.3f}_task_{task}_best_model.pth.tar'
    torch.save(state_dict, model_name)
    
def save_data(pred, target, file_name, feat, event_relevant_score, segment_similar, is_event_scores, fold, epoch):
    data_name = f'{args.snapshot_pref}/data_fold_{fold}_epoch_{epoch}.pth'
    # pdb.set_trace()
    data = {'pred' : pred,
            'target' : target,
            'file_name' : file_name,
            'feature' : feat,
            'event_relevant_score' : event_relevant_score,
            'segment_similar' : segment_similar,
            'is_event_scores' : is_event_scores}
    torch.save(data, data_name)


if __name__ == '__main__':
    main()