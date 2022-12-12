import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from dataloaders import utils
from dataloaders.dataset import (RandomGenerator,
                                 TwoStreamBatchSampler)
from dataloaders.isic_dataset import BaseDataSets
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume
from config import get_config
from hint_loss import DistillHint
from hint_loss import ClassConsisten


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='D:/isic2018_deal/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ISIC/MedTrans', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='MedTrans', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')
args = parser.parse_args()
config = get_config(args)


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate" in dataset:
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        ref_dict = {"3": 55, "7": 181}
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=3,
                            class_num=num_classes, config=config, img_size=args.patch_size)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None)
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=0)

    model.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=0)

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)

    hint_loss_cnn = DistillHint(embed_dim_in=256, embed_dim_out=256)
    hint_loss_trans = DistillHint(embed_dim_in=768, embed_dim_out=768)

    class_loss_cnn = ClassConsisten()
    class_loss_trans = ClassConsisten()

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.729

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            noise = torch.clamp(torch.randn_like(
                unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise

            outputs_cnn, outputs_trans, feature1, feature2 = model(volume_batch)

            outputs_soft_cnn = torch.softmax(outputs_cnn, dim=1)
            outputs_soft_trans = torch.softmax(outputs_trans, dim=1)

            outputs_soft = (outputs_soft_cnn + outputs_soft_trans) / 2  # student result mean

            b, n, d = feature2.shape
            b, d1, h, w = feature1.shape

            feature2 = feature2.reshape(b, 7, 7, d).permute(0, 3, 1, 2)

            with torch.no_grad():
                ema_output_cnn, ema_output_trans, ema_cnn_fea, ema_trans_fea = ema_model(ema_inputs)
                ema_output_soft_cnn = torch.softmax(ema_output_cnn, dim=1)
                ema_output_soft_trans = torch.softmax(ema_output_trans, dim=1)
                b, n, d = ema_trans_fea.shape
                b, d1, h, w = ema_cnn_fea.shape

                ema_trans_fea = ema_trans_fea.reshape(b, 7, 7, d).permute(0, 3, 1, 2)

                uncertainty_cnn = -1.0 * \
                                  torch.sum(ema_output_soft_cnn * torch.log(ema_output_soft_cnn + 1e-6), dim=1,
                                            keepdim=True)
                uncertainty_trans = -1.0 * \
                                    torch.sum(ema_output_soft_trans * torch.log(ema_output_soft_trans + 1e-6), dim=1,
                                              keepdim=True)

                uncentain_map_1 = torch.zeros_like(uncertainty_cnn)
                uncentain_map_1[uncertainty_cnn < uncertainty_trans] = 1

                uncentain_map_2 = torch.zeros_like(uncertainty_cnn)
                uncentain_map_2[uncertainty_cnn > uncertainty_trans] = 1

                cnn_filter = uncentain_map_1 * ema_output_soft_cnn
                trans_filter = uncentain_map_2 * ema_output_soft_trans
                ema_output_soft = cnn_filter + trans_filter

            loss_ce_cnn = ce_loss(outputs_cnn[:args.labeled_bs],
                              label_batch[:][:args.labeled_bs].long()) * 0.5
            loss_ce_trans = ce_loss(outputs_trans[:args.labeled_bs],
                              label_batch[:][:args.labeled_bs].long()) * 0.5
            loss_ce = loss_ce_trans + loss_ce_cnn

            loss_dice_cnn = dice_loss(
                outputs_soft_cnn[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)) * 0.5

            loss_dice_trans = dice_loss(
                outputs_soft_trans[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)) * 0.5

            loss_dice = loss_dice_cnn + loss_dice_trans

            supervised_loss = 0.5 * (loss_dice + loss_ce)

            pseudo_outputs_cnn = torch.argmax(
                outputs_soft_cnn[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs_trans = torch.argmax(
                outputs_soft_trans[args.labeled_bs:].detach(), dim=1, keepdim=False)

            pseudo_supervision1 = dice_loss(
                outputs_soft_cnn[args.labeled_bs:], pseudo_outputs_trans.unsqueeze(1))
            pseudo_supervision2 = dice_loss(
                outputs_soft_trans[args.labeled_bs:], pseudo_outputs_cnn.unsqueeze(1))
            unsupervised_loss = 0.5 * (pseudo_supervision1 + pseudo_supervision2)

            consistency_weight = get_current_consistency_weight(iter_num//150)

            if iter_num < 1000:
                consistency_loss = 0.0
                loss_hint = 0.0
                loss_cls = 0.0
            else:
                consistency_loss_cnn = dice_loss(
                    outputs_soft_cnn[args.labeled_bs:], torch.argmax(ema_output_soft, dim=1).unsqueeze(1)
                ) * 0.5
                consistency_loss_trans = dice_loss(
                    outputs_soft_trans[args.labeled_bs:], torch.argmax(ema_output_soft, dim=1).unsqueeze(1)
                ) * 0.5
                consistency_loss = consistency_loss_trans + consistency_loss_cnn

                loss_hint_cnn = hint_loss_cnn(ema_cnn_fea.detach(), feature1[args.labeled_bs:])
                loss_hint_trans = hint_loss_trans(ema_trans_fea.detach(), feature2[args.labeled_bs:])
                loss_hint = 0.5 * (loss_hint_trans + loss_hint_cnn)

                loss_cls_cnn = class_loss_cnn(ema_cnn_fea.detach(), feature1[args.labeled_bs:],
                                              ema_output_soft_cnn.detach(), outputs_soft_cnn[args.labeled_bs:])
                loss_cls_trans = class_loss_trans(ema_trans_fea.detach(), feature2[args.labeled_bs:],
                                                  ema_output_soft_trans.detach(), outputs_soft_trans[args.labeled_bs:])
                loss_cls = 0.5 * (loss_cls_trans + loss_cls_cnn)

            loss = supervised_loss + consistency_weight * consistency_loss + loss_hint * 0.5 + unsupervised_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)
            writer.add_scalar('info/loss_hint', loss_hint, iter_num)
            writer.add_scalar('info/loss_cls', loss_cls, iter_num)
            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_hint: %f, loss_cls: %f'%
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_hint, loss_cls))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(outputs_soft, dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0

                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]

                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                      'iter_{}_dice_{}.pth'.format(
                                                          iter_num, round(performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                                 '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
