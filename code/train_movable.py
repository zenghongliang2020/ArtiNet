import os
import sys
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import sapien_dateset
import argparse
import importlib

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
path = os.path.join(rootPath, 'code', 'models')
sys.path.append(os.path.join(rootPath, 'utils'))
sys.path.append(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='movable_module', help='Model file name [default: movable_module]')
    parser.add_argument('--train_cat_index', default='11,24,31,36', help='Index of training cat [default: 11,24,31,36]')
    parser.add_argument('--batch_size', default=64, type=int, help='Trained batch_size [default: 64]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Trained learning rate [default: 0.001]')
    parser.add_argument('--max_epoch', default=10, type=int, help='Trained epochs [default: 10]')
    parser.add_argument('--data_dir', default='../data', type=str, help='Path of data [default: ../data]')
    parser.add_argument('--log_dir', default='../log/log_train_movable', type=str, help='Dump dir to save model checkpoint [default: ../log/log_movable_train]')
    parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data-loading processes [default: 4]')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay value [default: 1e-4]')
    parser.add_argument('--lr_decay_steps', default='8,9',
                        help='When to decay the learning rate(in epochs) [default: 10, 15]')
    parser.add_argument('--lr_decay_rates', default='0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1]')

    args = parser.parse_args()
    return args


args = parse_args()

# ---------------------------------------------------------------GLOBAL CONFIG
BATCH_SIZE = args.batch_size
BASE_LEARNING_RATE = args.learning_rate
MAX_EPOCH = args.max_epoch
LOG_DIR = args.log_dir
DATA_DIR = args.data_dir
TRAIN_IDX = args.train_cat_index
NUM_WORKERS = args.num_workers
WEIGHT_DECAY = args.weight_decay
LR_DECAY_STEPS = [int(x) for x in args.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in args.lr_decay_rates.split(',')]
DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint_idx%s.tar' % TRAIN_IDX)
CHECKPOINT_PATH = args.checkpoint_path if args.checkpoint_path is not None else DEFAULT_CHECKPOINT_PATH

# Prepare LOG_DIR
if os.path.exists(LOG_DIR) and args.overwrite:
    print('Log folder %s already exists, Are you sure to overwrite? (Y/N)' % LOG_DIR)
    k = input()
    if k == 'n' or k == 'N':
        print('Exiting...')
        exit()
    elif k == 'y' or k == 'Y':
        print('Overwrite the files in the log folder...')
        os.system('rm -r %s' % LOG_DIR)

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_movable_train_idx:%s' % TRAIN_IDX), 'a')
LOG_FOUT.write(str(args) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# Create Dataset and Dataloader
TRAIN_DATA_FLODER = os.path.join(DATA_DIR, 'movable_train_cat_idx[%s]'%TRAIN_IDX)
TEST_DATA_FLODER = os.path.join(DATA_DIR, 'movable_val_cat_idx[%s]'%TRAIN_IDX)

TRAIN_DATASET = sapien_dateset(TRAIN_DATA_FLODER, 'movable')
TEST_DATASET = sapien_dateset(TEST_DATA_FLODER, 'movable')

TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, worker_init_fn=my_worker_init_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, worker_init_fn=my_worker_init_fn)
print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))

# Init the model and optimizer
MODEL = importlib.import_module(args.model)
net = MODEL.movable_net()

if torch.cuda.device_count() > 1:
    log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
    net = nn.DataParallel(net)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)
criterion = nn.BCEWithLogitsLoss()

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Load checkpoint if there is any
it = -1
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))


def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i, lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
        return lr


def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_one_epoch():
    stat_dict = {'loss': 0, 'acc': 0}
    adjust_learning_rate(optimizer, EPOCH_CNT)
    net.train()
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        optimizer.zero_grad()
        inputs = {'img': batch_data_label['img'], 'point_clouds': batch_data_label['point_clouds']}
        outputs = net(inputs)
        outputs = outputs.squeeze(dim=1)

        # Compute loss and gradient, update parameters
        label = batch_data_label['label'].squeeze(dim=1)
        loss = criterion(outputs, label.float())
        loss.backward()
        optimizer.step()

        correct = 0
        predictions = (outputs > 0.5).long()
        correct += (predictions == label).sum().item()
        accuracy = correct / label.size(0)
        stat_dict['loss'] += loss
        stat_dict['acc'] += accuracy

        batch_interval = 20
        if (batch_idx + 1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx + 1))
            for key in sorted(stat_dict.keys()):
                if key == 'acc':
                    log_string('mean %s: %.2f%%' % (key, (stat_dict[key] * 100) / batch_interval))
                else:
                    log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval))
                stat_dict[key] = 0


def evaluate_one_epoch():
    stat_dict = {'loss': 0, 'acc': 0}
    net.eval()
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if batch_idx % 20 == 0:
            print('Eval batch: %d' % batch_idx)
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        inputs = {'img': batch_data_label['img'], 'point_clouds': batch_data_label['point_clouds']}
        with torch.no_grad():
            outputs = net(inputs)
        outputs = outputs.squeeze(dim=1)

        # Compute loss
        label = batch_data_label['label'].squeeze(dim=1)
        loss = criterion(outputs, label.float())

        correct = 0
        predictions = (outputs > 0.5).long()
        correct += (predictions == label).sum().item()
        accuracy = correct / label.size(0)
        stat_dict['loss'] += loss
        stat_dict['acc'] += accuracy

    for key in sorted(stat_dict.keys()):
        if key == 'acc':
            log_string('mean %s: %.2f%%' % (key, (stat_dict[key] * 100) / float(batch_idx + 1)))
        else:
            log_string('mean %s: %f' % (key, stat_dict[key] / float(batch_idx + 1)))
    mean_loss = stat_dict['loss'] / float(batch_idx + 1)
    return mean_loss


def train(start_epoch):
    global EPOCH_CNT
    loss = 0
    for epoch in range(start_epoch, MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f' % (get_current_lr(epoch)))
        log_string(str(datetime.now()))
        np.random.seed()
        train_one_epoch()
        if EPOCH_CNT == 0 or EPOCH_CNT % 5 == 4:  # Eval every 5 epochs
            loss = evaluate_one_epoch()
        # Save checkpoint
        save_dict = {'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
                     'optimizer_state_dict': optimizer.state_dict(),
                     'loss': loss,
                     }
        try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint_cat_idx[%s].tar' % TRAIN_IDX))


if __name__ == '__main__':
    train(start_epoch)
