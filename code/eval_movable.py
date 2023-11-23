import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import sapien_dateset
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='movable_module', help='Model file name [default: movable_module]')
    parser.add_argument('--train_cat_index', default='11,24,31,36', help='Index of training cat [default: 11,24,31,36]')
    parser.add_argument('--test_cat_index', default='14', help='test cat index [default: 45]')
    parser.add_argument('--batch_size', default=64, type=int, help='Eval batch_size [default: 64]')
    parser.add_argument('--data_dir', default='../data', type=str, help='Path of data [default: ../data]')
    parser.add_argument('--log_dir', default='../log/log_movable_train', type=str, help='Dump dir to load model checkpoint [default: ../log/log_movable_train]')
    parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
    parser.add_argument('--dump_dir', default='../log/log_movable_eval', help='Dump dir to save sample outputs [../log/log_movable_eval]')
    args = parser.parse_args()
    return args


args = parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
BATCH_SIZE = args.batch_size
LOG_DIR = args.log_dir
if args.checkpoint_path is None:
    CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint_cat_idx[%s].tar' % args.train_cat_index)
else:
    CHECKPOINT_PATH = args.checkpoint_path
assert (CHECKPOINT_PATH is not None)
DUMP_DIR = args.dump_dir
TEST_INDEX = args.test_cat_index
DATA_DIR = args.data_dir

# Prepare DUMP_DIR
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
DUMP_FOUT = open(os.path.join(DUMP_DIR, 'log_movable_eval_idx:[%s].txt' % TEST_INDEX), 'w')
DUMP_FOUT.write(str(args) + '\n')


def log_string(out_str):
    DUMP_FOUT.write(out_str + '\n')
    DUMP_FOUT.flush()
    print(out_str)


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


TEST_DATA_FLODER = os.path.join(DATA_DIR, 'movable_test_cat_idx[%s]' % TEST_INDEX)
TEST_DATASET = sapien_dateset(TEST_DATA_FLODER, 'movable')
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=4, worker_init_fn=my_worker_init_fn)

# Init the model and optimzier
MODEL = importlib.import_module(args.model)
net = MODEL.movable_net()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Load checkpoint if there is any
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    log_string("Loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, epoch))

# ------------------------------------------------------------------------- GLOBAL CONFIG END

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



def eval():
    log_string(str(datetime.now()))
    np.random.seed()
    loss = evaluate_one_epoch()


if __name__ == '__main__':
    eval()