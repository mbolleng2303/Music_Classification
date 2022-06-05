import argparse
import json
import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json

from matplotlib import pyplot as plt

from nets.LoadNet import load_model
from data.Dataset import MusicDataset
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader
from train.training import train_epoch_graph as train_epoch
from train.training import evaluate_network_graph as evaluate_network
from tensorboardX import SummaryWriter
from tqdm import tqdm



def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


"""
    VIEWING MODEL CONFIG AND PARAMS
"""


def view_model_param(MODEL_NAME, net_params):
    model = load_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    # print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    TRAINING CODE
"""


def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):
    avg_test_acc = []
    avg_train_acc = []
    avg_epochs = []
    t0 = time.time()
    per_epoch_time = []
    DATASET_NAME = dataset.name
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n""".format(
            DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        if params['cross_fold']:
            split = [i for i in range(5)]
        else:
            split = [params['fold']]
        for split_number in split:

            t0_split = time.time()
            log_dir = os.path.join(root_log_dir, "RUN_" + str(split_number))
            writer = SummaryWriter(log_dir=log_dir)
            # setting seeds
            random.seed(params['seed'])
            np.random.seed(params['seed'])
            torch.manual_seed(params['seed'])
            if device.type == 'cuda':
                torch.cuda.manual_seed(params['seed'])
            print("RUN NUMBER: ", split_number)
            trainset, valset, testset = dataset.train[split_number], dataset.val[split_number], dataset.test[
                split_number]
            print("Training samples: ", len(trainset))
            print("Validation samples: ", len(valset))
            print("Test samples: ", len(testset))
            print("Number of Classes: ", net_params['n_classes'])
            model = load_model(MODEL_NAME, net_params)
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             factor=params['lr_reduce_factor'],
                                                             patience=params['lr_schedule_patience'],
                                                             verbose=True, min_lr=params['min_lr'])

            epoch_train_losses, epoch_val_losses = [], []
            epoch_train_accs, epoch_val_accs = [], []
            epoch_test_accs = []

            drop_last =False

            train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last,
                                      collate_fn=dataset.collate)
            val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last,
                                    collate_fn=dataset.collate)
            test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last,
                                     collate_fn=dataset.collate)

            with tqdm(range(params['epochs'])) as t:
                for epoch in t:
                    t.set_description('Epoch %d' % epoch)
                    start = time.time()
                    epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)
                    epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader, epoch)
                    _, epoch_test_acc = evaluate_network(model, device, test_loader, epoch)
                    epoch_train_losses.append(epoch_train_loss)
                    epoch_val_losses.append(epoch_val_loss)
                    epoch_train_accs.append(epoch_train_acc)
                    epoch_val_accs.append(epoch_val_acc)
                    epoch_test_accs.append(epoch_test_acc)
                    """
                    try :
                        if epoch_val_loss < epoch_val_losses[-2]:
                            nbr_nochange = 0
                            save = True
                        else :
                            nbr_nochange+=1
                            save = True
                    except IndexError:
                        nbr_nochange = 0
                        save = True
                    """
                    writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                    writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                    writer.add_scalar('train/_acc', epoch_train_acc, epoch)
                    writer.add_scalar('val/_acc', epoch_val_acc, epoch)
                    writer.add_scalar('test/_acc', epoch_test_acc, epoch)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)


                    epoch_train_acc = 100. * epoch_train_acc
                    epoch_test_acc = 100. * epoch_test_acc
                    epoch_val_acc = 100. * epoch_val_acc

                    t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                                  train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                                  train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                                  test_acc=epoch_test_acc)

                    per_epoch_time.append(time.time() - start)

                    # Saving checkpoint
                    #if save :
                    ckpt_dir = os.path.join(root_ckpt_dir, "RUN_" + str(split_number))
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                    files = glob.glob(ckpt_dir + '/*.pkl')
                    for file in files:
                        epoch_nb = file.split('_')[-1]
                        epoch_nb = int(epoch_nb.split('.')[0])
                        if epoch_nb < epoch - 1:
                             os.remove(file)

                    scheduler.step(epoch_val_loss)

                    if optimizer.param_groups[0]['lr'] <= params['min_lr']:
                        print('LR min')
                        break

                    # Stop training after params['max_time'] hours
                    if time.time() - t0_split > params[
                        'max_time'] * 3600 / 5:  # Dividing max_time by 5, since there are 5 runs
                        print('-' * 89)
                        print(
                            "Max_time for one train-val-test split experiment elapsed {:.3f} hours, so stopping".format(
                                params['max_time'] / 10))
                        break

            _, test_acc = evaluate_network(model, device, test_loader,epoch)
            _, train_acc = evaluate_network(model, device, train_loader,epoch)
            _, val_acc = evaluate_network(model, device, val_loader,epoch)


            avg_epochs.append(epoch)
            plt.figure()
            plt.plot(epoch_train_accs)
            plt.plot(epoch_val_accs)
            plt.legend(['train', 'val'])
            plt.ylabel('acc')
            plt.xlabel('epoch')
            plt.title('Training summary fold '.format(split_number))
            plt.savefig(log_dir +'/acc')
            plt.figure()
            plt.plot(epoch_train_losses)
            plt.plot(epoch_val_losses)
            plt.legend(['train', 'val'])
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.title('Training summary fold '.format(split_number))
            plt.savefig(log_dir + '/loss')

            '''np.array(epoch_train_losses).tofile('train_loss',sep = ',')
            np.array(epoch_val_losses).tofile('val_loss',sep = ',')
            np.array(epoch_train_accs).tofile('train_acc',sep = ',')
            np.array(epoch_val_accs).tofile('val_acc',sep = ',')'''
            best_idx = np.argmax(epoch_val_accs)
            print("Best epoch : {}".format(best_idx+1))
            print("Test Accuracy [BEST]: {:.4f} ".format(epoch_test_accs[best_idx]*100))
            print("Train Accuracy [BEST]: {:.4f}".format(epoch_train_accs[best_idx]*100))
            print("Val Accuracy [BEST]: {:.4f}".format(epoch_val_accs[best_idx]*100))
            avg_test_acc.append(epoch_test_accs[best_idx])
            avg_train_acc.append(epoch_train_accs[best_idx])

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    print("TOTAL TIME TAKEN: {:.4f}hrs".format((time.time() - t0) / 3600))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    # Final test accuracy value averaged over 5-fold
    print("""\n\n\nFINAL RESULTS\n\nTEST ACCURACY averaged: {:.4f} with s.d. {:.4f}""".format(
        np.mean(np.array(avg_test_acc)) * 100, np.std(avg_test_acc) * 100))
    print("\nAll splits Test Accuracies:\n", avg_test_acc)
    print("""\n\n\nFINAL RESULTS\n\nTRAIN ACCURACY averaged: {:.4f} with s.d. {:.4f}""".format(
        np.mean(np.array(avg_train_acc)) * 100, np.std(avg_train_acc) * 100))
    print("\nAll splits Train Accuracies:\n", avg_train_acc)
    writer.close()

    """
        Write the results in out/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST ACCURACY averaged: {:.3f}\n with test acc s.d. {:.3f}\nTRAIN ACCURACY averaged: {:.3f}\n with train s.d. {:.3f}\n\n
    Convergence Time (Epochs): {:.3f}\nTotal Time Taken: {:.3f} hrs\nAverage Time Per Epoch: {:.3f} s\n\n\nAll Splits Test Accuracies: {}\n\nAll Splits Train Accuracies: {}""" \
                .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                        np.mean(np.array(avg_test_acc)) * 100, np.std(avg_test_acc) * 100,
                        np.mean(np.array(avg_train_acc)) * 100, np.std(avg_train_acc) * 100,
                        np.mean(np.array(avg_epochs)),
                        (time.time() - t0) / 3600, np.mean(per_epoch_time), avg_test_acc, avg_train_acc))



def main():
    """
        USER CONTROLS
    """

    config_path = "configs/config_graph.json"  # <-- change this


    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=config_path, help="Please give a config_conv.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--data_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--L_mlp', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel_size', help="Please give a value for kernel")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--in_feat_dropout', help="Please give a value for dropout")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--data_feat', help="Please give a value for data_mode")
    parser.add_argument('--pool', help="Please give a value for num_pool")
    parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--cross_fold', help="Please give a value for max_time")
    parser.add_argument('--fold', help="Please give a value for max_time")
    parser.add_argument('--decrease_conv', help="Please give a value for max_time")

    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        data_dir = config['data_dir']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    if args.data_feat is not None:
        data_feat = args.data_feat
    else:
        data_feat = config['data_feat']
    dataset = MusicDataset(name=DATASET_NAME, data_dir=data_dir,data_feat=data_feat)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    if args.cross_fold is not None:
        params['cross_fold'] = args.cross
    if args.fold is not None:
        params['fold'] = args.fold
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.L is not None:
        net_params['L_mlp'] = int(args.L_mlp)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)
    if args.residual is not None:
        net_params['residual'] = True if args.residual == 'True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel_size is not None:
        net_params['kernel_size'] = int(args.kernel_size)
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.decrease_conv is not None:
        net_params['decrease_conv'] = True if args.decrease_conv == 'True' else False
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm == 'True' else False
    if args.data_feat is not None:
        net_params['data_feat'] = args.data_feat
    if args.pool is not None:
        net_params['pool'] = str(args.pool)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)
    net_params['n_classes'] = 15

    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')

    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)


main()

