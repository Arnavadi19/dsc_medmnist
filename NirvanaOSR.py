import os
import argparse
import datetime
import time
import csv
import pandas as pd
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torchvision.transforms as tf
import numpy as np

from modules.dchs import  NirvanaOpenset_loss
from Networks.models import classifier32
from Networks.resnet import resnet50
from datasets.osr_dataloader import Random300K_Images,BloodMNIST_OSR, OCTMnist_OSR, BloodMNIST_224_OSR
from utils import Logger, save_networks, load_networks
from core import test_ddfm_b9, train_Nirvana_oe
from split import splits_2020 as splits
from extractor import ViTExtractor

from medmnist import BloodMNIST, OCTMNIST

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser("Training")


# Dataset
parser.add_argument('--dataset', type=str, default='bloodmnist', choices=['bloodmnist', 'octmnist'], help="Dataset selection")
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--outf', type=str, default='./logs_results', help='Directory to save results')
# Optimization
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--max-epoch', type=int, default=100)
# model
parser.add_argument('--noisy-ratio', type=float, default=0.0, help="noisy ratio for ablation study")
parser.add_argument('--margin', type=float, default=48.0, help="margin for hinge")
parser.add_argument('--Expand', default=200, type=int, metavar='N', help='Expand factor of centers')
parser.add_argument('--model', type=str, default='classifier32', help='resnet50, classifier32, vit')
parser.add_argument('--loss', type=str, default='NirvanaOpenset')
# misc
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='../log_random30k_noisy_rampfalse')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--oe', action='store_true', help="Outlier Exposure", default=True)


def main_worker(options):
    is_best_acc_avg = False
    best_acc_avg = 0.
    results_best = dict()
    best_acc_avg_b9 = 0.
    results_b9_best = dict()
    options['ramp_activate'] = False
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: use_gpu = False
    
    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    # Dataset
    print("{} Preparation".format(options['dataset']))
    if options['dataset'] == 'bloodmnist':

        Data = BloodMNIST_OSR(
            known=options['known'],
            dataroot=options['dataroot'],
            use_gpu=not options['use_cpu'],
            batch_size=options['batch_size']
        )
        trainloader = Data.train_loader
        testloader = Data.test_loader 
        outloader = Data.out_loader

        if options['oe']:
            print("Outlier exposure mode is on. Other datasets")
            try:
                background_path = os.path.join(os.path.dirname(options['dataroot']), 
                                            '300K_random_images', 
                                            '300K_random_images.npy')
                if not os.path.exists(background_path):
                    raise FileNotFoundError(f"Background dataset not found at {background_path}")
                    
                oe_data = Random300K_Images(
                    file_path=background_path,
                    transform=tf.Compose([
                        tf.RandomCrop(32, padding=4),
                        tf.RandomHorizontalFlip(), 
                        tf.ToTensor(),
                    ]),
                    extendable=options['noisy_ratio']
                )
                print(f"Loaded background dataset with {len(oe_data)} images")
                
                trainloader_oe = torch.utils.data.DataLoader(
                    oe_data, 
                    batch_size=options['batch_size'], 
                    shuffle=True,
                    num_workers=0,
                    drop_last=True
                )
                print(f"Background loader created with {len(trainloader_oe)} batches")
            except Exception as e:
                print(f"Warning: Failed to load background dataset: {str(e)}")
                trainloader_oe = None

    elif options['dataset'] == 'octmnist':
        split_dict = splits['octmnist'][i]
        known = split_dict['known']
        unknown = split_dict['unknown']
        img_size = 32
        
        Data = OCTMnist_OSR(
            known=known,
            dataroot=options['dataroot'],
            use_gpu=not options['use_cpu'],
            batch_size=options['batch_size']
        )
        trainloader = Data.train_loader
        testloader = Data.test_loader
        outloader = Data.out_loader

        if options['oe']:
            print("Outlier exposure mode is on. Other datasets")
            try:
                background_path = os.path.join(os.path.dirname(options['dataroot']), 
                                            '300K_random_images', 
                                            '300K_random_images.npy')
                if not os.path.exists(background_path):
                    raise FileNotFoundError(f"Background dataset not found at {background_path}")
                    
                oe_data = Random300K_Images(
                    file_path=background_path,
                    transform=tf.Compose([
                        tf.RandomCrop(32, padding=4),
                        tf.RandomHorizontalFlip(), 
                        tf.ToTensor(),
                    ]),
                    extendable=options['noisy_ratio']
                )
                print(f"Loaded background dataset with {len(oe_data)} images")
                
                trainloader_oe = torch.utils.data.DataLoader(
                    oe_data, 
                    batch_size=options['batch_size'], 
                    shuffle=True,
                    num_workers=0,
                    drop_last=True
                )
                print(f"Background loader created with {len(trainloader_oe)} batches")
            except Exception as e:
                print(f"Warning: Failed to load background dataset: {str(e)}")
                trainloader_oe = None
                            
    else:
        print('No dataset chosen.')
    
    print("Outlier exposure mode is on. Other datasets")
    # oe_data = Random300K_Images(transform =tf.Compose(
    #     [tf.RandomCrop(32, padding=4),
    #     tf.RandomHorizontalFlip(), tf.ToTensor(),
    #     # tf.Normalize([0.519,0.511,0.508], [0.317,0.312,0.306]) ,
    #     ]),extendable=options['noisy_ratio'])
    if (options['noisy_ratio']):
        # oe_data.data = np.concatenate((Data.noisy_data,oe_data.data),axis=0)
        oe_data.data = oe_data.data[:30000]
        oe_data.data.extend(list(Data.noisy_data))
        print("#of background images {}".format(len(oe_data)))
        options['ramp_activate'] = True

    trainloader_oe = torch.utils.data.DataLoader(oe_data, batch_size=options['batch_size'], shuffle=True, num_workers = 0, drop_last=True)
    

    options['num_classes'] = Data.num_classes

    # Model
    print("Creating model: {}".format(options['model']))
    if options['model'] == 'classifier32':
        net = classifier32(num_classes=options['num_classes'])
        feat_dim = 128
    elif options['model'] == 'resnet50':
        net = resnet50(pretrained=True,num_classes=options['num_classes'])
        feat_dim = net.feat_dim
    elif options['model'] == 'vit':
        extractor = ViTExtractor(
        model_type='dino_vits8',  # ViT-S/8
        stride=8,
        device=device
        )
        img_size = 224
        net = extractor.model
        feat_dim = 384  # ViT-S feature dimension
        img_size = 224
        # Use 224x224 version of dataset
        Data = BloodMNIST_224_OSR(
            known=options['known'],
            dataroot=options['dataroot'],
            use_gpu=not options['use_cpu'],
            batch_size=options['batch_size']
        )
    else:
        raise 'model is not defined'
        
    # Loss
    options.update(
        {
            'feat_dim': feat_dim,
            'use_gpu':  use_gpu
        }
    )

    criterion = NirvanaOpenset_loss(num_classes=options['num_classes'], feat_dim = options['feat_dim'], precalc_centers = True, margin = options['margin'], Expand=options['Expand'])

    if use_gpu:
        net = net.cuda()
        criterion = criterion.cuda()

    dir_name = '{}_{}_{}_{}'.format(options['model'], options['loss'],options['margin'],options['oe'])
    model_path = os.path.join(options['outf'], 'models', options['dataset'],dir_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    
    file_name = '{}_{}_{}_{}_{}'.format(options['model'], options['loss'], options['item'], options['margin'],options['noisy_ratio'])

    if options['eval']:
        net, criterion = load_networks(net, model_path, file_name, criterion=criterion)
        results = test_ddfm_b9(net, criterion, testloader, outloader, epoch=0, **options)
        print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))
        return results

   
    optimizer = torch.optim.SGD(net.parameters(), lr=options['lr'], momentum=0.9, weight_decay=1e-4)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=options['max_epoch']*len(trainloader))
    start_time = time.time()
    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))
        if options['oe']:
            train_Nirvana_oe(net, criterion, optimizer,scheduler, trainloader, trainloader_oe, epoch=epoch, **options)
        else:
            train_Nirvana_oe(net, criterion, optimizer, scheduler,trainloader, epoch=epoch, **options)

        if options['eval_freq'] > 0 and (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch']:
            print("==> Test", options['loss'])
            results , results_b9= test_ddfm_b9(net, criterion, testloader, outloader, epoch=epoch, **options)
            print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))
            print("Normalized - Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results_b9['ACC'], results_b9['AUROC'], results_b9['OSCR']))
            avg_acc = (results['AUROC'] + results['OSCR'])/2.0
            
            if(avg_acc >= best_acc_avg):
                best_acc_avg = avg_acc
                results_best = results
                print("Best Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results_best['ACC'], results_best['AUROC'], results_best['OSCR']))
                save_networks(net, model_path, file_name, ext='best', criterion=criterion)
          
            avg_acc_b9 = (results_b9['AUROC'] + results_b9['OSCR'])/2.0
            if(avg_acc_b9 >= best_acc_avg_b9):
                best_acc_avg_b9 = avg_acc_b9
                results_b9_best = results_b9
                print("Normalized - Best Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results_b9_best['ACC'], results_b9_best['AUROC'], results_b9_best['OSCR']))
                save_networks(net, model_path, file_name, ext='best_b9', criterion=criterion)

            save_networks(net, model_path, file_name, criterion=criterion)
        
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
    return results_best, results_b9_best


if __name__ == '__main__':
    args = parser.parse_args()
    # Add after parsing args
    device = torch.device('cuda' if torch.cuda.is_available() and not args.use_cpu else 'cpu')
    print(f'Using device: {device}')
    options = vars(args)
    options['dataroot'] = os.path.join(options['dataroot'], options['dataset'])
    img_size = 32
    results = dict()
    results_b9 = dict()    

    for i in range(len(splits[options['dataset']])):
        known = splits[options['dataset']][len(splits[options['dataset']])-i-1]
        if options['dataset'] == 'bloodmnist':
            split_dict = splits['bloodmnist'][i]
            known = split_dict['known']
            unknown = split_dict['unknown']
            img_size = 32
            
            Data = BloodMNIST_OSR(
                known=known,
                dataroot=options['dataroot'],
                use_gpu=not options['use_cpu'],
                batch_size=options['batch_size']
            )
        else:
            unknown = list(set(list(range(0, 10))) - set(known))

        options.update(
            {
                'item':     i,
                'known':    known,
                'unknown':  unknown,
                'img_size': img_size
            }
        )

        dir_name = '{}_{}_{}_{}_{}'.format(options['model'], options['loss'],options['margin'],options['oe'],options['noisy_ratio'])
        dir_path = os.path.join(options['outf'], 'results', dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


        file_name = options['dataset'] + '.csv'
        file_name_b9 = options['dataset'] + '_b9.csv'

        #run experiment for this split
        res, res_b9 = main_worker(options)
        res['unknown'] = unknown
        res['known'] = known
        res_b9['unknown'] = unknown
        res_b9['known'] = known

        #save results
        results[str(i)] = res
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(dir_path, file_name))
        results_b9[str(i)] = res_b9
        df_b9 = pd.DataFrame(results_b9)
        df_b9.to_csv(os.path.join(dir_path, file_name_b9))

