import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer.tester import Test
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    test_data_loader = config.init_obj('test_data_loader', module_data)


    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)

    # get function handles of loss and metrics
    criterion = [getattr(module_loss, loss) for loss in config['loss']]
    # criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']] # getattr( object,name[, default] ) 函数用于返回一个对象object的属性值name。

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

       
    trainer = Test(model, criterion, metrics, optimizer, 
                      config=config,
                      data_loader=None, test_data_loader=test_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='../models/SDCNN/0615_124649/MAYO_Val/config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default='../models/SDCNN/0615_124649/MAYO_Val/checkpoint-epoch65.pth', type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='0', type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-t', '--tag', default=None, type=str,
                      help='experience name in tensorboard (default: None)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)

    import pathlib
    from pathlib import Path
    pathlib.PosixPath = pathlib.WindowsPath
    main(config)
