import utils
import argparse
from collections import OrderedDict
import numpy as np

class Options:
    def __init__(self):
        self.dataset = "MoNuSeg"
        self.ratio = 2.00
        self.description = "_"
        self.data_dir = ''  # YOUR DATA PATH
        self.label_dir = ''  # YOUR LABEL PATH
        self.save_dir = ''  # YOUR SAVE PATH
        self.gpus = [0]
        self.random_seed = 0
        self.T = 20  # number of MC dropout sampling #新加的


    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default=self.dataset)
        parser.add_argument('--ratio', type=float, default=self.ratio)
        parser.add_argument('--description', type=str, default=self.description)
        parser.add_argument('--data-dir', type=str, default=self.data_dir)
        parser.add_argument('--label-dir', type=str, default=self.label_dir)
        parser.add_argument('--save-dir', type=str, default=self.save_dir)
        parser.add_argument('--gpus', type=int, nargs='+', default=self.gpus)
        parser.add_argument('--random-seed', type=int, default=self.random_seed)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--epochs', type=int, default=100) #原来为100
        #parser.add_argument('--batch-size', type=int, default=32) #原来
        parser.add_argument('--batch-size', type=int, default=8)
        parser.add_argument('--threshold', type=float, default=0.5)
        args = parser.parse_args()

        self.dataset = args.dataset
        self.ratio = args.ratio
        self.description = args.description
        self.data_dir = args.data_dir
        self.label_dir = args.label_dir
        self.save_dir = args.save_dir
        self.gpus = args.gpus
        self.random_seed = args.random_seed

        self.train = {}



        self.train['img_dir'] = f'{args.data_dir}/{args.dataset}/images'
        self.train['img_dir2'] = f'{args.data_dir}/{args.dataset}/images2'
        self.train['vor_dir'] = f'{args.data_dir}/{args.dataset}/labels_voronoi'
        self.train['clu_dir'] = f'{args.data_dir}/{args.dataset}/labels_cluster'
        self.train['po_dir'] = f'{args.data_dir}/{args.dataset}/labels_po'
        self.train['mixed_dir'] = f'{args.data_dir}/{args.dataset}/labels_mixed'
        self.train['lab_dir'] = f'{args.data_dir}/{args.dataset}/labels_lab/'
        self.train['aff_dir'] = f'{args.data_dir}/{args.dataset}/labels_aff/'
        self.train['save_dir'] = f'{args.save_dir}/{args.dataset}/s'


        self.train['input_size'] = 224
        self.train['epochs'] = args.epochs
        self.train['lr'] = args.lr
        self.train['batch_size'] = args.batch_size
        self.train['scheduler_step'] = 30
        self.train['checkpoint_freq'] = 999999
        self.train['log_freq'] = 999999
        self.train['workers'] = 4
        self.train['weight'] = 0.5


        self.test = {}
        self.test['label_dir'] = args.label_dir
        self.test['thresh'] = args.threshold
        self.test['test_epoch'] = 0
        self.test['min_area'] = 20 
        self.test['overlap'] = 80
        self.test['patch_size'] = 224  # 新加的


    def save_options(self):
        filename = f'{self.save_dir}/{self.dataset}/s/options.txt'
        with open(filename, 'w') as file:
            file.write("# ---------- Options ---------- #\n")
            for group, options in self.__dict__.items():
                if type(options) == dict:
                    file.write('\n\n-------- {:s} --------\n'.format(group))
                    for k,v in options.items():
                        file.write(f'{k}: {v}\n')
                else:
                    file.write(f'{group}: {str(options)}\n')
