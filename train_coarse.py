import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
import numpy as np
import random
from models import ResUNet34
from dataset import CoarseDataset, CoarseDataset1, CoarseDataset2
from options_coarse import Options
from rich import print
from tqdm import tqdm
from PIL import Image
from utils import save_checkpoint, save_bestcheckpoint, AverageMeter

from model.vmambabottleneck import VSSM
import time
import matplotlib.pyplot as plt

def run(opt):
    opt.save_options()

    #model = ResUNet34()
    model = VSSM()
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), opt.train['lr'], betas=(0.9, 0.99), weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.train['scheduler_step'], gamma=0.1)
    criterion = torch.nn.BCELoss().cuda()


    train_set = CoarseDataset1(img_dir=opt.train['img_dir'] + '/train/', vor_dir=opt.train['vor_dir'] + '/train/', mixed_dir=opt.train['mixed_dir'] + '/train/', aug=True)
    val_set = CoarseDataset1(img_dir=opt.train['img_dir'] + '/val_patch/', vor_dir=opt.train['vor_dir'] + '/val_patch/', mixed_dir=opt.train['mixed_dir'] + '/val_patch/', aug=False)


    train_loader = DataLoader(train_set, batch_size=opt.train['batch_size'], shuffle=True, num_workers=opt.train['workers'])
    val_loader = DataLoader(val_set, batch_size=opt.train['batch_size'], shuffle=False, num_workers=opt.train['workers'])
    
    num_epoch = opt.train['epochs']

    min_loss = 100

    Train_loss = []
    Val_loss = []
    start = time.time()

    for epoch in range(num_epoch):

        b = time.time()

        print('Epoch: [{:d}/{:d}]'.format(epoch+1, num_epoch))

        train_results = train1(train_loader, model, optimizer, criterion, opt)
        train_loss, train_loss_vor, train_loss_mixed = train_results


        state = {'epoch': epoch + 1, 'state_dict': model.state_dict()}

        cp_flag = (epoch + 1) % opt.train['checkpoint_freq'] == 0

        save_checkpoint(state, epoch, opt.train['save_dir'], cp_flag)

        scheduler.step()

        val_loss = val1(val_loader, model, criterion, opt)

        print('val_loss:', val_loss, 'min_loss:', min_loss)
        if val_loss < min_loss:
            min_loss = val_loss
            save_bestcheckpoint(state, opt.train['save_dir'])

        Train_loss.append(train_loss)
        Val_loss.append(val_loss)
        print('train time per epoch: {}'.format(time.time() - b))

    plt.figure()
    plt.plot(range(1, opt.train['epochs'] + 1), Train_loss, label='Training Loss', color='blue')
    plt.plot(range(1, opt.train['epochs'] + 1), Val_loss, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train_Val Loss')
    plt.legend()
    plt.grid(True)
    # 保存训练-验证损失图
    plt.savefig(opt.train['save_dir'] + '/loss_graph')
    plt.close()
    msg = 'total training time: {:.4f} hours'.format((time.time() - start) / 3600)
    print(msg)

def train(train_loader, model, optimizer, criterion, opt):
    results = AverageMeter(4)

    torch.cuda.empty_cache()
    model.train()

    for i, sample in enumerate(train_loader):
        img, vor, clu, po = sample
        img = img.float().cuda()
        vor = vor.float().cuda()
        clu = clu.float().cuda()
        po = po.float().cuda()

        out = model(img)
        out = torch.sigmoid(out)

        vor_mask = vor != 2
        clu_mask = clu != 2
        po_mask = po != 2

        loss_vor = criterion(out[vor_mask], vor[vor_mask])
        loss_clu = criterion(out[clu_mask], clu[clu_mask])
        loss_po = criterion(out[po_mask], po[po_mask])
        loss = loss_vor *  0.5 + loss_clu * 0.25 + loss_po * 0.25

        result = [loss.item(), loss_vor.item(), loss_clu.item(), loss_po.item()]
        results.update(result, img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % opt.train['log_freq'] == 0:
            print('Iteration: [{:d}/{:d}]'
                        '\tLoss {r[0]:.4f}'
                        '\tLoss_vor {r[1]:.4f}'
                        '\tLoss_clu {r[2]:.4f}'
                        '\tLoss_po {r[3]:.4f}'.format(i, len(train_loader), r=results.avg))


    print('\t=> Train Avg: Loss {r[0]:.4f}'
                '\tloss_vor {r[1]:.4f}'
                '\tloss_clu {r[2]:.4f}'
                '\tLoss_po {r[3]:.4f}'.format(r=results.avg))

    return results.avg

def train1(train_loader, model, optimizer, criterion, opt):
    results = AverageMeter(3)

    torch.cuda.empty_cache()
    model.train()

    for i, sample in enumerate(train_loader):
        img, vor, mixed = sample
        img = img.float().cuda()
        vor = vor.float().cuda()
        mixed = mixed.float().cuda()

        out = model(img)
        out = torch.sigmoid(out)

        vor_mask = vor != 2
        mixed_mask = mixed != 2

        loss_vor = criterion(out[vor_mask], vor[vor_mask])
        loss_mixed = criterion(out[mixed_mask], mixed[mixed_mask])
        loss = loss_vor * opt.train['weight'] + loss_mixed * (1-opt.train['weight'])

        result = [loss.item(), loss_vor.item(), loss_mixed.item()]
        results.update(result, img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % opt.train['log_freq'] == 0:
            print('Iteration: [{:d}/{:d}]'
                        '\tLoss {r[0]:.4f}'
                        '\tLoss_vor {r[1]:.4f}'
                        '\tLoss_mixed {r[2]:.4f}'.format(i, len(train_loader), r=results.avg))


    print('\t=> Train Avg: Loss {r[0]:.4f}'
                '\tloss_vor {r[1]:.4f}'
                '\tloss_mixed {r[2]:.4f}'.format(r=results.avg))

    return results.avg

def train2(train_loader, model, optimizer, criterion, opt):
    results = AverageMeter(3)

    torch.cuda.empty_cache()
    model.train()

    for i, sample in enumerate(train_loader):
        img, vor, clu = sample
        img = img.float().cuda()
        vor = vor.float().cuda()
        clu = clu.float().cuda()

        out = model(img)
        out = torch.sigmoid(out)

        vor_mask = vor != 2
        clu_mask = clu != 2

        loss_vor = criterion(out[vor_mask], vor[vor_mask])
        loss_clu = criterion(out[clu_mask], clu[clu_mask])
        loss = loss_vor * opt.train['weight'] + loss_clu * (1-opt.train['weight'])

        result = [loss.item(), loss_vor.item(), loss_clu.item()]
        results.update(result, img.size(0))

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if i % opt.train['log_freq'] == 0:
            print('Iteration: [{:d}/{:d}]'
                        '\tLoss {r[0]:.4f}'
                        '\tLoss_vor {r[1]:.4f}'
                        '\tLoss_clu {r[2]:.4f}'.format(i, len(train_loader), r=results.avg))


    print('\t=> Train Avg: Loss {r[0]:.4f}'
                '\tloss_vor {r[1]:.4f}'
                '\tloss_clu {r[2]:.4f}'.format(r=results.avg))

    return results.avg

def val(val_loader, model, criterion, opt):
    model.eval()
    results = AverageMeter(4)

    for i, sample in enumerate(val_loader):
        img, vor, clu, po = sample
        img = img.float().cuda()
        vor = vor.float().cuda()
        clu = clu.float().cuda()
        po = po.float().cuda()

        vor_mask = vor != 2
        clu_mask = clu != 2
        po_mask = po != 2
        
        if not vor_mask.any() or not clu_mask.any() or not po_mask.any():
            continue

        out = model(img)
        out = torch.sigmoid(out)

        loss_vor = criterion(out[vor_mask], vor[vor_mask])
        loss_clu = criterion(out[clu_mask], clu[clu_mask])
        loss_po = criterion(out[po_mask], po[po_mask])
        loss = loss_vor *  0.5 + loss_clu * 0.25 + loss_po * 0.25

        result = [loss.item(), loss_vor.item(), loss_clu.item(), loss_po.item()]
        results.update(result, img.size(0))

    val_loss = results.avg[0]
    return val_loss


def val1(val_loader, model, criterion, opt):
    model.eval()
    results = AverageMeter(3)
    for i, sample in enumerate(val_loader):
        img, vor, mixed = sample
        img = img.float().cuda()
        vor = vor.float().cuda()
        mixed = mixed.float().cuda()

        vor_mask = vor != 2
        mixed_mask = mixed != 2

        if not vor_mask.any() or not mixed_mask.any():
            continue

        out = model(img)
        out = torch.sigmoid(out)

        loss_vor = criterion(out[vor_mask], vor[vor_mask])
        loss_mixed = criterion(out[mixed_mask], mixed[mixed_mask])
        loss = loss_vor * opt.train['weight'] + loss_mixed * (1 - opt.train['weight'])

        result = [loss.item(), loss_vor.item(), loss_mixed.item()]
        results.update(result, img.size(0))

    val_loss = results.avg[0]
    return val_loss

def val2(val_loader, model, criterion, opt):
    model.eval()
    results = AverageMeter(3)
    for i, sample in enumerate(val_loader):
        img, vor, clu = sample
        img = img.float().cuda()
        vor = vor.float().cuda()
        clu = clu.float().cuda()

        vor_mask = vor != 2
        clu_mask = clu != 2

        if not vor_mask.any() or not clu_mask.any():
            continue

        out = model(img)
        out = torch.sigmoid(out)

        loss_vor = criterion(out[vor_mask], vor[vor_mask])
        loss_clu = criterion(out[clu_mask], clu[clu_mask])
        loss = loss_vor * opt.train['weight'] + loss_clu * (1 - opt.train['weight'])

        result = [loss.item(), loss_vor.item(), loss_clu.item()]
        results.update(result, img.size(0))

    val_loss = results.avg[0]
    return val_loss

if __name__ == '__main__':
    opt = Options()
    opt.parse()
    if opt.random_seed >= 0:
        print('=> Using random seed {:d}'.format(opt.random_seed))
        torch.manual_seed(opt.random_seed)
        torch.cuda.manual_seed(opt.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(opt.random_seed)
        random.seed(opt.random_seed)
    else:
        torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.gpus)
    run(opt)