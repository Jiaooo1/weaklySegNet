import os
import tempfile
import imageio 
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import skimage.morphology as morph
import scipy.ndimage as ndi
from PIL import Image
from tqdm import tqdm
from rich import print
from models import AffResUNet34
from utils import AverageMeter, save_results, split_forward_double, create_folder
from metrics import compute_metrics
from my_transforms import get_transforms
from skimage import measure
from options_fine import Options
import utils.utils as utils
from model.vmambabottleneck2 import VSSM

def run(opt):
    print('========== evaluating fine model ==========')
    #opt.save_options()

    img_dir = f"{opt.train['img_dir']}/test/"
    label_dir = opt.test['label_dir']
    save_dir = opt.train['save_dir']
    score_dir1 = save_dir + '/score_edge/'
    score_dir2 = save_dir + '/score_seg/'
    pred_dir = save_dir + '/pred/'
    create_folder(score_dir1)
    create_folder(score_dir2)
    create_folder(pred_dir)

    metric_names = ['acc', 'p_F1', 'p_recall', 'p_precision', 'dice', 'aji', 'dq', 'sq', 'pq']
    test_results = dict()
    all_result = AverageMeter(len(metric_names))

    #model = AffResUNet34()
    model = VSSM()
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.eval()

    model_path = 'D:/pycharm/weaklySegNet/data_for_train/MoNuSeg/s/checkpoints/checkpoint_0.pth.tar'
    print(f"=> loading trained model in {model_path}")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print("=> loaded model at epoch {}".format(checkpoint['epoch']))

    test_transform = get_transforms({'to_tensor': 1})

    img_names = os.listdir(img_dir)
    img_process = tqdm(img_names)
    for iname in img_process:
        img_name = iname[:-4] # remove .png
        img_process.set_description_str(f'=> Evaluating image {img_name}')

        #img = Image.open(img_dir + '/' + img_name + '.png') #原来
        img = Image.open(img_dir + '/' + img_name + '.tif')
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        ori_h = img.size[1]
        ori_w = img.size[0]

        if isinstance(img, Image.Image):
            print("img 是 PIL Image 对象")

        img = test_transform((img,))[0].unsqueeze(0)

        seg, edge = split_forward_double(model, img, opt.train['input_size'], opt.test['overlap'])
        
        seg = torch.sigmoid(seg).squeeze()
        edge = torch.sigmoid(edge).squeeze()

        output1 = edge.cpu().numpy()
        Image.fromarray((output1 * 255).astype(np.uint8), mode='L').save(f'{score_dir1}/{img_name}.png')
        output2 = seg.cpu().numpy()
        Image.fromarray((output2 * 255).astype(np.uint8), mode='L').save(f'{score_dir2}/{img_name}.png')

        diff = seg-edge
        diff = diff.clamp(min=0, max=1).cpu().numpy()

        pred = np.zeros(diff.shape)
        pred[diff > opt.test['thresh']] = 1

        pred_labeled = measure.label(pred)
        pred_labeled = morph.remove_small_objects(pred_labeled, opt.test['min_area'])
        pred_labeled = ndi.binary_fill_holes(pred_labeled > 0)
        pred_labeled = measure.label(pred_labeled)
        pred_labeled = morph.dilation(pred_labeled)

        # save pictures
        final_pred = Image.fromarray((pred_labeled * 65535).astype(np.uint16))
        final_pred.save('{:s}/{:s}_seg.tiff'.format(pred_dir, img_name))
        # save colored objects
        pred_colored_instance = np.zeros((ori_h, ori_w, 3))
        for k in range(1, pred_labeled.max() + 1):
            pred_colored_instance[pred_labeled == k, :] = np.array(utils.get_random_color())
        filename = '{:s}/{:s}_seg_colored.png'.format(pred_dir, img_name)
        imageio.imsave(filename, (pred_colored_instance * 255).astype(np.uint8))

        # save npy
        np.save(f'{pred_dir}/{img_name}', pred_labeled.astype(np.uint16))

        gt = np.load(f'{label_dir}/{img_name}.npy')
        metrics = compute_metrics(pred_labeled, gt, metric_names)

        test_results[img_name] = [metrics[mn] for mn in metric_names]

        all_result.update([metrics[mn] for mn in metric_names])

        print( 'Average ' + '\t'.join(f'{mn}: {res:.4f}' for mn,res in zip(metric_names, all_result.avg)) )

    header = metric_names
    save_results(header, all_result.avg, test_results, f'{opt.train["save_dir"]}/test_results_fine.txt')

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
