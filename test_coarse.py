import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from rich import print
from models import ResUNet34
from utils import AverageMeter, save_results, split_forward_single, split_forward_single_uncertainty, create_folder
from metrics import compute_metrics
from my_transforms import get_transforms
import torch
import torch.backends.cudnn as cudnn
import skimage.morphology as morph
import scipy.ndimage as ndi
from skimage import measure
import imageio 
import random
#from options_plo import Options  # 新加的 测试不确定性模型
from options_coarse import Options
import utils.utils as utils
from model.vmambabottleneck import VSSM
import torch.nn.functional as F


def run(opt):
    print('========== evaluating coarse model ==========')
    opt.save_options()

    img_dir = f"{opt.train['img_dir']}/test/"
    label_dir = opt.test['label_dir']
    save_dir = opt.train['save_dir']
    pred_dir = save_dir + '/pred/'
    score_dir = save_dir + '/score/'
    create_folder(pred_dir)
    create_folder(score_dir)

    metric_names = ['acc', 'p_F1', 'p_recall', 'p_precision', 'dice', 'aji', 'dq', 'sq', 'pq']
    test_results = dict()
    all_result = AverageMeter(len(metric_names))

    model = VSSM()
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.eval()
    cudnn.benchmark = True

    model_path = f'{opt.train["save_dir"]}/checkpoints/checkpoint_{opt.test["test_epoch"]}.pth.tar'
    print(f"=> loading trained model in {model_path}")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model at epoch {}".format(checkpoint['epoch']))

    test_transform = get_transforms({'to_tensor': 1})

    img_names = os.listdir(img_dir)
    img_process = tqdm(img_names)
    for iname in img_process:
        img_name = iname[:-4] # remove .png
        img_process.set_description_str(f'=> Evaluating image {img_name}')

        #img = Image.open(img_dir + '/' + img_name + '.png')
        img = Image.open(img_dir + '/' + img_name + '.tif')  # 1000×1000
        ori_h = img.size[1]
        ori_w = img.size[0]
        img = test_transform((img,))[0].unsqueeze(0)

        output = split_forward_single(model, img, opt.train['input_size'], opt.test['overlap']).squeeze()
        output = torch.sigmoid(output).cpu().numpy()
        Image.fromarray((output * 255).astype(np.uint8), mode='L').save(f'{score_dir}/{img_name}.png')
        pred = np.zeros(output.shape)
        pred[output > opt.test['thresh']] = 1


        pred_labeled = measure.label(pred)
        pred_labeled = morph.remove_small_objects(pred_labeled, opt.test['min_area'])
        pred_labeled = ndi.binary_fill_holes(pred_labeled > 0)
        pred_labeled = measure.label(pred_labeled)

        # save pictures
        final_pred = Image.fromarray((pred_labeled * 65535).astype(np.uint16))
        final_pred.save('{:s}/{:s}_seg.tiff'.format(pred_dir, img_name))
        # save colored objects
        pred_colored_instance = np.zeros((ori_h, ori_w, 3))
        for k in range(1, pred_labeled.max() + 1):
            pred_colored_instance[pred_labeled == k, :] = np.array(utils.get_random_color())
        filename = '{:s}/{:s}_seg_colored.png'.format(pred_dir, img_name)
        imageio.imsave(filename, (pred_colored_instance * 255).astype(np.uint8))
        
        np.save(f'{pred_dir}/{img_name}', pred_labeled.astype(np.uint16))

        print('label_dir:',label_dir)

        gt = np.load(f'{label_dir}/{img_name}.npy')
        metrics = compute_metrics(pred_labeled, gt, metric_names)

        test_results[img_name] = [metrics[mn] for mn in metric_names]

        all_result.update([metrics[mn] for mn in metric_names])

        print( 'Average ' + '\t'.join(f'{mn}: {res:.4f}' for mn,res in zip(metric_names, all_result.avg)) )

    header = metric_names
    save_results(header, all_result.avg, test_results, f'{opt.train["save_dir"]}/test_results_coarse.txt')


if __name__ == '__main__':
    opt = Options()
    opt.with_uncertainty = False
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
