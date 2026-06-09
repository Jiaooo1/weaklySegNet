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
# from options_plo import Options  # 新加的 测试不确定性模型
from options_coarse import Options
import utils.utils as utils
from model.vmambabottleneck import VSSM
import torch.nn.functional as F


def run(opt):
    opt.save_options()


    img_dir = ''
    label_instance_dir = ''
    vor_dir = ''
    cluster_dir = ''
    po_dir = ''
    save_dir = ''


    metric_names = ['acc', 'p_F1', 'p_recall', 'p_precision', 'dice', 'aji', 'dq', 'sq', 'pq']
    test_results = dict()
    all_result = AverageMeter(len(metric_names))


    img_names = os.listdir(cluster_dir)
    img_process = tqdm(img_names)
    for iname in img_process:
        img_name = iname[:-18]  # remove .tif
        img_process.set_description_str(f'=> Evaluating image {img_name}')

        gt = np.load(f'{label_instance_dir}/{img_name}.npy')

        clu = Image.open(cluster_dir + '/' + img_name + '_label_cluster.png')
        po = Image.open(po_dir + '/' + img_name + '_label_po.png')
        vor = Image.open(vor_dir + '/' + img_name + '_label_vor.png')
        clu_array = np.array(clu)
        po_array = np.array(po)
        vor_array = np.array(vor)

        binary_clu_array = np.ones((clu_array.shape[0], clu_array.shape[1]), dtype=np.uint8) * 2
        binary_clu_array[clu_array[:, :, 0] == 255] = 0
        binary_clu_array[clu_array[:, :, 1] == 255] = 1

        binary_po_array = np.ones((po_array.shape[0], po_array.shape[1]), dtype=np.uint8) * 2
        binary_po_array[po_array[:, :, 0] == 255] = 0
        binary_po_array[po_array[:, :, 1] == 255] = 1

        binary_vor_array = np.ones((vor_array.shape[0], vor_array.shape[1]), dtype=np.uint8) * 2
        binary_vor_array[vor_array[:, :, 0] == 255] = 0
        binary_vor_array[vor_array[:, :, 1] == 255] = 1


        print(f"\nGenerate mixed pseudo labels {img_name}:")
        merged = np.zeros_like(binary_clu_array)
        merged[(binary_clu_array == 2) | (binary_po_array == 2)] = 2
        merged[(binary_clu_array == 1) | (binary_po_array == 1)] = 1


        optimized = merged.copy()


        save_npy_path = os.path.join(save_dir, img_name + '.npy')
        np.save(save_npy_path, optimized)
        save_path = os.path.join(save_dir, img_name + '_label_mixed.png')
        save_pseudo_label_as_png(optimized, save_path)



        binary_array = merged.copy()

        binary_array[binary_array == 2] = 0

        binary_array = measure.label(binary_array)


        metrics = compute_metrics(binary_array, gt, metric_names)
        test_results[img_name] = [metrics[mn] for mn in metric_names]
        all_result.update([metrics[mn] for mn in metric_names])
        print(f"\nEvaluating image {img_name}:")
        print('Average ' + '\t'.join(f'{mn}: {res:.4f}' for mn, res in zip(metric_names, all_result.avg)))

        test_results[img_name] = [metrics[mn] for mn in metric_names]

        all_result.update([metrics[mn] for mn in metric_names])

        print('Average ' + '\t'.join(f'{mn}: {res:.4f}' for mn, res in zip(metric_names, all_result.avg)))

    header = metric_names
    save_results(header, all_result.avg, test_results, f'{save_dir}/test_results.txt')


def save_pseudo_label_as_png(pseudo_label, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)


    h, w = pseudo_label.shape
    png_image = np.zeros((h, w, 3), dtype=np.uint8)

    background_mask = (pseudo_label == 0)
    png_image[background_mask] = [255, 0, 0]

    nuclei_mask = (pseudo_label == 1)
    png_image[nuclei_mask] = [0, 255, 0]

    vor_boundary_mask = (pseudo_label == 2)
    png_image[vor_boundary_mask] = [0, 0, 0]

    imageio.imwrite(save_path, png_image)

    print(f"\nSaved PNG label to: {save_path}")
    return save_path


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
