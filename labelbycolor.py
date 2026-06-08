import imageio
import os
import shutil
import numpy as np
from skimage import morphology, measure
from sklearn.cluster import KMeans
from scipy.ndimage.morphology import distance_transform_edt as dist_tranform
import glob
import json
import cv2
import torch
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import warnings
import scipy.ndimage as ndi
import skimage.morphology as morph
from scipy.spatial.distance import cdist
import math
from metrics import compute_metrics
from utils import AverageMeter, save_results

warnings.filterwarnings('ignore')


def main():
    dataset = 'MoNuSeg'
    data_dir = './{:s}'.format(dataset)
    img_dir = './{:s}/images'.format(dataset)
    label_instance_dir = './{:s}/labels_instance'.format(dataset)
    label_instance_png_dir = './{:s}/labels_instance_png'.format(dataset)
    label_point_dir = './{:s}/labels_point'.format(dataset)
    label_vor_dir = './{:s}/labels_voronoi'.format(dataset)
    label_cluster_dir = './{:s}/labels_cluster'.format(dataset)
    label_po_dir = './{:s}/labels_po'.format(dataset)
    create_folder(label_po_dir)


    with open('{:s}/train_val_test_MO.json'.format(data_dir), 'r') as file:
        data_list = json.load(file)
        train_list = data_list['train']
        val_list = data_list['val']
        test_list = data_list['test']


    combined_list = train_list+val_list+test_list


    create_po_label(img_dir, label_point_dir, label_vor_dir, label_cluster_dir, label_instance_dir,


def create_folder(folder_path):
    """创建文件夹"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def create_po_label(img_dir, point_dir, vor_dir, cluster_dir, label_instance_dir, label_instance_png_dir, save_dir,
                    combined_list):
    """创建伪标签"""
    print("Generating label...")


    img_list = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
    #img_list = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    N_total = len([name for name in img_list if name.split('.')[0] in combined_list])
    N_processed = 0

    metric_names = ['acc', 'p_F1', 'p_recall', 'p_precision', 'dice', 'aji', 'dq', 'sq', 'pq']
    test_results = dict()
    all_result = AverageMeter(len(metric_names))

    for img_name in img_list:
        name = img_name.split('.')[0]
        if name not in combined_list:
            continue

        N_processed += 1
        print(f'\r\t{N_processed}/{N_total}: Processing {img_name}', end='')


        img_path = os.path.join(img_dir, img_name)
        image = imageio.imread(img_path)


        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 4:

            image = image[:, :, :3]


        point_path = os.path.join(point_dir, name + '_label_point.png')
        point_mask = imageio.imread(point_path)


        if len(point_mask.shape) == 3:
            point_mask = point_mask[:, :, 0]
        point_mask = (point_mask > 0).astype(np.uint8)


        vor_path = os.path.join(vor_dir, name + '_label_vor.png')
        vor_mask = imageio.imread(vor_path)
        vor_mask1 = imageio.imread(vor_path)

        if len(vor_mask.shape) == 3:
            vor_mask = vor_mask[:, :, 0]
        vor_mask = (vor_mask > 0).astype(np.uint8)


        cluster_path = os.path.join(cluster_dir, name + '_label_cluster.png')
        cluster_mask = imageio.imread(cluster_path)

        gt = np.load(f'{label_instance_dir}/{name}.npy')
        gt_png_path = os.path.join(label_instance_png_dir, name + '.png')
        gt_mask = imageio.imread(gt_png_path)

        nuclei_centers = get_nuclei_centers(point_mask)
        if len(nuclei_centers) >= 1000:
            expanded_points = expand_points(point_mask, radius=5)
        else:
            expanded_points = expand_points(point_mask, radius=6)  # 扩展点标注（半径5个像素）


        nuclei_stats = calculate_color_statistics1(image, expanded_points, point_mask)

        cell_nuclei_means = [stats[0] for stats in nuclei_stats]  # 提取平均值
        average_color_difference_a = calculate_average_color_difference(cell_nuclei_means)

        uniformity_scores, color_variances, pixel_diff_avgs= calculate_color_uniformity(image, expanded_points, point_mask, average_color_difference_a)
        print(f"\nImage {name}: Mean uniformity = {np.mean(uniformity_scores):.3f}, "
              f"Avg color diff a = {average_color_difference_a:.3f}")


        unoptimized_label, uncertain_label = generate_unoptimized_pseudo_label(image, point_mask, nuclei_stats, uniformity_scores, pixel_diff_avgs)

        optimized_label = generate_optimized_pseudo_label(unoptimized_label, uncertain_label, vor_mask)

        optimized_label_copy = optimized_label.copy()
        optimized_label_copy[optimized_label_copy == 2] = 0
        optimized_label_diffrent_id = measure.label(optimized_label_copy)

        metrics = compute_metrics(optimized_label_diffrent_id, gt, metric_names)
        test_results[img_name] = [metrics[mn] for mn in metric_names]
        all_result.update([metrics[mn] for mn in metric_names])
        print(f"\nEvaluating image {name}:")
        print('Average ' + '\t'.join(f'{mn}: {res:.4f}' for mn, res in zip(metric_names, all_result.avg)))


        save_unopt_npy_path = os.path.join(save_dir, name + '_unoptimized.npy')
        np.save(save_unopt_npy_path, unoptimized_label)

        save_unopt_png_path = os.path.join(save_dir, name + '_unoptimized.png')
        save_pseudo_label_as_png(unoptimized_label, save_unopt_png_path, vor_mask=None)

        save_opt_npy_path = os.path.join(save_dir, name + '.npy')
        np.save(save_opt_npy_path, optimized_label)

        save_opt_png_path = os.path.join(save_dir, name + '_label_po.png')
        save_pseudo_label_as_png(optimized_label, save_opt_png_path, vor_mask)

        save_binary_path = os.path.join(save_dir, name + '_binary.png')
        save_binary_label(optimized_label, save_binary_path)

        generate_comparison(image, gt_mask, unoptimized_label, optimized_label, vor_mask, vor_mask1, cluster_mask, name, save_dir)

        save_individual_plots(image, gt_mask, unoptimized_label, optimized_label, vor_mask, name, save_dir)

    header = metric_names
    save_results(header, all_result.avg, test_results, f'{save_dir}/evaluation_results.txt')
    print("\nDone!")


def expand_points(point_mask, radius=5):
    """扩展点标注"""
    from scipy.ndimage import binary_dilation

    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    structure = x ** 2 + y ** 2 <= radius ** 2

    expanded = binary_dilation(point_mask, structure=structure)

    return expanded


def calculate_color_stats(image, expanded_mask1, expanded_mask2, point_mask):
    """计算颜色统计信息 - 处理RGB图像，确保每个细胞核都有平均值"""
    nuclei_centers = get_nuclei_centers(point_mask)

    labeled_mask = measure.label(expanded_mask1)

    nuclei_means = []

    for idx, center in enumerate(nuclei_centers):
        center_y, center_x = center

        if 0 <= center_y < labeled_mask.shape[0] and 0 <= center_x < labeled_mask.shape[1] and labeled_mask[
            center_y, center_x] > 0:
            region_label = labeled_mask[center_y, center_x]

            region_found = False
            for region in measure.regionprops(labeled_mask):
                if region.label == region_label:

                    minr, minc, maxr, maxc = region.bbox
                    region_mask = labeled_mask[minr:maxr, minc:maxc] == region.label


                    region_colors = image[minr:maxr, minc:maxc][region_mask]


                    if len(region_colors) > 0:
                        mean_color = np.mean(region_colors, axis=0)
                    else:
                        mean_color = np.array([128, 128, 128])  # 默认值

                    nuclei_means.append(mean_color)
                    region_found = True
                    break

            if not region_found:

                nuclei_means.append(np.array([128, 128, 128]))
        else:

            nuclei_means.append(np.array([128, 128, 128]))


    bg_mask = ~expanded_mask2
    bg_colors = image[bg_mask]
    bg_mean = np.mean(bg_colors, axis=0) if len(bg_colors) > 0 else np.array([128, 128, 128])

    return nuclei_means, bg_mean



def calculate_color_statistics1(image, expanded_mask, point_mask):
    h, w = image.shape[:2]


    nuclei_centers = get_nuclei_centers(point_mask)


    labeled_mask = measure.label(expanded_mask)


    nuclei_stats = []

    for idx, center in enumerate(nuclei_centers):
        center_y, center_x = center


        if 0 <= center_y < h and 0 <= center_x < w and labeled_mask[center_y, center_x] > 0:

            region_label = labeled_mask[center_y, center_x]


            region_mask = (labeled_mask == region_label)


            region_colors = image[region_mask]

            if len(region_colors) > 0:

                mean_color = np.mean(region_colors, axis=0)
                var_color = np.var(region_colors, axis=0)
                nuclei_stats.append((mean_color, var_color))
            else:

                nuclei_stats.append((np.array([128, 128, 128]), np.array([100, 100, 100])))
        else:

            nuclei_stats.append((np.array([128, 128, 128]), np.array([100, 100, 100])))

    return nuclei_stats

def find_voronoi_boundary(vor_mask):
    from scipy.ndimage import binary_dilation, binary_erosion


    dilated = binary_dilation(vor_mask > 0)
    eroded = binary_erosion(vor_mask > 0)


    boundaries = dilated & ~eroded

    return boundaries.astype(np.uint8)


def calculate_average_color_difference(nuclei_means):

    if len(nuclei_means) < 2:
        return 30.0

    differences = []
    n = len(nuclei_means)


    for i in range(n):
        for j in range(i + 1, n):

            diff = np.linalg.norm(nuclei_means[i] - nuclei_means[j])
            differences.append(diff)

    if differences:
        avg_difference = np.mean(differences)
        return avg_difference
    else:
        return 30.0


class AverageMeter(object):

    def __init__(self, shape=1):
        self.shape = shape
        self.reset()

    def reset(self):
        self.val = np.zeros(self.shape)
        self.avg = np.zeros(self.shape)
        self.sum = np.zeros(self.shape)
        self.count = 0

    def update(self, val, n=1):
        val = np.array(val)
        assert val.shape == self.val.shape
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def generate_unoptimized_pseudo_label(image, point_mask, nuclei_stats, uniformity_scores, pixel_diff_avgs):

    h, w = image.shape[:2]


    nuclei_centers = get_nuclei_centers(point_mask)


    if len(nuclei_centers) >= 1000:
        expanded_points1 = expand_points(point_mask, radius=5)
    else:
        expanded_points1 = expand_points(point_mask, radius=6)
    expanded_points2 = expand_points(point_mask, radius=15)



    nuclei_means, bg_mean = calculate_color_stats(image, expanded_points1, expanded_points2, point_mask)


    nuclei_pairwise_diffs = []
    for i in range(len(nuclei_means)):
        diffs = []
        for j in range(len(nuclei_means)):
            if i != j:
                diff = np.mean(np.abs(np.array(nuclei_means[i]) - np.array(nuclei_means[j])))
                diffs.append(diff)
        if diffs:
            nuclei_pairwise_diffs.append(np.mean(diffs))
        else:
            nuclei_pairwise_diffs.append(30.0)


    nuclei_bg_diffs = []
    for i in range(len(nuclei_means)):
        diff = np.mean(np.abs(np.array(nuclei_means[i]) - np.array(bg_mean)))
        nuclei_bg_diffs.append(diff)


    b_values = []
    for i in range(len(nuclei_means)):
        for j in range(i + 1, len(nuclei_means)):
            diff = np.mean(np.abs(np.array(nuclei_means[i]) - np.array(nuclei_means[j])))
            b_values.append(diff)
    if b_values:
        b2 = np.mean(b_values)
    else:
        b2 = 30.0


    if nuclei_means:
        nuclei_avg_rgb = np.mean([np.array(m) for m in nuclei_means], axis=0)
        nuclei_avg = np.mean(nuclei_avg_rgb)
        bg_avg = np.mean(bg_mean)
        c2 = np.abs(nuclei_avg - bg_avg)
    else:
        c2 = 60.0


    print(f"\nProcessing {len(nuclei_centers)} nuclei with adaptive thresholds...")
    print('len(nuclei_centers):', len(nuclei_centers), 'len(nuclei_means):',len(nuclei_means),'len(uniformity_scores):',len(uniformity_scores), 'len(nuclei_stats):', len(nuclei_stats))


    all_polygons1 = []
    all_polygons2 = []
    for idx, center in enumerate(nuclei_centers):

        if idx< len(uniformity_scores):
            current_uniformity = uniformity_scores[idx]
            current_diff = pixel_diff_avgs[idx]
            b1 = nuclei_pairwise_diffs[idx]
            c1 = nuclei_bg_diffs[idx]
            diff_bg_threshold1, diff_bg_threshold2, similarity_threshold1, similarity_threshold2, diff_nucleus_threshold= get_uniformity_threshold(current_uniformity, b1, b2, c1, c2, len(nuclei_centers))
        else:
            current_uniformity = 0.5
            current_diff = 10
            diff_bg_threshold1 = 10
            diff_bg_threshold2 = 8
            similarity_threshold1 = 2.5
            similarity_threshold2 = 2.7
            diff_nucleus_threshold = 10

        if idx < len(nuclei_stats):
            nucleus_stats = nuclei_stats[idx]
        else:

            nucleus_stats = (np.array([128, 128, 128]), np.array([100, 100, 100]))

        nuclei_mean, nuclei_var = nucleus_stats
        polygon1 = generate_nucleus_polygon1(image, center, nucleus_stats, nuclei_means, bg_mean, diff_bg_threshold1, similarity_threshold1, diff_nucleus_threshold, len(nuclei_centers))
        polygon2 = generate_nucleus_polygon2(image, center, nucleus_stats,  bg_mean, diff_bg_threshold2, similarity_threshold2)
        all_polygons1.append(polygon1)
        all_polygons2.append(polygon2)
        print(f"\r\tNucleus {idx + 1}: 当前细胞核与其他细胞核平均差异值={b1:.3f}, 细胞核之间的平均差异={b2:.3f}, 当前细胞核与背景的差异值={c1:.3f}, 所有细胞核与背景颜色差异性={c2:.3f}, "
              f"当前细胞核颜色均匀性={current_uniformity:.3f}", end='')



    pseudo_mask1 = create_pseudo_mask_from_polygons(all_polygons1, h, w)
    pseudo_mask2 = create_pseudo_mask_from_polygons(all_polygons2, h, w)

    return pseudo_mask1, pseudo_mask2


def generate_nucleus_polygon1(image, center, nucleus_stats, nuclei_means, bg_mean, diff_bg_threshold, similarity_threshold, diff_nucleus_threshold, num):

    h, w = image.shape[:2]
    y, x = center


    nucleus_mean, nucleus_var = nucleus_stats


    directions = [(0, 1), (1, 1), (1, 0), (1, -1),
                  (0, -1), (-1, -1), (-1, 0), (-1, 1)]


    point_set = []
    polygon_points = []


    for i, (dy, dx) in enumerate(directions):
        nx, ny = x + 1 * dx, y + 1 * dy


        nx = max(0, min(w - 1, nx))
        ny = max(0, min(h - 1, ny))

        point_set.append((nx, ny))


    polygon_points = point_set.copy()


    init_polygon = Polygon(polygon_points)
    avg_color, color = calculate_polygon_color(image, init_polygon)



    max_radius = 30
    changed = True


    for radius in range(2, max_radius):

        if not changed:
            break

        changed = False

        for i in range(len(directions)):
            dy, dx = directions[i]


            nx, ny = x + radius * dx, y + radius * dy


            nx = max(0, min(w - 1, nx))
            ny = max(0, min(h - 1, ny))


            new_point_set = point_set.copy()
            new_point_set[i] = (nx, ny)


            new_polygon = Polygon(new_point_set)


            if init_polygon.area > 0:
                diff_polygon = new_polygon.difference(init_polygon)
                if diff_polygon.area > 0:

                    diff_avg, color = calculate_polygon_color(image, diff_polygon)


                    diffs_candidate = []
                    for j in range(len(nuclei_means)):
                        if not np.array_equal(np.array(nucleus_mean), np.array(nuclei_means[j])):
                            diff_candidate = np.mean(np.abs(np.array(nuclei_means[j]) - diff_avg))
                            diffs_candidate.append(diff_candidate)

                    if diffs_candidate:
                        diff_nucleus=(np.mean(diffs_candidate))
                    else:
                        diff_nucleus=30.0


                    diff_bg = np.mean(np.abs(bg_mean - diff_avg))

                    d_nucleus = calculate_normalized_euclidean_distance(diff_avg, nucleus_mean, nucleus_var)

                    if diff_bg >= diff_bg_threshold and d_nucleus <= similarity_threshold and diff_nucleus <= diff_nucleus_threshold :

                        point_set = new_point_set
                        init_polygon = new_polygon
                        avg_color, color = calculate_polygon_color(image, init_polygon)
                        changed = True


    if len(polygon_points) < 3 or init_polygon.area < 90:
        circle_points = []
        for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            if num >= 1000:
                cx = x + 8 * np.cos(angle)
                cy = y + 8 * np.sin(angle)
            else:
                cx = x + 10 * np.cos(angle)
                cy = y + 10 * np.sin(angle)
            circle_points.append((cx, cy))
        final_polygon = Polygon(circle_points)

    else:

        smoothed_points = smooth_polygon(init_polygon)
        final_polygon= Polygon(smoothed_points)

    return final_polygon


def generate_nucleus_polygon2(image, center, nucleus_stats, bg_mean, diff_bg_threshold, similarity_threshold):

    h, w = image.shape[:2]
    y, x = center


    nucleus_mean, nucleus_var = nucleus_stats


    directions = [(0, 1), (1, 1), (1, 0), (1, -1),
                  (0, -1), (-1, -1), (-1, 0), (-1, 1)]


    point_set = []
    polygon_points = []


    for i, (dy, dx) in enumerate(directions):
        nx, ny = x + 1 * dx, y + 1 * dy


        nx = max(0, min(w - 1, nx))
        ny = max(0, min(h - 1, ny))

        point_set.append((nx, ny))


    polygon_points = point_set.copy()
    init_polygon = Polygon(polygon_points)


    max_radius = 30
    changed = True


    for radius in range(2, max_radius):

        if not changed:
            break

        changed = False

        for i in range(len(directions)):
            dy, dx = directions[i]


            nx, ny = x + radius * dx, y + radius * dy


            nx = max(0, min(w - 1, nx))
            ny = max(0, min(h - 1, ny))


            new_point_set = point_set.copy()
            new_point_set[i] = (nx, ny)


            new_polygon = Polygon(new_point_set)


            if init_polygon.area > 0:
                diff_polygon = new_polygon.difference(init_polygon)
                if diff_polygon.area > 0:

                    diff_avg, color = calculate_polygon_color(image, diff_polygon)


                    diff_bg = np.mean(np.abs(bg_mean - diff_avg))


                    d_nucleus = calculate_normalized_euclidean_distance(diff_avg, nucleus_mean, nucleus_var)

                    if diff_bg >= diff_bg_threshold and d_nucleus <= similarity_threshold:

                        point_set = new_point_set
                        init_polygon = new_polygon
                        changed = True


    smoothed_points = smooth_polygon(init_polygon)
    final_polygon = Polygon(smoothed_points)

    return final_polygon


def calculate_color_uniformity(image, expanded_mask, point_mask, average_color_difference_a):

    h, w = image.shape[:2]


    nuclei_centers = get_nuclei_centers(point_mask)


    labeled_mask = measure.label(expanded_mask)

    uniformity_scores = []
    color_variances = []
    pixel_diff_avgs = []

    for idx, center in enumerate(nuclei_centers):
        center_y, center_x = center


        if 0 <= center_y < h and 0 <= center_x < w and labeled_mask[center_y, center_x] > 0:

            region_label = labeled_mask[center_y, center_x]


            region_mask = (labeled_mask == region_label)


            region_colors = image[region_mask]

            if len(region_colors) > 0:

                channel_variances = np.var(region_colors, axis=0)  # [R_var, G_var, B_var]
                avg_channel_variance = np.mean(channel_variances)


                max_possible_std = 255 / np.sqrt(12)
                actual_std = np.sqrt(avg_channel_variance)
                uniformity_score = 1 - (actual_std / max_possible_std)
                uniformity_score = np.clip(uniformity_score, 0, 1)


                if len(region_colors) > 1:

                    mean_color = np.mean(region_colors, axis=0)

                    pixel_diffs = np.mean(np.abs(region_colors - mean_color), axis=1)

                    pixel_diff_avg = np.mean(pixel_diffs)
                else:

                    pixel_diff_avg = 0.0

                uniformity_scores.append(uniformity_score)
                color_variances.append(avg_channel_variance)

                pixel_diff_avgs.append(pixel_diff_avg)

            else:

                uniformity_scores.append(0.75)
                color_variances.append(0.0)
                pixel_diff_avgs.append(0.0)
        else:

            uniformity_scores.append(0.75)
            color_variances.append(0.0)
            pixel_diff_avgs.append(average_color_difference_a)

    return uniformity_scores, color_variances, pixel_diff_avgs


def get_uniformity_threshold(uniformity_score, b1, b2, c1, c2, num):


    if c1 >= c2:
        similarity_threshold1 = 2.7
        diff_bg_threshold = uniformity_score * c2
    else:
        similarity_threshold1 = 2.3
        diff_bg_threshold = uniformity_score * c1

    similarity_threshold2 = similarity_threshold1 + 0.4

    diff_bg_threshold2 = diff_bg_threshold - 0.4

    if b1 >= b2:
        diff_nucleus_threshold =  b1
    else:
        diff_nucleus_threshold =  b2

    return diff_bg_threshold, diff_bg_threshold2, similarity_threshold1, similarity_threshold2, diff_nucleus_threshold



def calculate_polygon_color_average(image, polygon):
    h, w = image.shape[:2]


    mask = np.zeros((h, w), dtype=np.uint8)


    if polygon.geom_type == 'Polygon':
        exterior = polygon.exterior
        if exterior:

            pts = []
            for coord in exterior.coords:
                x = int(max(0, min(w - 1, coord[0])))
                y = int(max(0, min(h - 1, coord[1])))
                pts.append([x, y])

            if len(pts) >= 3:
                pts_array = np.array(pts).reshape((-1, 1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [pts_array], 1)

    colors = image[mask == 1]
    if len(colors) > 0:
        return np.mean(colors, axis=0)
    else:
        return None


def calculate_normalized_euclidean_distance(color_vector, mean_vector, var_vector):
    epsilon = 1e-10
    adjusted_var = var_vector + epsilon

    normalized_diff = (color_vector - mean_vector) ** 2 / adjusted_var

    distance = np.sqrt(np.mean(normalized_diff))

    return distance

def calculate_polygon_color(image, polygon):
    h, w = image.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    if polygon.geom_type == 'Polygon':
        exterior = polygon.exterior
        if exterior:
            pts = np.array(exterior.coords).reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)

    colors = image[mask == 1]
    if len(colors) > 0:
        return np.mean(colors, axis=0), colors
    else:
        return np.array([0, 0, 0]), colors


def smooth_polygon(polygon, alpha=0.5):
    if polygon.geom_type != 'Polygon':
        return []

    exterior = polygon.exterior
    if not exterior:
        return []

    points = np.array(exterior.coords)

    smoothed = points.copy()
    n = len(points)

    for i in range(n):
        prev = points[(i - 1) % n]
        curr = points[i]
        next_p = points[(i + 1) % n]

        smoothed[i] = alpha * curr + (1 - alpha) / 2 * (prev + next_p)

    smoothed[:, 0] = np.clip(smoothed[:, 0], 0, 999)
    smoothed[:, 1] = np.clip(smoothed[:, 1], 0, 999)

    return smoothed


def get_nuclei_centers(point_mask):
    centers = []
    labeled = measure.label(point_mask)
    regions = measure.regionprops(labeled)

    for region in regions:

        center_y = int(region.centroid[0])
        center_x = int(region.centroid[1])
        centers.append((center_y, center_x))

    return centers


def create_pseudo_mask_from_polygons(polygons, h, w):

    mask = np.zeros((h, w), dtype=np.int32)

    for i, polygon in enumerate(polygons):
        if polygon.geom_type == 'Polygon' and polygon.area > 0:

            poly_mask = np.zeros((h, w), dtype=np.uint8)

            exterior = polygon.exterior
            if exterior:

                pts = []
                for coord in exterior.coords:
                    x = int(max(0, min(w - 1, coord[0])))
                    y = int(max(0, min(h - 1, coord[1])))
                    pts.append([x, y])

                if len(pts) >= 3:
                    pts_array = np.array(pts).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.fillPoly(poly_mask, [pts_array], i + 1)


                    mask[poly_mask > 0] = i + 1

    mask = ndi.binary_fill_holes(mask > 0)

    binary_mask = (mask > 0).astype(np.uint8)

    return binary_mask


def generate_optimized_pseudo_label(unoptimized_label, uncertain_label, vor_mask):

    optimized = unoptimized_label.copy()

    vor_boundary = find_voronoi_boundary(vor_mask)

    nuclei_mask = (optimized > 0)
    uncertain_mask = (uncertain_label > 0) & (optimized == 0)
    vor_boundary_nuclei_mask = nuclei_mask & (vor_boundary > 0)
    optimized[vor_boundary_nuclei_mask] = 2
    optimized[uncertain_mask] = 2

    return optimized


def save_pseudo_label_as_png(pseudo_label, save_path, vor_mask=None):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)


    h, w = pseudo_label.shape
    png_image = np.zeros((h, w, 3), dtype=np.uint8)

    background_mask = (pseudo_label == 0)
    png_image[background_mask] = [255, 0, 0]

    nuclei_mask = (pseudo_label == 1)
    png_image[nuclei_mask] = [0, 255, 0]

    if vor_mask is not None:
        vor_boundary_mask = (pseudo_label == 2)
        png_image[vor_boundary_mask] = [0, 0, 0]

    imageio.imwrite(save_path, png_image)

    print(f"\nSaved PNG label to: {save_path}")
    return save_path


def save_binary_label(label, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    h, w = label.shape
    binary_image = np.zeros((h, w), dtype=np.uint8)

    nuclei_mask = (label == 1)
    binary_image[nuclei_mask] = 255

    imageio.imwrite(save_path, binary_image)

    print(f"\nSaved binary label to: {save_path}")
    return save_path


def generate_comparison(image, gt_mask, unoptimized_label, optimized_label, vor_mask, vor_mask1, cluster_mask, name,
                        save_dir):
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original H&E Image')
    axes[0, 0].axis('off')


    axes[0, 1].imshow(gt_mask)
    axes[0, 1].set_title('Instance Label')
    axes[0, 1].axis('off')


    axes[0, 2].imshow(vor_mask1, cmap='gray')
    axes[0, 2].set_title('Voronoi Label')
    axes[0, 2].axis('off')


    axes[0, 3].imshow(cluster_mask, cmap='gray')
    axes[0, 3].set_title('Cluster Label')
    axes[0, 3].axis('off')


    unopt_display = np.zeros_like(image, dtype=np.uint8)

    unopt_display[unoptimized_label == 0] = [255, 0, 0]

    unopt_display[unoptimized_label == 1] = [0, 255, 0]
    axes[1, 0].imshow(unopt_display)
    axes[1, 0].set_title('Unoptimized Pseudo Label\n(Red:Background, Green:Nuclei)')
    axes[1, 0].axis('off')

    opt_display = np.zeros_like(image, dtype=np.uint8)

    opt_display[optimized_label == 0] = [255, 0, 0]

    opt_display[optimized_label == 1] = [0, 255, 0]

    opt_display[optimized_label == 2] = [0, 0, 0]
    axes[1, 1].imshow(opt_display)
    axes[1, 1].set_title('Optimized Pseudo Label\n(Red:BG, Green:Nuclei, Black:Uncertainty Region)')
    axes[1, 1].axis('off')



    nuclei_boundary_overlay = image.copy()

    binary_label = (optimized_label == 1).astype(np.uint8)
    binary_boundaries = find_boundaries(binary_label)
    nuclei_boundary_overlay[binary_boundaries] = [255, 0, 0]
    axes[1, 2].imshow(nuclei_boundary_overlay)
    axes[1, 2].set_title('Boundaries Overlay on H&E Image\n(Red: Nuclei Boundaries)')
    axes[1, 2].axis('off')


    nuclei_boundary_overlay = gt_mask.copy()
    binary_label = (optimized_label == 1).astype(np.uint8)
    binary_boundaries = find_boundaries(binary_label)
    nuclei_boundary_overlay[binary_boundaries] = [255, 0, 0]
    axes[1, 3].imshow(nuclei_boundary_overlay)
    axes[1, 3].set_title('Boundaries Overlay on Instance Label\n(Red: Nuclei Boundaries)')
    axes[1, 3].axis('off')

    boundary_comparison = image.copy()

    pseudo_boundaries = find_boundaries(optimized_label == 1)
    boundary_comparison[pseudo_boundaries] = [255, 0, 0]

    vor_boundary = find_voronoi_boundary(vor_mask)
    boundary_comparison[vor_boundary > 0] = [0, 255, 0]
    axes[2, 0].imshow(boundary_comparison)
    axes[2, 0].set_title('Boundary Comparison\n(Red: Pseudo, Green: Voronoi)')
    axes[2, 0].axis('off')


    boundary_comparison = gt_mask.copy()

    pseudo_boundaries = find_boundaries(optimized_label == 1)
    boundary_comparison[pseudo_boundaries] = [255, 0, 0]

    vor_boundary = find_voronoi_boundary(vor_mask)
    boundary_comparison[vor_boundary > 0] = [0, 255, 0]
    axes[2, 1].imshow(boundary_comparison)
    axes[2, 1].set_title('Boundary Comparison\n(Red: Pseudo, Green: Voronoi)')
    axes[2, 1].axis('off')

    nuclei_boundary_overlay = cluster_mask.copy()

    binary_label = (optimized_label == 1).astype(np.uint8)
    binary_boundaries = find_boundaries(binary_label)
    nuclei_boundary_overlay[binary_boundaries] = [0, 0, 255]
    axes[2, 2].imshow(nuclei_boundary_overlay)
    axes[2, 2].set_title('Boundaries Overlay on Cluster Label\n(Blue: Nuclei Boundaries)')
    axes[2, 2].axis('off')


    diff = optimized_label.astype(int) - unoptimized_label.astype(int)
    diff_display = np.zeros_like(image, dtype=np.uint8)

    diff_display[:] = [128, 128, 128]

    added_nuclei = (diff == 1)
    diff_display[added_nuclei] = [0, 255, 0]
    removed_nuclei = (diff == -1)
    diff_display[removed_nuclei] = [255, 0, 0]

    added_voronoi = (optimized_label == 2) & (unoptimized_label != 2)
    diff_display[added_voronoi] = [0, 0, 0]
    axes[2, 3].imshow(diff_display)
    axes[2, 3].set_title('Optimization Changes\n(Black:Uncertainty Region)')
    axes[2, 3].axis('off')

    plt.tight_layout()

    comparison_path = os.path.join(save_dir, f'{name}_comparison.png')
    plt.savefig(comparison_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"\nSaved comparison to: {comparison_path}")


def save_individual_plots(image, gt_mask, unoptimized_label, optimized_label, vor_mask, name, save_dir):

    subplot_dir = os.path.join(save_dir, 'individual_plots')
    os.makedirs(subplot_dir, exist_ok=True)

    opt_overlay_path = os.path.join(subplot_dir, f'{name}_overlay_on_instance_label.png')
    overlay_opt = gt_mask.copy()

    nuclei_boundaries = find_boundaries(optimized_label == 1)
    overlay_opt[nuclei_boundaries] = [255, 0, 0]  # 红色边界
    plt.figure(figsize=(10, 10), dpi=600)
    plt.imshow(overlay_opt)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(opt_overlay_path, dpi=600, bbox_inches='tight')
    plt.close()


    binary_path = os.path.join(subplot_dir, f'{name}_binary_label.png')
    binary_image = np.zeros_like(image, dtype=np.uint8)
    nuclei_mask = (optimized_label == 1)
    binary_image[nuclei_mask] = [255, 255, 255]  # 白色细胞核
    plt.figure(figsize=(10, 10), dpi=600)
    plt.imshow(binary_image)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(binary_path, dpi=600, bbox_inches='tight')
    plt.close()


    nuclei_boundary_overlay_path = os.path.join(subplot_dir, f'{name}_nuclei_boundary_overlay.png')
    nuclei_boundary_overlay = image.copy()

    binary_label = (optimized_label == 1).astype(np.uint8)
    binary_boundaries = find_boundaries(binary_label)
    nuclei_boundary_overlay[binary_boundaries] = [255, 0, 0]
    plt.figure(figsize=(10, 10), dpi=600)
    plt.imshow(nuclei_boundary_overlay)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(nuclei_boundary_overlay_path, dpi=600, bbox_inches='tight')
    plt.close()

    boundary_comparison_path = os.path.join(subplot_dir, f'{name}_boundary_comparison.png')
    boundary_comparison = image.copy()

    pseudo_boundaries = find_boundaries(optimized_label == 1)
    boundary_comparison[pseudo_boundaries] = [255, 0, 0]

    vor_boundary = find_voronoi_boundary(vor_mask)
    boundary_comparison[vor_boundary > 0] = [0, 255, 0]
    plt.figure(figsize=(10, 10), dpi=600)
    plt.imshow(boundary_comparison)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(boundary_comparison_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"Saved individual plots (600dpi) to: {subplot_dir}")


def find_boundaries(mask):

    from scipy.ndimage import binary_dilation, binary_erosion


    dilated = binary_dilation(mask > 0)
    eroded = binary_erosion(mask > 0)

    boundaries = dilated & ~eroded

    return boundaries


if __name__ == '__main__':
    main()