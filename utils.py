import torch
import time
import itertools
import numpy as np
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
import torch.nn as nn
import pickle
import random


def prep_images(images):
    """
    preprocess images
    Args:
        images: pytorch tensor
    """
    images = images.div(255.0)

    images = torch.sub(images, 0.5)
    images = torch.mul(images, 2.0)

    return images


def calc_pairwise_distance(X, Y):
    """
    computes pairwise distance between each element
    Args: 
        X: [N,D]
        Y: [M,D]
    Returns:
        dist: [N,M] matrix of euclidean distances
    """
    rx = X.pow(2).sum(dim=1).reshape((-1, 1))
    ry = Y.pow(2).sum(dim=1).reshape((-1, 1))
    dist = rx - 2.0 * X.matmul(Y.t()) + ry.t()
    return torch.sqrt(dist)


def sincos_encoding_2d(positions, d_emb):
    """
    Args:
        positions: [N,2]
    Returns:
        positions high-dimensional representation: [N,d_emb]
    """

    N = positions.shape[0]

    d = d_emb // 2

    idxs = [np.power(1000, 2 * (idx // 2) / d) for idx in range(d)]
    idxs = torch.FloatTensor(idxs).to(device=positions.device)

    idxs = idxs.repeat(N, 2)  # N, d_emb

    pos = torch.cat([positions[:, 0].reshape(-1, 1).repeat(1, d), positions[:, 1].reshape(-1, 1).repeat(1, d)], dim=1)

    embeddings = pos / idxs

    embeddings[:, 0::2] = torch.sin(embeddings[:, 0::2])  # dim 2i
    embeddings[:, 1::2] = torch.cos(embeddings[:, 1::2])  # dim 2i+1

    return embeddings


def print_log(file_path, save_log_file, *args):
    print(*args)
    if file_path is not None:
        if save_log_file:
            with open(file_path, 'a') as f:
                print(*args, file=f)


def show_config(cfg):
    print_log(cfg.log_path, cfg.save_log_file, '=====================Config=====================')
    for k, v in cfg.__dict__.items():
        print_log(cfg.log_path, cfg.save_log_file, k, ': ', v)
    print_log(cfg.log_path, cfg.save_log_file, '======================End=======================')


def show_epoch_info(phase, log_path, info, cfg):
    print_log(log_path, cfg.save_log_file, '')
    if phase == 'Test':
        print_log(log_path, cfg.save_log_file, '====> %s at epoch #%d' % (phase, info['epoch']))
    else:
        print('experiment_index: ' + str(cfg.experiment_index))
        print_log(log_path, cfg.save_log_file, '%s at epoch #%d' % (phase, info['epoch']))

    if phase == 'Test':
        print_log(log_path, cfg.save_log_file, 
                  'Global f1: %.2f%%,action f1: %.2f%%,social activity f1: %.2f%%, group_det f1: %.2f%%, Overall f1: %.2f%%, Loss: %.5f, Using %.1f seconds' % (
                      info['activities_f1'], info['actions_f1'], info['social_activities_f1'],
                      info['group_detection_acc'], info['overall_f1'],info['loss'],
                      info['time']))
        print_log(log_path, cfg.save_log_file, 
                  'Group_Activity p: %.2f%%,action p: %.2f%%,social activity p: %.2f%%' % (
                      info['activities_p'], info['actions_p'], info['social_activities_p']))
        print_log(log_path, cfg.save_log_file, 
                  'Group_Activity r: %.2f%%,action r: %.2f%%,social activity r: %.2f%%' % (
                      info['activities_r'], info['actions_r'], info['social_activities_r']))

    else:
        print_log(log_path, cfg.save_log_file, 
                  'Global f1: %.2f%%,action f1: %.2f%%, social activity f1: %.2f%%, Overall f1: %.2f%%, Loss: %.5f, Using %.1f seconds' % (
                      info['activities_f1'], info['actions_f1'], info['social_activities_f1'], info['overall_f1'],info['loss'],
                      info['time']))


def log_final_exp_result(log_path, data_path, exp_result):
    no_display_cfg = ['num_workers', 'use_gpu', 'use_multi_gpu', 'device_list',
                      'batch_size_test', 'test_interval_epoch', 'train_random_seed',
                      'result_path', 'log_path', 'device']

    with open(log_path, 'a') as f:
        print('', file=f)
        print('', file=f)
        print('', file=f)
        print('=====================Config=====================', file=f)

        for k, v in exp_result['cfg'].__dict__.items():
            if k not in no_display_cfg:
                print(k, ': ', v, file=f)

        print('=====================Result======================', file=f)

        print('Best result:', file=f)
        print(exp_result['best_result'], file=f)

        print('Cost total %.4f hours.' % (exp_result['total_time']), file=f)

        print('======================End=======================', file=f)

    data_dict = pickle.load(open(data_path, 'rb'))
    data_dict[exp_result['cfg'].exp_name] = exp_result
    pickle.dump(data_dict, open(data_path, 'wb'))


class AverageMeter(object):
    """
    Computes the average value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """
    class to do timekeeping
    """

    def __init__(self):
        self.last_time = time.time()

    def timeit(self):
        old_time = self.last_time
        self.last_time = time.time()
        return self.last_time - old_time


def create_P_matrix(pred):
    people_num = max(pred) + 1
    bbox_num = len(pred)
    matrix_temp = np.identity(bbox_num)
    for i in range(people_num):
        temp = []
        for j in range(bbox_num):
            if pred[j] == i:
                temp.append(j)
        cc = list(itertools.combinations(temp, 2))
        for (p, q) in cc:
            matrix_temp[p][q] = 1
            matrix_temp[q][p] = 1

    return matrix_temp


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def adjust_lr(optimizer, new_lr):
    print('change learning rate:', new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def generate_labels(scores, threshold):
    result = np.zeros((scores.shape[0], scores.shape[1]))
    # result = [np.zeros(scores.shape[1],) for _ in range(scores.shape[0])]
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            if scores[i][j] > threshold:
                result[i][j] = 1
    return result


# calculate the laplacian matrix
def get_laplacian(matrix_A):
    matrix_D = torch.sum(matrix_A, dim=1)
    matrix_D = torch.diag(matrix_D)
    matrix_L = matrix_D - matrix_A
    return matrix_L


# get the eig loss in jrdb paper
def get_eig_loss(predict_matrix, gt_matrix, device):
    eig_loss = torch.zeros((1,), requires_grad=True).to(device=device)
    for i in range(len(predict_matrix)):
        this_num = int(gt_matrix[i].shape[1] ** 0.5)
        this_gt_matrix = torch.tensor(gt_matrix[i]).reshape((this_num, this_num)).to(device=device)
        this_predict_matrix = predict_matrix[i].reshape((this_num, this_num))
        this_laplacian_matrix = get_laplacian(this_predict_matrix).double()

        # get eigenvectors and eigenvalues
        (evals, evecs) = torch.eig(this_gt_matrix, eigenvectors=True)

        # get zero eigenvectors of this gt matrix
        zero_evecs = []
        for val in range(evals.shape[0]):
            # find zero eigenvalues
            if torch.abs(evals[val][0]).item() == 0 and torch.abs(evals[0][1]).item() == 0:
                zero_evecs.append(evecs[val].double())

        if len(zero_evecs) > 0:
            for this_zero_evec in zero_evecs:
                temp = torch.mm(this_zero_evec.reshape(1, -1), this_laplacian_matrix.t())
                temp = torch.mm(temp, this_laplacian_matrix)
                this_loss_1 = torch.mm(temp, this_zero_evec.reshape(1, -1).t())
                this_loss_2 = 1 / torch.exp(torch.trace(torch.mm(this_laplacian_matrix.t(), this_laplacian_matrix)))
                this_loss = this_loss_1 + this_loss_2
                eig_loss += this_loss.reshape(1, )
        return eig_loss


def cosine(source, target):
    return (1 - cosine_similarity(source, target)).mean()

def temporal_shuffle(feature_matrix):
    num_frame, num_person, feat_dim = feature_matrix.shape
    num_to_shuffle = num_person // 2
    indices_to_shuffle = torch.randperm(num_person)[:num_to_shuffle]
    shuffled_matrix = feature_matrix.clone()
    for idx in indices_to_shuffle:
        shuffled_matrix[:, idx] = feature_matrix[torch.randperm(num_frame), idx]
    
    return shuffled_matrix, indices_to_shuffle

class PretClassHead(nn.Module):
    def __init__(self, feat_dim):
        super(PretClassHead, self).__init__()
        self.fc1 = nn.Linear(feat_dim, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_pret_gts(num_person, swapped_indices):
    gt_labels = torch.zeros(num_person, dtype=torch.long)
    if swapped_indices.shape[0] != 0:  # item modified
        gt_labels[swapped_indices] = 1
    return gt_labels

def get_swap_grp(grp_dividing_result, swapped_p_idx):
    swapped_grp_idx = []
    for grp_id, grp_mem in enumerate(grp_dividing_result):
        for swap_pid in swapped_p_idx:
            if swap_pid in grp_mem:
                swapped_grp_idx.append(grp_id)
                break
    swapped_grp_idx = torch.tensor(swapped_grp_idx, device=swapped_p_idx.device)
    
    return swapped_grp_idx

def saptial_swap(feature_matrix, mode='ind'):
    num_frame, num_person, feat_dim = feature_matrix.shape
    if mode == 'ind':
        num_to_swap = num_person // 2
    else:
        num_to_swap = num_person // 4
    if num_to_swap % 2 != 0:
        num_to_swap -= 1
    indices_to_swap = torch.randperm(num_person)[:num_to_swap]
    indices_to_swap = indices_to_swap.view(-1, 2)
    swapped_matrix = feature_matrix.clone()
    for pair in indices_to_swap:
        idx1, idx2 = pair
        swapped_matrix[:, idx1], swapped_matrix[:, idx2] = feature_matrix[:, idx2].clone(), feature_matrix[:, idx1].clone()
    swapped_indices = indices_to_swap.view(-1)
    
    return swapped_matrix, swapped_indices

def calc_pairwise_distance_3d(X, Y):
    """
    computes pairwise distance between each element
    Args:
        X: [B,N,D]
        Y: [B,M,D]
    Returns:
        dist: [B,N,M] matrix of euclidean distances
    """
    # B = X.shape[0]

    # rx = X.pow(2).sum(dim=2).reshape((B, -1, 1))
    # ry = Y.pow(2).sum(dim=2).reshape((B, -1, 1))

    # dist = rx - 2.0 * X.matmul(Y.transpose(1, 2)) + ry.transpose(1, 2)

    X = X.reshape(-1, 2)
    Y = Y.reshape(-1, 2)
    num_n = X.shape[0]
    result = torch.zeros((1, num_n, num_n))
    for i in range(num_n):
        for j in range(num_n):
            temp = (X[i][0] - Y[j][0]) ** 2 + (X[i][1] - Y[j][1]) ** 2
            temp = torch.sqrt(temp)
            result[0][i][j] = temp

    result_original = torch.sqrt(result)
    if True in torch.isnan(result_original):
        print('debug')
    return result_original


def calc_foot_distance(boxes_positions):
    boxes_positions = boxes_positions.reshape(-1, 4)  # N, 4
    num_n = boxes_positions.shape[0]
    xy_foot = torch.zeros((num_n, 2))
    xy_foot[:, 0] = (boxes_positions[:, 0] + boxes_positions[:, 2]) / 2
    xy_foot[:, 1] = boxes_positions[:, 3]

    result = torch.zeros((1, num_n, num_n))
    for i in range(num_n):
        for j in range(num_n):
            temp = min(abs(xy_foot[i][0] - xy_foot[j][0]), (467 - abs(xy_foot[i][0] - xy_foot[j][0]))) ** 2 + \
                   (xy_foot[i][1] - xy_foot[j][1]) ** 2
            temp = torch.sqrt(temp)
            result[0][i][j] = temp

    return result


def calc_spatial_distance(boxes_positions):
    boxes_positions = boxes_positions.reshape(-1, 4)  # N, 4
    num_n = boxes_positions.shape[0]
    square = torch.zeros((num_n, 1))
    square[:, 0] = abs((boxes_positions[:, 3] - boxes_positions[:, 1]) * (boxes_positions[:, 2] - boxes_positions[:, 0]))
    result = torch.zeros((1, num_n, num_n))
    for i in range(num_n):
        for j in range(num_n):
            temp = square[i, 0]/square[j, 0]
            if temp < 1:
                temp = 1 / temp
            result[0][i][j] = temp
    
    return result
    

def normalize_distance(boxes_positions, boxes_distance_foot):
    boxes_positions = boxes_positions.reshape(-1, 4)
    num_n = boxes_positions.shape[0]
    for i in range(num_n):
        for j in range(num_n):
            sum_area_i = (abs(boxes_positions[i][2] - boxes_positions[i][0]) * abs(
                boxes_positions[i][3] - boxes_positions[i][1]))
            sum_area_j = (abs(boxes_positions[j][2] - boxes_positions[j][0]) * abs(
                boxes_positions[j][3] - boxes_positions[j][1]))
            boxes_distance_foot[0][i][j] = boxes_distance_foot[0][i][j] / torch.sqrt(sum_area_i + sum_area_j)
    return boxes_distance_foot


def generate_list(a: np.ndarray):
    res = []
    searched = []
    N, _ = a.shape
    for i in range(N):
        if i not in searched:
            temp = [i]
            for j in range(N):
                if a[i, j] != 0 and i != j:
                    temp.append(j)
            if len(temp) > 1:
                res.append(temp)
            for k in temp:
                searched.append(k)
    return res


def generate_social_acty_labels(social_activity_in, group_id_in, is_train):
    social_group_gt = create_P_matrix(np.array(group_id_in.cpu()))
    social_group_gt = generate_list(social_group_gt)
    social_group_gt.sort()

    social_activity_in_new = []
    for group in social_group_gt:
        if is_train:
            social_activity_in_new.append(social_activity_in[group[0]])
        else:
            social_activity_in_new.append(np.array(social_activity_in[group[0]].cpu()))

    return social_activity_in_new, social_group_gt

def truncate_bboxes(bboxes, max_x=217, max_y=122):
    """
    Adjust bounding boxes to ensure they are within specified bounds and do not exceed the image boundary.
    
    Args:
    - bboxes (torch.Tensor): Tensor of shape (frames, people, 4) containing bbox coordinates.
    - max_x (int): Maximum allowable x-coordinate (width limit).
    - max_y (int): Maximum allowable y-coordinate (height limit).
    
    Returns:
    - torch.Tensor: Adjusted bounding boxes.
    """
    # Clone the tensor to avoid modifying the original
    adjusted_bboxes = bboxes.permute(1, 0, 2).clone()

    # Clamp values to ensure they are within the bounds
    adjusted_bboxes[..., 0] = torch.clamp(adjusted_bboxes[..., 0], 0, max_x)  # x1
    adjusted_bboxes[..., 2] = torch.clamp(adjusted_bboxes[..., 2], 0, max_x)  # x2
    adjusted_bboxes[..., 1] = torch.clamp(adjusted_bboxes[..., 1], 0, max_y)  # y1
    adjusted_bboxes[..., 3] = torch.clamp(adjusted_bboxes[..., 3], 0, max_y)  # y2

    # Find invalid bboxes (those that are completely out of bounds)
    invalid_bboxes = (adjusted_bboxes[..., 0] > adjusted_bboxes[..., 2]) | (adjusted_bboxes[..., 1] > adjusted_bboxes[..., 3])

    # Iterate over each person and each frame
    for person in range(adjusted_bboxes.shape[0]):
        for frame in range(adjusted_bboxes.shape[1]):
            if invalid_bboxes[person, frame].any():
                # Find the nearest valid bbox
                valid_bbox = None
                # Search backwards
                for f in range(frame - 1, -1, -1):
                    if not invalid_bboxes[person, f].any():
                        valid_bbox = adjusted_bboxes[person, f]
                        break
                # If not found, search forwards
                if valid_bbox is None:
                    for f in range(frame + 1, adjusted_bboxes.shape[1]):
                        if not invalid_bboxes[person, f].any():
                            valid_bbox = adjusted_bboxes[person, f]
                            break
                # Replace invalid bbox with the nearest valid one
                if valid_bbox is not None:
                    adjusted_bboxes[person, frame] = valid_bbox

    return adjusted_bboxes

def remove_diagonal(mat):
    if mat.size(0) != mat.size(1):
        raise ValueError("Input matrix must be square.")
    size = mat.size(0)
    mask = ~torch.eye(size, dtype=torch.bool).to(mat.device)
    off_diagonal_elements = mat[mask]
    new_mat = off_diagonal_elements.reshape(size, size - 1)

    return new_mat

def add_back_diagonals(off_diagonal_elements, diagonal_elements=None):
    if diagonal_elements is not None and off_diagonal_elements.size(0) != diagonal_elements.size(0):
        raise ValueError("Size of diagonal elements must match the size of the off-diagonal matrix.")
    size = off_diagonal_elements.size(0)
    new_mat = torch.zeros((size, size), dtype=off_diagonal_elements.dtype).to(off_diagonal_elements.device)
    mask = torch.eye(size, dtype=torch.bool).to(off_diagonal_elements.device)
    if diagonal_elements is not None:
        new_mat[mask] = diagonal_elements
    else:
        new_mat[mask] = 0
    new_mat[~mask] = off_diagonal_elements.flatten()
    
    return new_mat

def padding_sim_matrixs(list_of_tensors, max_size):
    padded_tensors = []
    masks = []
    for tensor in list_of_tensors:
        mask = torch.ones(tensor.size(), dtype=torch.float32, device=tensor.device)
        padding = (0, max_size[1] - tensor.size(1), 0, max_size[0] - tensor.size(0))
        padded_tensor = F.pad(tensor, padding)
        mask = F.pad(mask, padding)
        
        padded_tensors.append(padded_tensor)
        masks.append(mask)
    
    stacked_tensors = torch.stack(padded_tensors)  # b x max_n x max_m
    stacked_masks = torch.stack(masks)  # b x max_n x max_m
    
    return stacked_tensors, stacked_masks

def sim_matrix_mse_loss(outputs, targets, mask):
    difference = outputs - targets
    squared_diff = difference ** 2
    masked_squared_diff = squared_diff * mask
    loss = torch.sum(masked_squared_diff) / torch.sum(mask)
    return loss

def calculate_area(bboxes):
    """Calculate the area of each bbox."""
    x1, y1, x2, y2 = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    return (x2 - x1) * (y2 - y1)

def swap_features(matrix_a, matrix_b):
    t, n, featdim = matrix_a.shape
    t, m, featdim = matrix_b.shape
    if n < m:
        smaller_matrix = matrix_a
        larger_matrix = matrix_b
        smaller_size = n
        larger_size = m
    else:
        smaller_matrix = matrix_b
        larger_matrix = matrix_a
        smaller_size = m
        larger_size = n

    num_to_swap = smaller_size // 2
    indices_to_swap_smaller = torch.randperm(smaller_size)[:num_to_swap]
    indices_to_swap_larger = torch.randperm(larger_size)[:num_to_swap]

    new_smaller_matrix = smaller_matrix.clone()
    new_larger_matrix = larger_matrix.clone()

    new_smaller_matrix[:, indices_to_swap_smaller] = larger_matrix[:, indices_to_swap_larger]
    new_larger_matrix[:, indices_to_swap_larger] = smaller_matrix[:, indices_to_swap_smaller]

    swapped_indices_smaller = indices_to_swap_larger
    swapped_indices_larger = indices_to_swap_smaller

    if n < m:
        return new_smaller_matrix, new_larger_matrix, swapped_indices_smaller.tolist(), swapped_indices_larger.tolist()
    else:
        return new_larger_matrix, new_smaller_matrix, swapped_indices_larger.tolist(), swapped_indices_smaller.tolist()

def calculate_movement(bboxes):
    """Calculate the movement from the first to the last frame and normalize."""
    first_frame = bboxes[0]
    last_frame = bboxes[-1]
    
    # Calculate centroids
    center_first = (first_frame[..., :2] + first_frame[..., 2:]) / 2
    center_last = (last_frame[..., :2] + last_frame[..., 2:]) / 2
    
    # Calculate Euclidean distances
    distances = torch.norm(center_last - center_first, dim=-1)
    
    # Calculate areas
    area_first = calculate_area(first_frame)
    area_last = calculate_area(last_frame)
    
    # Normalize movements
    normalized_distances = distances / torch.sqrt(area_first + area_last)
    
    # Calculate area change ratio
    area_change_ratio = torch.abs(area_last - area_first) / torch.clamp(area_first, min=1e-6)
    
    return normalized_distances, area_change_ratio

def find_extreme_movers(bboxes, k, n_percent):
    """Find the farthest and nearest k movers, excluding the top n% area changers only for nearest movers."""
    movements, area_changes = calculate_movement(bboxes)
    
    # Find the farthest k movers without any exclusions
    farthest_values, farthest_indices = torch.topk(movements, k)
    
    # Determine the exclusion threshold for area change ratio
    threshold = torch.quantile(area_changes, 1 - n_percent / 100)
    
    # Exclude the top n% area changers for nearest movers
    valid_mask = area_changes < threshold
    valid_movements_near = torch.where(valid_mask, movements, torch.tensor(float('inf'), device=valid_mask.device))
    nearest_values, nearest_indices = torch.topk(valid_movements_near, k, largest=False)
    
    return farthest_indices, nearest_indices, farthest_values, nearest_values

def is_fully_unfrozen(model):
    return all(param.requires_grad for param in model.parameters())

class RelationNetwork(nn.Module):
    def __init__(self):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output = torch.matmul(x, x.T)
        return output