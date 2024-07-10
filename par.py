import torch
from torch.utils import data
import torchvision.transforms as transforms
import os
import random
from PIL import Image
import numpy as np
import pandas as pd
import json
import time

class PAR_Dataset(data.Dataset):
    """
    Characterize PAR dataset for pytorch
    """
    
    def __init__(self, num_actions, num_activities, num_social_activities, vid_id_map, all_dic, images_path, image_size,
                 feature_size, num_boxes=60, clp_len=None, num_frames=1, tracking=True,
                 is_training=True, is_finetune=False):
        self.vid_id_map = vid_id_map
        self.all_dic = all_dic
        self.images_path = images_path
        self.image_size = image_size
        self.feature_size = feature_size  # OH, OW
        self.num_boxes = num_boxes
        self.is_training = is_training
        self.is_finetune = is_finetune
        self.num_frames = num_frames
        self.num_actions = num_actions
        self.num_activities = num_activities
        self.num_social_activities = num_social_activities
        self.clp_len = clp_len  # 15 or 30
        self.tracking = tracking
        self.clp_id_map = self.get_clp_id_map()
    
    def __len__(self):
        """
        Return the total number of samples
        """
        
        return len(self.all_dic)
    
    def __getitem__(self, index):
        """
        Generate one sample of the dataset
        """

        selected_clp = self.get_clp(index)
        sample = self.load_samples_sequence(selected_clp, self.clp_len)

        return sample
    
    def one_hot(self, labels, num_categories):
        result = [0 for _ in range(num_categories)]
        for label in labels:
            if label == -2:
                continue
            result[label] = 1
        return result
    
    def get_clp_id_map(self):
        clp_id_map = {}
        for i, clp_id in enumerate(self.all_dic.keys()):
            clp_id_map[i] = clp_id
        
        return clp_id_map
    
    def get_clp(self, index):
        clp_id = self.clp_id_map[index]
        selected_clp = self.all_dic[clp_id]
        
        return selected_clp
    
    def sample_frame(self, current_f_id):
        if self.is_training:
                sample_frame = random.sample(range(current_f_id - self.num_frames, current_f_id + self.num_frames + 1), 1)
                return sample_frame[0]
        else:
            return current_f_id
    
    def get_img_pth(self, this_vid_id, this_f_id, sample=False):
        this_vid_name = None
        for vid_name, vid_id in self.vid_id_map.items():
            if int(vid_id) == this_vid_id:
                this_vid_name = vid_name
                break
        assert this_vid_name != None
        this_vid_id, this_f_id = str(this_vid_id), str(this_f_id)
        img_root = self.images_path
        this_vid_name = this_vid_name.strip(' ')
        this_vid_name = this_vid_name.replace(' ', '/')
        this_img_pth = os.path.join(img_root, this_vid_name, this_f_id+'.jpg')
        if not sample:
            assert os.path.exists(this_img_pth)
        
        return this_img_pth
           
    def load_samples_sequence(self, selected_clp, clp_len):
        assert len(selected_clp['frames']) == clp_len  # selected_clp['frames'] list of frame_dic
        OH, OW = self.feature_size
        clp_global_acty = int(selected_clp['global_acty'])
        clp_vid_id = int(selected_clp['vid_id'])
        zero_action = [0 for _ in range(self.num_actions)]
        zero_grp_acty = [0 for _ in range(self.num_social_activities)]
        
        # clp each frame annos
        clp_f_bboxs = []
        clp_f_imgs = []
        clp_f_actions = []
        clp_f_global_actys = []
        clp_f_grp_actys = []
        clp_f_grp_ids = []
        clp_f_p_ids = []
        clp_f_bbox_nums = []
        clp_f_vid_id = []
        clp_f_frame_id = []
        clp_key_f_pos = []
        
        # get key_frame_pos
        key_frame_pos = selected_clp['key_f_pos']
        
        for i, frame_id in enumerate(selected_clp['frame_id_lst']):
            frame_dic = selected_clp['frames'][i]
            assert frame_id.keys()[0] == frame_id               
            
            current_f_bboxs = []
            current_f_actions = []
            current_f_global_actys = []  
            current_f_grp_actys = []
            current_f_grp_ids = []
            current_f_p_ids = []
            current_f_vid_id = []
            current_f_frame_id = []
            
            for current_p_id in frame_dic.keys():
                current_p_dic = frame_dic[current_p_id]
                
                current_vid_id = clp_vid_id
                current_f_id = frame_id
                current_global_acty = [clp_global_acty]
                current_coord = current_p_dic['coord']
                current_track_id = current_p_dic['track_id']
                current_action = current_p_dic['action']
                
                # reading true & fake grp_id & grp_acty
                if 'grp_id' in current_p_dic.keys():
                    current_grp_id = current_p_dic['grp_id']
                    current_grp_acty = current_p_dic['grp_acty']
                else:
                    current_grp_id = -1
                    current_grp_acty = [-1]
            
                # sample_frame
                # sample_f_id = self.sample_frame(current_f_id)
                sample_f_id = current_f_id
                
                # process current frame bboxs
                y1, x1, y2, x2 = current_coord
                w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                current_f_bboxs.append((w1, h1, w2, h2))         
                # actions current frame
                current_f_actions.append(self.one_hot(current_action, self.num_actions))                      
                # grp actys current frame
                current_f_grp_actys.append(self.one_hot(current_grp_acty, self.num_social_activities))              
                # p_ids current frame
                if self.tracking:
                    current_f_p_ids.append(current_track_id)
                else:
                    current_f_p_ids.append(current_p_id)               
                # grp_ids current frame
                current_f_grp_ids.append(current_grp_id)
                # bboxes num
                current_f_bbox_nums = len(current_f_bboxs)
            
            # actys current frame
            current_f_global_actys.append(self.one_hot(current_global_acty, self.num_activities))  # last acty
            current_f_vid_id.append(current_vid_id)
            current_f_frame_id.append(current_f_id)
            
            # adding fake person annos
            assert current_f_bbox_nums <= self.num_boxes
            while len(current_f_bboxs) != self.num_boxes:
                current_f_bboxs.append((0, 0, 0, 0))  
                current_f_actions.append(zero_action)
                current_f_grp_actys.append(zero_grp_acty)
                current_f_p_ids.append(-1)
                current_f_grp_ids.append(-1)
            
            # bbox
            clp_f_bboxs.append(current_f_bboxs)
            # img
            sample_f_img_path = self.get_img_pth(clp_vid_id, sample_f_id, sample=True)
            f_img_path = self.get_img_pth(clp_vid_id, frame_id, sample=False)
            if os.path.exists(sample_f_img_path):
                img = Image.open(sample_f_img_path)
            else:
                img = Image.open(f_img_path)
            img = transforms.functional.resize(img, self.image_size)
            img = np.array(img)
            img = img.transpose(2, 0, 1)  # H,W,3 -> 3,H,W
            clp_f_imgs.append(img)
            # action
            clp_f_actions.append(current_f_actions)
            # social acty
            clp_f_grp_actys.append(current_f_grp_actys)
            # grp_id
            clp_f_grp_ids.append(current_f_grp_ids)
            # p_id
            clp_f_p_ids.append(current_f_p_ids)
            # bbox nums
            clp_f_bbox_nums.append(current_f_bbox_nums)
            # seq_id
            clp_f_vid_id.append(current_vid_id)
            # frame_id
            clp_f_frame_id.append(current_f_frame_id)
            # key_frame_pos
            clp_key_f_pos.append(key_frame_pos)       
            # acty
            clp_f_global_actys.append(current_f_global_actys)
            
        # converting       
        clp_f_imgs = np.stack(clp_f_imgs)      
        clp_f_global_actys = np.array(clp_f_global_actys, dtype=np.int32).reshape((-1, self.num_activities))
        clp_f_bbox_nums = np.array(clp_f_bbox_nums, dtype=np.int32)
        clp_f_bboxs = np.array(clp_f_bboxs, dtype=np.float).reshape((-1, self.num_boxes, 4))
        clp_f_actions = np.array(clp_f_actions, dtype=np.int32).reshape((-1, self.num_boxes, self.num_actions))
        clp_f_grp_actys = np.array(clp_f_grp_actys, dtype=np.int32).reshape(
            (-1, self.num_boxes, self.num_social_activities))
        clp_f_p_ids = np.array(clp_f_p_ids, dtype=np.int32).reshape(-1, self.num_boxes)
        clp_f_grp_ids = np.array(clp_f_grp_ids, dtype=np.int32).reshape(-1, self.num_boxes)
        clp_f_vid_id = np.array(clp_f_vid_id, dtype=np.int32).reshape((-1, 1))
        clp_f_frame_id = np.array(clp_f_frame_id, dtype=np.int32).reshape((-1, 1))
        clp_key_f_pos = np.array(clp_key_f_pos, dtype=np.int32)
        
        # convert to pytorch tensor
        clp_f_imgs = torch.from_numpy(clp_f_imgs).float()
        clp_f_bboxs = torch.from_numpy(clp_f_bboxs).float()
        clp_f_actions = torch.from_numpy(clp_f_actions).long()
        clp_f_grp_actys = torch.from_numpy(clp_f_grp_actys).long()
        clp_f_p_ids = torch.from_numpy(clp_f_p_ids).long()
        clp_f_grp_ids = torch.from_numpy(clp_f_grp_ids).long()
        clp_f_global_actys = torch.from_numpy(clp_f_global_actys).long()
        clp_f_bbox_nums = torch.from_numpy(clp_f_bbox_nums).int()
        clp_f_vid_id = torch.from_numpy(clp_f_vid_id).int()
        clp_f_frame_id = torch.from_numpy(clp_f_frame_id).int()
        clp_key_f_pos = torch.from_numpy(clp_key_f_pos).int()
        
        return clp_f_imgs, clp_f_bboxs, clp_f_actions, clp_f_global_actys, \
               clp_f_bbox_nums, clp_f_grp_actys, clp_f_p_ids, clp_f_grp_ids, \
               clp_f_vid_id, clp_f_frame_id, clp_key_f_pos

class PAR_Dataset_offline(data.Dataset):
    """
    Characterize PAR dataset for pytorch
    """
    
    def __init__(self, num_actions, num_activities, num_social_activities, vid_id_map, all_dic, images_path, image_size,
                 feature_size, num_boxes=60, clp_len=None, num_frames=1, tracking=True,
                 is_training=True, is_finetune=False, sample_interval=None):
        self.vid_id_map = vid_id_map
        self.all_dic = all_dic
        self.images_path = images_path
        self.image_size = image_size
        self.feature_size = feature_size  # OH, OW
        self.num_boxes = num_boxes
        self.is_training = is_training
        self.is_finetune = is_finetune
        self.num_frames = num_frames
        self.num_actions = num_actions
        self.num_activities = num_activities
        self.num_social_activities = num_social_activities
        self.clp_len = clp_len  # 15 or 30
        self.tracking = tracking
        self.clp_id_map = self.get_clp_id_map()
        self.sample_interval = sample_interval
    
    def __len__(self):
        """
        Return the total number of samples
        """
        
        return len(self.all_dic)
    
    def __getitem__(self, index):
        """
        Generate one sample of the dataset
        """

        selected_clp = self.get_clp(index)
        sample = self.load_samples_sequence(selected_clp, self.clp_len)

        return sample
    
    def one_hot(self, labels, num_categories):
        result = [0 for _ in range(num_categories)]
        for label in labels:
            if label == -2 or label == -1:
                continue
            assert not label == 0
            result[label-1] = 1
        return result
    
    def get_clp_id_map(self):
        clp_id_map = {}
        for i, clp_id in enumerate(self.all_dic.keys()):
            clp_id_map[i] = clp_id
        
        return clp_id_map
    
    def get_clp(self, index):
        clp_id = self.clp_id_map[index]
        selected_clp = self.all_dic[clp_id]
        
        return selected_clp
    
    def sample_frame(self, current_f_id):
        if self.is_training:
                sample_frame = random.sample(range(current_f_id - self.num_frames, current_f_id + self.num_frames + 1), 1)
                return sample_frame[0]
        else:
            return current_f_id
    
    def sample_clp(self, key_f_pos):
        sampled_frames = [i for i in range(self.clp_len) if i % self.sample_interval == 0]
        if key_f_pos not in sampled_frames:
            nearest_pos = [abs(i - key_f_pos) for i in sampled_frames]
            nearest_pos = nearest_pos.index(min(nearest_pos))
            sampled_frames[nearest_pos] = key_f_pos
        new_key_f_pos = sampled_frames.index(key_f_pos)
        
        return sampled_frames, new_key_f_pos
    
    def get_img_pth(self, this_vid_id, this_f_id, sample=False):
        this_vid_name = None
        for vid_name, vid_id in self.vid_id_map.items():
            if int(vid_id) == this_vid_id:
                this_vid_name = vid_name
                break
        assert this_vid_name != None
        this_vid_id, this_f_id = str(this_vid_id), str(this_f_id)
        img_root = self.images_path
        this_vid_name = this_vid_name.strip(' ')
        this_vid_name = this_vid_name.replace(' ', '/')
        this_img_pth = os.path.join(img_root, this_vid_name, this_f_id+'.jpg')
        if not sample:
            assert os.path.exists(this_img_pth)
        
        return this_img_pth
           
    def load_samples_sequence(self, selected_clp, clp_len):
        assert len(selected_clp['frames']) == clp_len  # selected_clp['frames'] list of frame_dic
        OH, OW = self.feature_size
        clp_global_acty = selected_clp['global_acty']
        clp_vid_id = int(selected_clp['vid_id'])
        zero_action = [0 for _ in range(self.num_actions)]
        zero_grp_acty = [0 for _ in range(self.num_social_activities)]
        
        # clp each frame annos
        clp_f_bboxs = []
        clp_f_imgs = []
        clp_f_actions = []
        clp_f_global_actys = []
        clp_f_grp_actys = []
        clp_f_grp_ids = []
        clp_f_p_ids = []
        clp_f_bbox_nums = []
        clp_f_vid_id = []
        clp_f_frame_id = []
        clp_key_f_pos = []
        
        # clp offline graph annos
        clp_traj_graph = selected_clp['traj_graph']
        clp_graph_order = selected_clp['graph_order']
        
        # get key_frame_pos
        key_frame_pos = selected_clp['key_f_pos']
        
        # only sample part of the clip for feature extraction
        if self.sample_interval != None:
            clp_samp_f_pos, key_frame_pos = self.sample_clp(key_frame_pos)  # overwrite key_frame_pos
        else:
            clp_samp_f_pos = [i for i in range(len(selected_clp['frames']))]
        
        for i, frame_dic in enumerate(selected_clp['frames']):
            if i in clp_samp_f_pos:
                frame_id = selected_clp['frame_id_lst'][i]             
                
                current_f_bboxs = []
                current_f_actions = []
                current_f_global_actys = []  
                current_f_grp_actys = []
                current_f_grp_ids = []
                current_f_p_ids = []
                current_f_vid_id = []
                current_f_frame_id = []
                
                for current_p_id in frame_dic.keys():
                    current_p_dic = frame_dic[current_p_id]
                    
                    current_vid_id = clp_vid_id
                    current_f_id = frame_id
                    current_global_acty = [int(s) for s in clp_global_acty]
                    current_coord = current_p_dic['coord']
                    current_track_id = current_p_dic['track_id']
                    current_action = current_p_dic['action']
                    
                    # reading true & fake grp_id & grp_acty
                    if 'grp_id' in current_p_dic.keys():
                        current_grp_id = current_p_dic['grp_id']
                        current_grp_acty = current_p_dic['grp_acty']
                    else:
                        current_grp_id = -1
                        current_grp_acty = [-1]
                
                    # sample_frame
                    # sample_f_id = self.sample_frame(current_f_id)
                    sample_f_id = current_f_id
                    
                    # process current frame bboxs
                    y1, x1, y2, x2 = current_coord
                    w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                    current_f_bboxs.append((w1, h1, w2, h2))         
                    # actions current frame
                    current_f_actions.append(self.one_hot(current_action, self.num_actions))                      
                    # grp actys current frame
                    current_f_grp_actys.append(self.one_hot(current_grp_acty, self.num_social_activities))              
                    # p_ids current frame
                    if self.tracking:
                        current_f_p_ids.append(current_track_id)
                    else:
                        current_f_p_ids.append(current_p_id)               
                    # grp_ids current frame
                    current_f_grp_ids.append(current_grp_id)
                    # bboxes num
                    current_f_bbox_nums = len(current_f_bboxs)
                
                # actys current frame
                current_f_global_actys.append(self.one_hot(current_global_acty, self.num_activities))  # last acty
                current_f_vid_id.append(current_vid_id)
                current_f_frame_id.append(current_f_id)
                
                # adding fake person annos
                assert current_f_bbox_nums <= self.num_boxes
                while len(current_f_bboxs) != self.num_boxes:
                    current_f_bboxs.append((0, 0, 0, 0))  
                    current_f_actions.append(zero_action)
                    current_f_grp_actys.append(zero_grp_acty)
                    current_f_p_ids.append(-1)
                    current_f_grp_ids.append(-1)
                
                # bbox
                clp_f_bboxs.append(current_f_bboxs)
                # img
                sample_f_img_path = self.get_img_pth(clp_vid_id, sample_f_id, sample=True)
                f_img_path = self.get_img_pth(clp_vid_id, frame_id, sample=False)
                if os.path.exists(sample_f_img_path):
                    img = Image.open(sample_f_img_path)
                else:
                    img = Image.open(f_img_path)
                img = transforms.functional.resize(img, self.image_size)
                img = np.array(img)
                img = img.transpose(2, 0, 1)  # H,W,3 -> 3,H,W
                clp_f_imgs.append(img)
                # action
                clp_f_actions.append(current_f_actions)
                # social acty
                clp_f_grp_actys.append(current_f_grp_actys)
                # grp_id
                clp_f_grp_ids.append(current_f_grp_ids)
                # p_id
                clp_f_p_ids.append(current_f_p_ids)
                # bbox nums
                clp_f_bbox_nums.append(current_f_bbox_nums)
                # seq_id
                clp_f_vid_id.append(current_vid_id)
                # frame_id
                clp_f_frame_id.append(current_f_frame_id)
                # key_frame_pos
                clp_key_f_pos.append(key_frame_pos)       
                # acty
                clp_f_global_actys.append(current_f_global_actys)
            
        # converting       
        clp_f_imgs = np.stack(clp_f_imgs)      
        clp_f_global_actys = np.array(clp_f_global_actys, dtype=np.int32).reshape((-1, self.num_activities))
        clp_f_bbox_nums = np.array(clp_f_bbox_nums, dtype=np.int32)
        clp_f_bboxs = np.array(clp_f_bboxs, dtype=np.float).reshape((-1, self.num_boxes, 4))
        clp_f_actions = np.array(clp_f_actions, dtype=np.int32).reshape((-1, self.num_boxes, self.num_actions))
        clp_f_grp_actys = np.array(clp_f_grp_actys, dtype=np.int32).reshape(
            (-1, self.num_boxes, self.num_social_activities))
        clp_f_p_ids = np.array(clp_f_p_ids, dtype=np.int32).reshape(-1, self.num_boxes)
        clp_f_grp_ids = np.array(clp_f_grp_ids, dtype=np.int32).reshape(-1, self.num_boxes)
        clp_f_vid_id = np.array(clp_f_vid_id, dtype=np.int32).reshape((-1, 1))
        clp_f_frame_id = np.array(clp_f_frame_id, dtype=np.int32).reshape((-1, 1))
        clp_key_f_pos = np.array(clp_key_f_pos, dtype=np.int32)
        
        clp_traj_graph = np.array(clp_traj_graph, dtype=np.float)
        clp_graph_order = np.array(clp_graph_order, dtype=np.int32)
        
        # convert to pytorch tensor
        clp_f_imgs = torch.from_numpy(clp_f_imgs).float()
        clp_f_bboxs = torch.from_numpy(clp_f_bboxs).float()
        clp_f_actions = torch.from_numpy(clp_f_actions).long()
        clp_f_grp_actys = torch.from_numpy(clp_f_grp_actys).long()
        clp_f_p_ids = torch.from_numpy(clp_f_p_ids).long()
        clp_f_grp_ids = torch.from_numpy(clp_f_grp_ids).long()
        clp_f_global_actys = torch.from_numpy(clp_f_global_actys).long()
        clp_f_bbox_nums = torch.from_numpy(clp_f_bbox_nums).int()
        clp_f_vid_id = torch.from_numpy(clp_f_vid_id).int()
        clp_f_frame_id = torch.from_numpy(clp_f_frame_id).int()
        clp_key_f_pos = torch.from_numpy(clp_key_f_pos).int()
        
        clp_traj_graph = torch.from_numpy(clp_traj_graph).float()
        clp_graph_order = torch.from_numpy(clp_graph_order).long()
        
        return clp_f_imgs, clp_f_bboxs, clp_f_actions, clp_f_global_actys, \
               clp_f_bbox_nums, clp_f_grp_actys, clp_f_p_ids, clp_f_grp_ids, \
               clp_f_vid_id, clp_f_frame_id, clp_key_f_pos, clp_traj_graph, clp_graph_order

def read_anno_par(json_pth):
    assert os.path.exists(json_pth)
    with open(json_pth, mode='r') as f:
        anno = json.load(f)
    
    return anno

# def train_test_spilt_par(train_seqs, test_seqs, all_dic):
#     train_dic = {}
#     test_dic = {}
#     for clp_id in all_dic.keys():
#         seq_id = int(all_dic[clp_id]['vid_id'])
#         if seq_id in train_seqs:
#             train_dic[clp_id] = all_dic[clp_id]
#         elif seq_id in test_seqs:
#             test_dic[clp_id] = all_dic[clp_id]
    
#     return train_dic, test_dic

# if __name__ == '__main__':
#     t1 = time.time()
#     with open('/home/yanhaomin/songmiao/project_new/PAR_dataset/vid_id_map.json', mode='r') as f:
#         vid_id_map = json.load(f)
    # with open('/home/yanhaomin/songmiao/project_new/PAR_dataset/all_dic_w_fake1_clp_track.json', mode='r') as f:
    #     all_dic = json.load(f)
#     images_path = '/HDDs/sdd1/wsm/PAR/PAR_new'
#     image_size = 480, 3760
#     feature_size = 57, 467
#     par = PAR_Dataset(num_actions=24, num_activities=6, num_social_activities=12, vid_id_map=vid_id_map,all_dic=all_dic,
#                       images_path=images_path, image_size=image_size, feature_size=feature_size, num_boxes=60, clp_len=15, num_frames=1, 
#                       tracking=True, is_training=True, is_finetune=False)
#     data_loader = data.DataLoader(par, batch_size=3)
#     t2 = time.time()
#     print('reading time:', t2-t1)
#     t_st = t2
#     for batch_data in data_loader:
#         batch_data
#         t = time.time()
#         print(t-t_st)
#         t_st = t