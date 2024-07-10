import torch
import json
import numpy as np
from tqdm import tqdm

class Traj_offline_jrdb:
    def __init__(self, data_dic, clp_len, tracking) -> None:
        self.data_dic = data_dic
        self.out_size = 57, 467
        self.nan_coord = [-1, -1, -1, -1]
        self.nan = torch.ones(4) * -1
        self.clp_len = clp_len
        self.max_person_num = 60
        self.zero_graph = [-1. for i in range(60)]
        self.tracking = tracking
        self.OW = 467
        self.pos_threshold = 0.6
        
    def calc_foot_distance(self, boxes_positions):
        boxes_positions = boxes_positions.reshape(-1, 4)  # N, 4
        num_n = boxes_positions.shape[0]
        xy_foot = torch.zeros((num_n, 2))
        xy_foot[:, 0] = (boxes_positions[:, 0] + boxes_positions[:, 2]) / 2
        xy_foot[:, 1] = boxes_positions[:, 3]
        nan = self.nan

        result = torch.zeros((1, num_n, num_n))
        for i in range(num_n):
            for j in range(num_n):
                temp = min(abs(xy_foot[i][0] - xy_foot[j][0]), (467 - abs(xy_foot[i][0] - xy_foot[j][0]))) ** 2 + \
                    (xy_foot[i][1] - xy_foot[j][1]) ** 2
                temp = torch.sqrt(temp)
                
                i_nan = (boxes_positions[i] == nan).all()  # ij not on the same frame
                j_nan = (boxes_positions[j] == nan).all()
                if i_nan or j_nan:
                    temp = -1
                
                result[0][i][j] = temp

        return result
    
    def normalize_distance(self, boxes_positions, boxes_distance_foot):
        boxes_positions = boxes_positions.reshape(-1, 4)
        num_n = boxes_positions.shape[0]
        nan = self.nan
        for i in range(num_n):
            for j in range(num_n):
                sum_area_i = (abs(boxes_positions[i][2] - boxes_positions[i][0]) * abs(
                    boxes_positions[i][3] - boxes_positions[i][1]))
                sum_area_j = (abs(boxes_positions[j][2] - boxes_positions[j][0]) * abs(
                    boxes_positions[j][3] - boxes_positions[j][1]))
                squres = sum_area_i + sum_area_j
                
                i_nan = (boxes_positions[i] == nan).all()  # ij not on the same frame
                j_nan = (boxes_positions[j] == nan).all()
                if i_nan or j_nan:
                    squres = torch.tensor(1.)
                
                boxes_distance_foot[0][i][j] = boxes_distance_foot[0][i][j] / torch.sqrt(squres)
                
        return boxes_distance_foot
    
    def get_relation_graphs(self, clp_traj):
        batch_traj = clp_traj  # N, T, 4
        batch_traj = batch_traj.permute(1, 0, 2)  # T, N, 4
        T = batch_traj.shape[0]
        dis_lst = []
        for t in range(T):
            boxes_positions = batch_traj[t]
            dis = self.calc_foot_distance(boxes_positions)
            dis_norm = self.normalize_distance(boxes_positions, dis).squeeze(0)
            dis_lst.append(dis_norm)
        # dynamic mean
        clp_dis = torch.zeros_like(dis_norm)
        for i in range(dis_norm.shape[0]):
            for j in range(dis_norm.shape[1]):
                num_f = 0
                dis = 0
                for t in range(T):
                    this_dis = dis_lst[t][i][j]
                    if this_dis != -1:
                        dis += this_dis
                        num_f += 1
                assert num_f != 0
                clp_dis[i][j] = dis / num_f
        # mean                
        # clp_dis = sum(dis_lst)
        
        # max
        # clp_dis = max(dis_lst)
        
        # # mask
        # position_mask = (clp_dis > (self.pos_threshold * self.OW))
        
        clp_dis = 1 / clp_dis
        clp_dis = torch.nan_to_num(clp_dis, 0, 1)
        # clp_dis[position_mask] = -float('inf')
        
        return clp_dis
    
    def get_offline_graphs(self):
        OH, OW = self.out_size
        for clp_id in tqdm(self.data_dic.keys()):
            # read key frame boxes
            key_frame_pos = self.data_dic[clp_id]['key_frame_pos']
            key_frame_dic = self.data_dic[clp_id]['frames'][key_frame_pos]
            key_frame_box = {}
            all_frame_dics = []
            this_clp_pid_lst = []
            for p_id in key_frame_dic.keys():
                current_bbox = key_frame_dic[p_id]['bbox']
                y1, x1, y2, x2 = current_bbox
                w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                if self.tracking:
                    track_id = key_frame_dic[p_id]['track_id']
                    key_frame_box[track_id] = [w1, h1, w2, h2]
                    this_clp_pid_lst.append(track_id)
                else:
                    key_frame_box[p_id] = [w1, h1, w2, h2]
                    this_clp_pid_lst.append(p_id)
            # read all frames boxes
            for pos, frame_dic in enumerate(self.data_dic[clp_id]['frames']):
                this_frame_box = {}
                for p_id in frame_dic.keys():
                    current_bbox = frame_dic[p_id]['bbox']
                    y1, x1, y2, x2 = current_bbox
                    w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                    if self.tracking:
                        track_id = key_frame_dic[p_id]['track_id']
                        this_frame_box[track_id] = [w1, h1, w2, h2]
                    else:
                        this_frame_box[p_id] = [w1, h1, w2, h2]
                all_frame_dics.append(this_frame_box)
            
            clp_data = []
            for p_id in this_clp_pid_lst:
                this_p_data = []
                for frame_pos in range(self.clp_len):
                    if p_id in all_frame_dics[frame_pos].keys():
                        this_pid_this_f = all_frame_dics[frame_pos][p_id]
                    else:
                        this_pid_this_f = self.nan_coord
                    this_p_data.append(this_pid_this_f)  # clp_len, 4
                clp_data.append(this_p_data)  # N, clp_len, 4
            
            clp_data = torch.tensor(clp_data, device='cpu')
            clp_traj_graph = self.get_relation_graphs(clp_traj=clp_data)
            clp_traj_graph = clp_traj_graph.numpy().tolist()
            
            # adding fake anno for dataloader
            for i, person_lst in enumerate(clp_traj_graph):
                while len(person_lst) < self.max_person_num:
                    clp_traj_graph[i].append(-1.)
            while len(clp_traj_graph) < self.max_person_num:
                clp_traj_graph.append(self.zero_graph)
            while len(this_clp_pid_lst) < self.max_person_num:
                this_clp_pid_lst.append(-1.)
                
            self.data_dic[clp_id]['traj_graph'] = clp_traj_graph
            self.data_dic[clp_id]['graph_order'] = this_clp_pid_lst


class Traj_offline_par:
    def __init__(self, data_dic, clp_len, tracking) -> None:
        self.data_dic = data_dic
        self.out_size = 57, 467
        self.nan_coord = [-1, -1, -1, -1]
        self.nan = torch.ones(4) * -1
        self.clp_len = clp_len
        self.max_person_num = 60
        self.zero_graph = [-1. for i in range(60)]
        self.tracking = tracking
        
    def calc_foot_distance(self, boxes_positions):
        boxes_positions = boxes_positions.reshape(-1, 4)  # N, 4
        num_n = boxes_positions.shape[0]
        xy_foot = torch.zeros((num_n, 2))
        xy_foot[:, 0] = (boxes_positions[:, 0] + boxes_positions[:, 2]) / 2
        xy_foot[:, 1] = boxes_positions[:, 3]
        nan = self.nan

        result = torch.zeros((1, num_n, num_n))
        for i in range(num_n):
            for j in range(num_n):
                temp = min(abs(xy_foot[i][0] - xy_foot[j][0]), (467 - abs(xy_foot[i][0] - xy_foot[j][0]))) ** 2 + \
                    (xy_foot[i][1] - xy_foot[j][1]) ** 2
                temp = torch.sqrt(temp)
                
                i_nan = (boxes_positions[i] == nan).all()  # ij not on the same frame
                j_nan = (boxes_positions[j] == nan).all()
                if i_nan or j_nan:
                    temp = -1
                
                result[0][i][j] = temp

        return result
    
    def normalize_distance(self, boxes_positions, boxes_distance_foot):
        boxes_positions = boxes_positions.reshape(-1, 4)
        num_n = boxes_positions.shape[0]
        nan = self.nan
        for i in range(num_n):
            for j in range(num_n):
                sum_area_i = (abs(boxes_positions[i][2] - boxes_positions[i][0]) * abs(
                    boxes_positions[i][3] - boxes_positions[i][1]))
                sum_area_j = (abs(boxes_positions[j][2] - boxes_positions[j][0]) * abs(
                    boxes_positions[j][3] - boxes_positions[j][1]))
                squres = sum_area_i + sum_area_j
                
                i_nan = (boxes_positions[i] == nan).all()  # ij not on the same frame
                j_nan = (boxes_positions[j] == nan).all()
                if i_nan or j_nan:
                    squres = torch.tensor(1.)
                
                boxes_distance_foot[0][i][j] = boxes_distance_foot[0][i][j] / torch.sqrt(squres)
                
        return boxes_distance_foot
    
    def get_relation_graphs(self, clp_traj):
        batch_traj = clp_traj  # N, T, 4
        batch_traj = batch_traj.permute(1, 0, 2)  # T, N, 4
        T = batch_traj.shape[0]
        dis_lst = []
        for t in range(T):
            boxes_positions = batch_traj[t]
            dis = self.calc_foot_distance(boxes_positions)
            dis_norm = self.normalize_distance(boxes_positions, dis).squeeze(0)
            dis_lst.append(dis_norm)
        # dynamic mean
        clp_dis = torch.zeros_like(dis_norm)
        for i in range(dis_norm.shape[0]):
            for j in range(dis_norm.shape[1]):
                num_f = 0
                dis = 0
                for t in range(T):
                    this_dis = dis_lst[t][i][j]
                    if this_dis != -1:
                        dis += this_dis
                        num_f += 1
                assert num_f != 0
                clp_dis[i][j] = dis
        # mean                
        # clp_dis = sum(dis_lst)
        
        # max
        # clp_dis = max(dis_lst)
        
        clp_dis = 1 / clp_dis
        clp_dis = torch.nan_to_num(clp_dis, 0, 1)
        
        return clp_dis
    
    def get_offline_graphs(self):
        OH, OW = self.out_size
        for clp_id in tqdm(self.data_dic.keys()):
            # read key frame boxes
            key_frame_pos = self.data_dic[clp_id]['key_f_pos']
            key_frame_dic = self.data_dic[clp_id]['frames'][key_frame_pos]
            key_frame_box = {}
            all_frame_dics = []
            this_clp_pid_lst = []
            for p_id in key_frame_dic.keys():
                current_bbox = key_frame_dic[p_id]['coord']
                y1, x1, y2, x2 = current_bbox
                w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                if self.tracking:
                    track_id = key_frame_dic[p_id]['track_id']
                    key_frame_box[track_id] = [w1, h1, w2, h2]
                    this_clp_pid_lst.append(track_id)
                else:
                    key_frame_box[p_id] = [w1, h1, w2, h2]
                    this_clp_pid_lst.append(p_id)
            # read all frames boxes
            for pos, frame_dic in enumerate(self.data_dic[clp_id]['frames']):
                this_frame_box = {}
                for p_id in frame_dic.keys():
                    current_bbox = frame_dic[p_id]['coord']
                    y1, x1, y2, x2 = current_bbox
                    w1, h1, w2, h2 = x1 * OW, y1 * OH, x2 * OW, y2 * OH
                    if self.tracking:
                        track_id = frame_dic[p_id]['track_id']
                        this_frame_box[track_id] = [w1, h1, w2, h2]
                    else:
                        this_frame_box[p_id] = [w1, h1, w2, h2]
                all_frame_dics.append(this_frame_box)
            
            clp_data = []
            for p_id in this_clp_pid_lst:
                this_p_data = []
                for frame_pos in range(self.clp_len):
                    if p_id in all_frame_dics[frame_pos].keys():
                        this_pid_this_f = all_frame_dics[frame_pos][p_id]
                    else:
                        this_pid_this_f = self.nan_coord
                    this_p_data.append(this_pid_this_f)  # clp_len, 4
                clp_data.append(this_p_data)  # N, clp_len, 4
            
            clp_data = torch.tensor(clp_data, device='cpu')
            clp_traj_graph = self.get_relation_graphs(clp_traj=clp_data)
            clp_traj_graph = clp_traj_graph.numpy().tolist()
            
            # adding fake anno for dataloader
            for i, person_lst in enumerate(clp_traj_graph):
                while len(person_lst) < self.max_person_num:
                    clp_traj_graph[i].append(-1.)
            while len(clp_traj_graph) < self.max_person_num:
                clp_traj_graph.append(self.zero_graph)
            while len(this_clp_pid_lst) < self.max_person_num:
                this_clp_pid_lst.append(-1.)
                
            self.data_dic[clp_id]['traj_graph'] = clp_traj_graph
            self.data_dic[clp_id]['graph_order'] = this_clp_pid_lst


if __name__ == '__main__':
    print('/HDDs/sdd1/wsm/data4code/for_track/test_73_clp_traj_new.json')
    with open('/HDDs/sdd1/wsm/data4code/for_track/test_73_clp_mottrack_traj_new.json', mode='r') as f:
        all_dic = json.load(f)
    offline = Traj_offline_par(data_dic=all_dic, clp_len=15, tracking=False)
    offline.get_offline_graphs()
    with open('/HDDs/sdd1/wsm/data4code/for_track/test_73_clp_traj_new.json', mode='w') as f:
        json.dump(offline.data_dic, f)
    
    # with open('/home/yanhaomin/songmiao/project_new/PAR-utils-new/all_15f_long_track_traj.json', mode='r') as f:
    #     all_dic = json.load(f)
    # for clp_id in all_dic.keys():
    #     for frame_dic in all_dic[clp_id]['frames']:
    #         for p_id in frame_dic.keys():
    #             if str(p_id) not in all_dic[clp_id]['graph_order']:
    #                 print(p_id, all_dic[clp_id]['graph_order'])