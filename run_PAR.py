import sys
from config import *
import os
from utils import create_P_matrix, generate_list
import numpy as np
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

sys.path.append("scripts")
from train_net import *
from par import read_anno_par

class Saver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.is_save = cfg.is_save
        self.actions_scores, self.social_acty_scores, self.activities_scores = None, None, None
        self.vid_id_map = read_anno_par(cfg.vid_id_map_path)
        self.images_path = cfg.data_path
        self.all_results = {}
    
    def save_scores(self, actions_scores, social_acty_scores, activities_scores):
        if self.is_save:
            self.actions_scores, self.social_acty_scores, self.activities_scores = \
                actions_scores, social_acty_scores, activities_scores
    
    def save_annos(self, *annos):
        if self.is_save:
            self.clp_all_f_bboxs, self.clp_all_f_bbox_nums, self.clp_all_f_p_ids, self.clp_key_f_pos, self.seq_id, self.frame_id, self.group_id = annos
        # clp_all_f_bboxs [b, clp_len, max_n, 4]
        # clp_all_f_bbox_nums [b, clp_len]
        # clp_all_f_p_ids [b, clp_len, max_n]
        # clp_key_f_pos [b, clp_len]
        # seq_id [b, 1, 1]
        # frame_id [b, 1, 1]
        # group_id [b, 1, max_n]
    
    def save_grp_div(self, grp_dividing_results):
        if self.is_save:
            self.grp_dividing_results = grp_dividing_results
    
    def generate_labels(self, scores, threshold):
        result = np.zeros((scores.shape[0], scores.shape[1]))
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                if scores[i][j] > threshold:
                    result[i][j] = 1
        return result
    
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
    
    def get_labels(self):
        self.actions_labels, self.social_acty_labels, self.activities_labels = [], [], []
        for b in range(len(self.actions_scores)):
            self.actions_labels.append(self.generate_labels(self.actions_scores[b], cfg.action_threshold))
            self.social_acty_labels.append(self.generate_labels(self.social_acty_scores[b], cfg.social_activity_threshold))
            self.activities_labels.append(self.generate_labels(self.activities_scores[b], cfg.activity_threshold))
    
    def grp_div_gt(self, grp_ids):
        social_group_gt = create_P_matrix(np.array(grp_ids))
        social_group_gt = generate_list(social_group_gt)
        social_group_gt.sort()
        
        return social_group_gt
    
    def save_batch_results(self):
        if self.is_save:
            self.get_labels()
            batch_size = self.clp_key_f_pos.shape[0]
            clp_all_f_bboxs = self.clp_all_f_bboxs.cpu()
            clp_all_f_p_ids = self.clp_all_f_p_ids.cpu()
            group_id = self.group_id.cpu()
            for b in range(batch_size):
                key_f = self.clp_key_f_pos[b][0]
                bboxs_nums = self.clp_all_f_bbox_nums[b][key_f]
                bboxs = clp_all_f_bboxs[b][key_f][:bboxs_nums].numpy().tolist()
                p_ids = clp_all_f_p_ids[b][key_f][:bboxs_nums].numpy().tolist()
                seq_id = int(self.seq_id[b][0][0])
                frame_id = int(self.frame_id[b][0][0])
                labels = [self.actions_labels[b].tolist(), self.social_acty_labels[b].tolist(), self.activities_labels[b].tolist()]
                img_pth = self.get_img_pth(seq_id, frame_id)
                grp_dividing_result = self.grp_dividing_results[b]
                grp_ids = group_id[b][0][:bboxs_nums]
                grp_dividing_gt = self.grp_div_gt(grp_ids)
                self.all_results[img_pth] = {
                    'bboxs': bboxs,
                    'p_ids': p_ids,
                    'labels': labels,
                    'grp_div': grp_dividing_result,
                    'grp_div_gt': grp_dividing_gt
                }

    def write_json(self, pth):
        if self.is_save:
            print('saving result at {}'.format(pth))
            with open(pth, mode='w') as f:
                json.dump(self.all_results, f)
        self.all_results = {}
    
cfg = Config('PAR')

cfg.device_list = "0,1"
cfg.training_stage = 2
cfg.stage1_model_path = '/home/yanhaomin/songmiao/PAR-main-new-7fps/66.04%.pth'  # PATH OF THE BASE MODEL

cfg.image_size = 720, 1280  # 480, 3760
cfg.out_size = 122, 217  # 57， 467
cfg.num_boxes = 100
cfg.num_frames = 1

cfg.num_graph = cfg.num_graphs = cfg.num_SGAT=4
cfg.tau_sqrt = True
cfg.num_actions = 24
cfg.num_activities = 8
cfg.num_social_activities = 12

cfg.actions_loss_weight = 4
cfg.social_activities_loss_weight = 6
cfg.activities_loss_weight = 2
cfg.grpdiv_loss_weight = 2  # 2

cfg.pos_threshold = 0.2
cfg.cluster_threshold = 0.6 # 0.6
cfg.square_threshold = 8
cfg.action_threshold = 0.6
cfg.social_activity_threshold = 0.6 # 0.6
cfg.activity_threshold = 0.4  # 0.5

cfg.module_name = 'Block_par'

cfg.batch_size = 3  # 10
cfg.test_batch_size = 1  # 1
cfg.test_interval_epoch = 1  # 1
cfg.start_test_epoch = 0  # 5 
cfg.train_learning_rate = 1e-5  # 0.6e-4
cfg.lr_plan = {30: 1e-5} # {30: 1e-5} {30: 2e-5}
cfg.train_dropout_prob = 0.2 # 0.2
cfg.weight_decay = 0.005
cfg.max_epoch = 20

cfg.tracking = False
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
cfg.clp_len = 15
cfg.sample_interval = 7  # 7
cfg.num_LSTM_layers = 1
cfg.save_log_file = True  # save log txt
cfg.is_save = False  # save results for visualization
cfg.use_conLoss = False
cfg.offline_graph = True
cfg.bg_feature = False
cfg.num_lite_graph = 1  # num graph in (spat) gcn 1
# cfg.num_temp_graph_global = 1  # num graph in temp gcn 1 (only used in orig version)

cfg.num_graph_ind2grp = 1
cfg.num_graph_grp2glb = 1
cfg.num_graph_ind2glb = 1

s = Saver(cfg)
cfg.min_clusters = 1
cfg.max_clusters = 34
cfg.test_before_train = False

cfg.train_backbone = False
cfg.finetune_from_grp_pret = False

# grp div trained motion and 2cls pred
# cfg.pret_model_path = '/home/yanhaomin/songmiao/PAR/PanoAct_source-code/result/[a repretrain]<2024-06-19_23-30-06>/state_dict_epoch_8.pth'
# cfg.pret_models = ('fc_emb_1', 'nl_emb_1', 'gcn_ind', 'LSTM_1', 'proj')
# cfg.unfreeze_epoch = 15  # better equal to start_test_epoch

# only used in finetuning(2stage finetuning)
# cfg.tuning_interval = 7

# coords in par dataset are normalized
# imgs size in par dataset are not unified

# cfg.my_note = 'par pret cls, grpdiv loss + motdiv loss, cor matrix loss'
# cfg.exp_note = 'par grpwmot cls cor pret'

cfg.my_note = 'MUP freeze backbone'
cfg.exp_note = 'MUP'


# MUP
cfg.crop_size = 32, 32
cfg.num_features_boxes = 768
# num_features_relation = 768
cfg.num_features_gcn = 768


train_net(cfg, s)
# pret_net(cfg, s)

# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("Total number of parameters: {}".format(total_params))

# for name, submodule in model.named_children():
#     # 计算每个子模块的可训练参数数量
#     submodule_params = sum(p.numel() for p in submodule.parameters() if p.requires_grad)
#     print("{}: {} trainable parameters".format(name, submodule_params))
# assert False
    