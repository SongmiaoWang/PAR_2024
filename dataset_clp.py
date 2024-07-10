# from jrdb import *
from jrdb_clp import *
from jrdb_clp_offline import *
from par import *

def return_dataset(cfg):
    if cfg.dataset_name == "JRDB":
        all_dic = read_anno_jrdb(cfg.json_anno_path)
        
        train_anno_dic, test_anno_dic = train_test_spilt_jrdb(cfg.train_seqs, cfg.test_seqs, all_dic)
        
        if not cfg.offline_graph:
            training_set = JRDB_Dataset(cfg.num_actions, cfg.num_activities, cfg.num_social_activities, train_anno_dic, 
                                        cfg.data_path, cfg.image_size, cfg.out_size, cfg.num_boxes, cfg.clp_len, cfg.num_frames, False, 
                                        is_training=True, is_finetune=False)

            testing_set = JRDB_Dataset(cfg.num_actions, cfg.num_activities, cfg.num_social_activities, test_anno_dic, 
                                        cfg.data_path, cfg.image_size, cfg.out_size, cfg.num_boxes, cfg.clp_len, cfg.num_frames, cfg.tracking,
                                        is_training=False, is_finetune=False)
        else:
            training_set = JRDB_Dataset_offline(cfg.num_actions, cfg.num_activities, cfg.num_social_activities, train_anno_dic, 
                                                cfg.data_path, cfg.image_size, cfg.out_size, cfg.num_boxes, cfg.clp_len, cfg.num_frames, False, 
                                                is_training=True, is_finetune=False)

            testing_set = JRDB_Dataset_offline(cfg.num_actions, cfg.num_activities, cfg.num_social_activities, test_anno_dic, 
                                                cfg.data_path, cfg.image_size, cfg.out_size, cfg.num_boxes, cfg.clp_len, cfg.num_frames, cfg.tracking,
                                                is_training=False, is_finetune=False)
    
    elif cfg.dataset_name == "PAR":
        vid_id_map = read_anno_par(cfg.vid_id_map_path)
        
        train_anno_dic = read_anno_par(cfg.train_json_anno_path)
        
        test_anno_dic = read_anno_par(cfg.test_json_anno_path)
        
        if not cfg.offline_graph:
            training_set = PAR_Dataset(cfg.num_actions, cfg.num_activities, cfg.num_social_activities, vid_id_map, train_anno_dic,
                                    cfg.data_path, cfg.image_size, cfg.out_size, cfg.num_boxes, cfg.clp_len, cfg.num_frames, False,
                                    is_training=True, is_finetune=False)
            
            testing_set = PAR_Dataset(cfg.num_actions, cfg.num_activities, cfg.num_social_activities, vid_id_map, test_anno_dic,
                                    cfg.data_path, cfg.image_size, cfg.out_size, cfg.num_boxes, cfg.clp_len, cfg.num_frames, cfg.tracking,
                                    is_training=False, is_finetune=False)
        else:
            training_set = PAR_Dataset_offline(cfg.num_actions, cfg.num_activities, cfg.num_social_activities, vid_id_map, train_anno_dic,
                                    cfg.data_path, cfg.image_size, cfg.out_size, cfg.num_boxes, cfg.clp_len, cfg.num_frames, False,
                                    is_training=True, is_finetune=False, sample_interval=cfg.sample_interval)
            
            testing_set = PAR_Dataset_offline(cfg.num_actions, cfg.num_activities, cfg.num_social_activities, vid_id_map, test_anno_dic,
                                    cfg.data_path, cfg.image_size, cfg.out_size, cfg.num_boxes, cfg.clp_len, cfg.num_frames, cfg.tracking,
                                    is_training=False, is_finetune=False, sample_interval=cfg.sample_interval)


    print('Reading dataset finished...')
    print('%d train samples' % len(train_anno_dic))
    print('%d test samples' % len(test_anno_dic))

    return training_set, testing_set