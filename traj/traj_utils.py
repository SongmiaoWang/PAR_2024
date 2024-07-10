import torch

# torch.Size([1, 15, 3, 480, 3760])
# torch.Size([1, 15, 60, 4])
# torch.Size([1, 15])
# torch.Size([1, 15, 60])
# torch.Size([1, 15, 60])
# images_in, boxes_in, bboxes_num_in, bboxes_social_group_id_in, person_id_in

# def get_traj(boxes_in, bboxes_num_in, person_id_in, key_f):
#     """_summary_ traj_dic {batch:{person_id:{'pos':position in tensor, 'coord':[...]}}}

#     Args:
#         boxes_in (_type_): _description_
#         bboxes_num_in (_type_): _description_
#         person_id_in (_type_): _description_
#     """
#     device = boxes_in.device
#     boxes_in = boxes_in.to('cpu')
#     bboxes_num_in = bboxes_num_in.to('cpu')
#     person_id_in = person_id_in.to('cpu')
#     B = boxes_in.shape[0]
#     T = boxes_in.shape[1]
#     traj_dic = {}
#     traj = []
#     key_map = {}
#     for b in range(B):
#         traj_dic[b] = {}
#         p_id_lst = []
#         p_num_key_f = bboxes_num_in[b][key_f]
#         p_id_key_f = person_id_in[b][key_f][:p_num_key_f]
#         this_traj = torch.zeros(p_num_key_f, T, 4, device=boxes_in.device)
#         for i, p_id in enumerate(p_id_key_f):
#             p_id = int(p_id)
#             key_map[p_id] = i
#             p_id_lst.append(p_id)
#         for t in range(T):
#             p_num_this_f = bboxes_num_in[b][t]
#             for p in range(p_num_this_f):
#                 this_p_id = person_id_in[b][t][p]
#                 this_p_id = int(this_p_id)
#                 if this_p_id in p_id_lst:
#                     this_traj[key_map[this_p_id]][t] = boxes_in[b][t][p]
#                     if this_p_id not in traj_dic[b].keys():
#                         traj_dic[b][this_p_id] = {
#                             'pos': key_map[this_p_id],
#                             'coord': [boxes_in[b][t][p]]
#                         }
#                     else:
#                         traj_dic[b][this_p_id]['coord'].append(boxes_in[b][t][p])
#         traj.append(this_traj.to(device=device))
#     return traj

def d_v_distance(traj_i, traj_j, clp_len):
    dis_lst = []
    for t in range(clp_len):
        vec_i = traj_i[t]
        vec_j = traj_j[t]
        pos_i = (vec_i[: 2] + vec_i[2 :]) / 2
        pos_j = (vec_j[: 2] + vec_j[2 :]) / 2
        square_i = (vec_i[: 2] - vec_i[2 :])[0] * (vec_i[: 2] - vec_i[2 :])[1]
        square_j = (vec_j[: 2] - vec_j[2 :])[0] * (vec_j[: 2] - vec_j[2 :])[1]
        this_dis = E_distance(pos_i, pos_j, square_i, square_j)
        if this_dis != None:
            dis_lst.append(this_dis)
    if len(dis_lst) == 0:
        dis = torch.tensor(-1, device='cpu')
    else:
        dis = max(dis_lst)
    # dis_vec = torch.stack(dis_lst, dim=0)       
    # dis = (dis_vec[:-1] - dis_vec[1:]) ** 2
    # dis = torch.std(dis_vec)
    dis = 1 / dis
    # relation = dis.sum() / dis_vec.shape[0]
    # if len(dis_lst) == 1:
    #     dis = torch.tensor(0.0, device=traj_i.device)
    
    assert not torch.isnan(dis).any()
    assert not torch.isinf(dis).any()
    
    return dis

def get_traj(boxes_in, bboxes_num_in, person_id_in, key_f):
    """_summary_ traj_dic {batch:{person_id:{'pos':position in tensor, 'coord':[...]}}}

    Args:
        boxes_in (_type_): _description_
        bboxes_num_in (_type_): _description_
        person_id_in (_type_): _description_
    """
    B = boxes_in.shape[0]
    T = boxes_in.shape[1]
    traj = []
    key_map = {}
    for b in range(B):
        p_id_lst = []
        p_num_key_f = bboxes_num_in[b][key_f]
        p_id_key_f = person_id_in[b][key_f][:p_num_key_f]
        this_traj = torch.zeros(p_num_key_f, T, 4, device=boxes_in.device)
        for i, p_id in enumerate(p_id_key_f):
            p_id = int(p_id)
            key_map[p_id] = i
            p_id_lst.append(p_id)
        for t in range(T):
            p_num_this_f = bboxes_num_in[b][t]
            for p in range(p_num_this_f):
                this_p_id = person_id_in[b][t][p]
                this_p_id = int(this_p_id)
                if this_p_id in p_id_lst:
                    this_traj[key_map[this_p_id]][t] = boxes_in[b][t][p]
        traj.append(this_traj)
    return traj

def E_distance(vec_i, vec_j, squ_i ,squ_j):
    device = vec_i.device
    E_dis = (vec_i[0] - vec_j[0]) ** 2 + (vec_i[1] - vec_j[1]) ** 2
    ret = torch.sqrt(E_dis) / torch.sqrt(squ_i + squ_j)
    ret = torch.tensor(ret, device=device)
    device=vec_i.device
    zero = torch.ones(2, device=device) * -1
    flag_i = (vec_i == zero)
    flag_i = flag_i.all()
    flag_j = (vec_j == zero)
    flag_j = flag_j.all()
    if flag_i or flag_j:
        ret = None
            
    return ret

def norm(x):
    #[bsize,seq_len]
    mean=torch.mean(x,dim=1)#[bsize]
    std=torch.std(x,dim=1)#[bsize]
    return (x-mean.unsqueeze(1))/std.unsqueeze(1)

def get_traj_matrix(boxes_in, bboxes_num_in, person_id_in):
    """_summary_

    Args:
        boxes_in (_type_): _description_ (t, n_max, coord)
        bboxes_num_in (_type_): _description_ (t)
        person_id_in (_type_): _description_ (t, p_num)
    """
    device = boxes_in.device
    boxes_in = boxes_in.to('cpu')
    bboxes_num_in = bboxes_num_in.to('cpu')
    person_id_in = person_id_in.to('cpu')
    T = boxes_in.shape[0]  # clip length
    person_lst = []
    get_global_id = {}
    for t in range(T):
        person_id_this_f = person_id_in[t]
        for p_id in person_id_this_f:
            if p_id == -1:
                continue
            if p_id not in person_lst:
                person_lst.append(p_id)
    for global_id, p_id in enumerate(person_lst):
        get_global_id[int(p_id)] = global_id
    batch_traj = []
    for t in range(T):
        this_box_matrix = []
        person_id_this_f = person_id_in[t]
        for p_id in person_lst:
            if p_id not in person_id_this_f:
                this_box_matrix.append(torch.ones(4) * -1)
            else:
                for i, id_ in enumerate(person_id_this_f):
                    if id_ == p_id:
                        this_box_matrix.append(boxes_in[t][i])
                        break
        this_box_matrix = torch.stack(this_box_matrix, dim=0)
        batch_traj.append(this_box_matrix)
    batch_traj = torch.stack(batch_traj, dim=0)
    batch_traj = batch_traj.permute(1, 0, 2)  # n, t, 4
    N = batch_traj.shape[0]
    batch_adj = torch.zeros(N, N, device='cpu')
    for i in range(N):
        for j in range(N):
            if j > i:
                continue
            traj_i = batch_traj[i]
            traj_j = batch_traj[j]
            if i == j:
                batch_adj[i][j] = torch.tensor(0., device='cpu')
            else:
                batch_adj[i][j] = d_v_distance(traj_i, traj_j, clp_len=T)
    for i in range(N):
        for j in range(N):
            if j > i:
                batch_adj[i][j] = batch_adj[j][i]
    all_relation_graphs = []
    for t in range(T):
        bboxes_num_this_f = bboxes_num_in[t]
        person_id_this_f = person_id_in[t][:bboxes_num_this_f]
        this_relation_graph = torch.zeros(bboxes_num_this_f, bboxes_num_this_f, device='cpu')
        for index_i, p_id_i in enumerate(person_id_this_f):
            global_p_id_i = get_global_id[int(p_id_i)]
            for index_j, p_id_j in enumerate(person_id_this_f):
                global_p_id_j = get_global_id[int(p_id_j)]
                this_relation_graph[index_i][index_j] = batch_adj[global_p_id_i][global_p_id_j]
        this_relation_graph = this_relation_graph.to(device=device)
        all_relation_graphs.append(this_relation_graph)
    
    batch_traj = linear_interpolation(batch_traj)
    batch_vecs = get_moving_vecs(batch_traj)  # t, n, 4  batch_traj->n, t, 4
    all_mov_vecs = []
    for t in range(T):
        batch_vec_this_f = batch_vecs[t]
        bboxes_num_this_f = bboxes_num_in[t]
        person_id_this_f = person_id_in[t][:bboxes_num_this_f]
        this_mov_vec = torch.zeros(bboxes_num_this_f, 4, device='cpu')
        for index_i, p_id_i in enumerate(person_id_this_f):
            global_p_id_i = get_global_id[int(p_id_i)]
            this_mov_vec[index_i] = batch_vec_this_f[global_p_id_i]
        this_mov_vec = this_mov_vec.to(device=device)
        all_mov_vecs.append(this_mov_vec)
        
    return all_relation_graphs, all_mov_vecs

def linear_interpolation(A):
    # A : N, T, 4
    coord_del = torch.ones(4) * -1
    N = A.shape[0]
    T = A.shape[1]
    for n in range(N):
        ref_dic = {}
        del_lst = []
        for t in range(T):
            if True not in (A[n][t] == coord_del):
                ref_dic[t] = A[n][t]
            else:
                del_lst.append(t)
        assert len(del_lst) != T
        for del_f in del_lst:
            sum_weight = 0
            coord = torch.zeros(4)
            for ref_f, ref_coord in ref_dic.items():
                f_dis = abs(ref_f - del_f)
                # if f_dis <= 3:
                sum_weight += 1 / f_dis
                coord += 1 / f_dis * ref_coord
            coord = coord / sum_weight
            A[n][del_f] = coord
    
    return A

def get_moving_vecs(batch_traj):
    # batch_traj: batch_n, t, 4
    T = batch_traj.shape[1]
    batch_traj = batch_traj.permute(1, 0, 2)  # t, batch_n, 4
    batch_mov_vec = []
    for t in range(T):
        if t == 0:
            this_f_vec = batch_traj[t+1] - batch_traj[t]
        elif t == T - 1:
            this_f_vec = batch_traj[t] - batch_traj[t-1]
        else:
            this_f_vec = (batch_traj[t+1] - batch_traj[t-1]) / 2
        batch_mov_vec.append(this_f_vec)
    batch_mov_vec = torch.stack(batch_mov_vec, dim=0)  # t, batch_n, 4
    
    return batch_mov_vec