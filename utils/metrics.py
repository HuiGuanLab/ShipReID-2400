import torch
import numpy as np
import os
from utils.reranking import re_ranking
import os.path as osp


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat

def get_distance(qf, gf):
    m = qf.shape[1]
    n = gf.shape[1]
    q_num = qf.shape[0]
    g_num = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=2, keepdim=True).expand(q_num, m, n) + \
               torch.pow(gf, 2).sum(dim=2, keepdim=True).expand(g_num, n, m).permute(0, 2, 1)
    dist_mat = dist_mat - 2 * (qf @ gf.permute(0, 2, 1))
    return dist_mat

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False, cfg=None):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.cfg = cfg

    def reset(self):
        self.feats = []
        self.part_feats = []
        self.two_feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, part_feat, two_feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.part_feats.append(part_feat.cpu())
        self.two_feats.append(two_feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        part_feats = torch.cat(self.part_feats, dim=0)
        two_feats = torch.cat(self.two_feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
            part_feats = torch.nn.functional.normalize(part_feats, dim=1, p=2)  # along channel
            two_feats = torch.nn.functional.normalize(two_feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        global_cmc, global_mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        # part branch
        qf = part_feats[:self.num_query]     
        gf = part_feats[self.num_query:]
   
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            part_distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        else:
            print('=> Computing DistMat with euclidean_distance')
            part_distmat = euclidean_distance(qf, gf)
        part_cmc, part_mAP = eval_func(part_distmat, q_pids, g_pids, q_camids, g_camids)

        # two branch
        qf = two_feats[:self.num_query]     
        gf = two_feats[self.num_query:]
   
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            part_distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        else:
            print('=> Computing DistMat with euclidean_distance')
            two_distmat = euclidean_distance(qf, gf)
        two_cmc, two_mAP = eval_func(two_distmat, q_pids, g_pids, q_camids, g_camids)        

        if self.cfg is not None and self.cfg.SAVE_NUMPY:
            save_dir = osp.join(self.cfg.OUTPUT_DIR, 'numpy')
            os.makedirs(save_dir, exist_ok=True)
            np.save(osp.join(save_dir, 'feats'), feats)
            np.save(osp.join(save_dir, 'part_feats'), part_feats)
            np.save(osp.join(save_dir, 'part_patch_feats'), part_patch_feats)
            np.save(osp.join(save_dir, 'distmat'), distmat)
            np.save(osp.join(save_dir, 'q_pids'), q_pids)
            np.save(osp.join(save_dir, 'g_pids'), g_pids)
            np.save(osp.join(save_dir, 'q_camids'), q_camids)
            np.save(osp.join(save_dir, 'g_camids'), g_camids)
        return global_cmc, global_mAP, part_cmc, part_mAP, two_cmc, two_mAP, distmat, self.pids, self.camids, qf, gf



