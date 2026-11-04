from turtle import pd
import torch
from torch import nn

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n) #B, B
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    B = labels.size(0)
    dist_mat = dist_mat.view(B, B, -1).min(-1)[0]

    is_pos = labels.expand(B, B).eq(labels.expand(B, B).t()) 
    is_neg = labels.expand(B, B).ne(labels.expand(B, B).t())

    dist_ap = dist_mat[is_pos].view(B, -1).min(-1)[0]
    dist_an = dist_mat[is_neg].view(B, -1).min(-1)[0]

    return dist_ap, dist_an

# edit by wtz
def sinkhorn(log_alpha, n_iters=50, epsilon=1e-6):
    """
    Sinkhorn-Knopp algorithm in log-space, batched over input.
    Args:
        log_alpha: tensor of shape (..., N, M)
        n_iters: number of iterations
    Returns:
        Doubly stochastic matrix (..., N, M)
    """
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)  # Row normalize
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)  # Column normalize
    return torch.exp(log_alpha)

def part_hungarian_loss(dist_mat, labels, tau=0.05, num_feat=4, top_k=1.0):
    B = labels.size(0)

    is_pos = labels.expand(B, B).eq(labels.expand(B, B).t())
    is_neg = labels.expand(B, B).ne(labels.expand(B, B).t())

    dist_ap = dist_mat[is_pos].view(B, -1)
    # 将 dist_mat 分为 [48, 4] 行组 和 [4, 4] 列组
    dist_ap = dist_ap.view(int(B/num_feat), num_feat, -1, num_feat)  # [48组, 4行, 4列组, 4列]
    # 对每个 4x4 block 求最小值（行、列维度）
    dist_ap = dist_ap.permute(0, 2, 1, 3)
    dist_ap = dist_ap.reshape(-1, num_feat, num_feat)
    logits = -dist_ap / tau
    soft_assignment = sinkhorn(logits, n_iters=50)
    # 生成top4的mask
    soft_assignment = soft_assignment.view(dist_ap.size(0), -1)
    topk_values, topk_indices = torch.topk(soft_assignment, k=num_feat, dim=1)
    mask = torch.zeros_like(soft_assignment)
    mask.scatter_(dim=1, index=topk_indices, value=1)
    mask = mask.view(dist_ap.size(0), num_feat, num_feat)
    dist_ap = dist_ap[mask.bool()].view(-1, num_feat)
    values, indices = torch.topk(-dist_ap, int(top_k), dim=1)
    dist_ap = (-values).mean(dim=1)
    # 最小值
    # dist_ap = dist_ap.min(dim=1).values
    dist_ap = dist_ap.view(-1, 4)
    dist_ap = dist_ap.mean(dim=1)
    dist_an = dist_mat[is_neg].view(B, -1).min(-1)[0]

    dist_an = dist_an.view(-1, num_feat).mean(dim=1)

    return dist_ap, dist_an

class TokenTripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
            feat = normalize(feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, feat.view(-1, feat.size(-1))) #B,B
        dist_ap, dist_an = hard_example_mining(dist_mat, labels) 

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1) 
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss

# edit by wtz
class PartTokenAssignmentLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, feat, labels, normalize_feature=False, top_k = 1.0):
        if normalize_feature:
            feat = normalize(feat, axis=-1)
        dist_mat = euclidean_dist(feat.view(-1, feat.size(-1)), feat.view(-1, feat.size(-1))) #B,B
        dist_ap, dist_an = part_hungarian_loss(dist_mat=dist_mat, labels=labels.repeat_interleave(feat.size(1)), num_feat=feat.size(1), top_k=top_k)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss
