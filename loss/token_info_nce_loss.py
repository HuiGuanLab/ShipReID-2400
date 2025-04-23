import torch
import torch.nn.functional as F
from torch import nn

class TokenInfoNCE(nn.Module):
    def __init__(self, T=0.1):
        super(TokenInfoNCE, self).__init__()
        self.T = T

    def forward(self, global_feat, part_feat, labels):
        global_feat = F.normalize(global_feat, dim=-1)
        part_feat = F.normalize(part_feat, dim=-1)
        
        B = global_feat.size(0)
        PART_NUM = part_feat.size(1)    
        
        global_feat = global_feat.repeat_interleave(PART_NUM, dim=0)
        part_feat = part_feat.view(B * PART_NUM, -1)
        labels = labels.repeat_interleave(PART_NUM)
        
        dist = torch.matmul(global_feat, part_feat.t())
        
        mask = torch.zeros((B * PART_NUM, B * PART_NUM)).to(global_feat.device)
        pos_mask_tmp = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos_mask = mask.masked_fill(~pos_mask_tmp, -10000.0)
        pos_dist = (dist + pos_mask) / self.T
        
        neg_mask_tmp = labels.unsqueeze(0) != labels.unsqueeze(1)
        neg_mask = mask.masked_fill(~neg_mask_tmp, -10000.0)
        neg_dist1 = dist + neg_mask

        neg_dist = neg_dist1 / self.T

        nominator = torch.logsumexp(pos_dist, dim=-1)
        denominator = torch.log(torch.exp(pos_dist).sum(-1) + torch.exp(neg_dist).sum(-1))
        return torch.mean(denominator - nominator)