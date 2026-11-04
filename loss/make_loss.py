# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com

Modified:
- ✅ align 不再每 iteration new
- ✅ align 是模型参数（挂到 model 上），会进 optimizer
- temperature 固定 0.07（不学习 logit_scale）
"""

import torch
import torch.nn.functional as F

from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss
from .token_triplet_loss import TokenTripletLoss, PartTokenAssignmentLoss
from .center_loss import CenterLoss
from .token_info_nce_loss import TokenInfoNCE
from .token_info_nce_loss_max import TokenInfoNCEMax


def make_loss(cfg, num_classes, model=None):
    """
    需要把 model 传进来（model 内部有 itc_align / itc_loss）：
      loss_func, center_criterion = make_loss(cfg, num_classes, model)
    """
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)

    # token contrast selector
    part_token_assignment = None
    if cfg.MODEL.TOKEN_CONTRAST_TYPE == 'sum':
        token_contrast = TokenInfoNCE(cfg.SOLVER.INFO_NCE_T)
    elif cfg.MODEL.TOKEN_CONTRAST_TYPE == 'max':
        token_contrast = TokenInfoNCEMax(cfg.SOLVER.INFO_NCE_T)
    elif cfg.MODEL.TOKEN_CONTRAST_TYPE == 'triplet':
        if cfg.MODEL.NO_TOKEN_MARGIN:
            token_contrast = TokenTripletLoss()
        else:
            token_contrast = TokenTripletLoss(cfg.SOLVER.TOKEN_MARGIN)
            part_token_assignment = PartTokenAssignmentLoss(cfg.SOLVER.TOKEN_MARGIN)
    else:
        token_contrast = TokenInfoNCE(cfg.SOLVER.INFO_NCE_T)

    # metric loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
        triplet = TripletLoss(cfg.SOLVER.MARGIN)

    # ID loss
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)
    else:
        xent = None

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif sampler == 'softmax_triplet':

        def loss_func(
            score, part_score,
            feat, part_feat_left, part_feat_right, part_features,
            text_feat, text_mask,
            target, target_cam, epoch,
            M=None, i2tscore=None
        ):
            if cfg.MODEL.METRIC_LOSS_TYPE != 'triplet':
                print('expected METRIC_LOSS_TYPE should be triplet but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
                return None

            # =========================
            # Label smoothing branch
            # =========================
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                # ---- ID loss ----
                if isinstance(score, list):
                    ID_LOSS = sum([xent(scor, target) for scor in score[0:]])
                else:
                    ID_LOSS = xent(score, target)

                if isinstance(part_score, list):
                    PART_ID_LOSS = sum([xent(scor, target) for scor in part_score[0:]])
                else:
                    PART_ID_LOSS = xent(part_score, target)

                # ---- Triplet loss ----
                if isinstance(feat, list):
                    TRI_LOSS = sum([triplet(feats, feats, target)[0] for feats in feat[0:]])
                else:
                    TRI_LOSS = triplet(feat, feat, target)[0]

                if isinstance(part_feat_left, list):
                    PART_TRI_LOSS_LEFT = sum([triplet(feats, feats, target)[0] for feats in part_feat_left[0:]])
                else:
                    PART_TRI_LOSS_LEFT = triplet(part_feat_left, part_feat_left, target)[0]

                if isinstance(part_feat_right, list):
                    PART_TRI_LOSS_RIGHT = sum([triplet(feats, feats, target)[0] for feats in part_feat_right[0:]])
                else:
                    PART_TRI_LOSS_RIGHT = triplet(part_feat_right, part_feat_right, target)[0]

                # ---- Token contrast ----
                TOKEN_CONTRAST_LOSS = sum([token_contrast(f, pf, target) for f, pf in list(zip(feat, part_features))])

                # ---- Part token assignment (optional) ----
                PART_TOKEN_ASSIGNMENT_LEFT_LOSS = torch.tensor(0.0, device=target.device)
                PART_TOKEN_ASSIGNMENT_RIGHT_LOSS = torch.tensor(0.0, device=target.device)
                if part_token_assignment is not None:
                    PART_TOKEN_ASSIGNMENT_LEFT_LOSS = sum([
                        part_token_assignment(pf[:, :4, :].contiguous(), target, top_k=cfg.MODEL.TOP_K)
                        for pf in part_features[0:]
                    ])
                    PART_TOKEN_ASSIGNMENT_RIGHT_LOSS = sum([
                        part_token_assignment(pf[:, 4:, :].contiguous(), target, top_k=cfg.MODEL.TOP_K)
                        for pf in part_features[0:]
                    ])

                # ---- weights (keep original) ----
                ID_LOSS = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS
                TRI_LOSS = cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                PART_ID_LOSS = cfg.MODEL.PART_ID_LOSS_WEIGHT * PART_ID_LOSS * 1.5
                PART_TRI_LOSS_LEFT = cfg.MODEL.PART_TRIPLET_LOSS_WEIGHT * PART_TRI_LOSS_LEFT
                PART_TRI_LOSS_RIGHT = cfg.MODEL.PART_TRIPLET_LOSS_WEIGHT * PART_TRI_LOSS_RIGHT

                TOKEN_CONTRAST_LOSS = cfg.MODEL.TOKEN_CONTRAST_LOSS_WEIGHT * TOKEN_CONTRAST_LOSS

                PART_TOKEN_ASSIGNMENT_LEFT_LOSS = 100 * PART_TOKEN_ASSIGNMENT_LEFT_LOSS
                PART_TOKEN_ASSIGNMENT_RIGHT_LOSS = 100 * PART_TOKEN_ASSIGNMENT_RIGHT_LOSS

                # =========================
                # ITC loss (NO per-iter new)
                # =========================
                IMG_TXT_LOSS = torch.tensor(0.0, device=target.device)
                if text_feat is not None and (model is not None) and hasattr(model, "itc_align") and hasattr(model, "itc_loss"):
                    align = model.itc_align
                    itc = model.itc_loss

                    img_feats_aligned = align.forward_img([feat[0], feat[1], feat[2]])  # 3x [B,512]
                    txt_aligned = align.forward_txt(text_feat)                          # [B,512] or [B,K,512]

                    for img_a in img_feats_aligned:
                        IMG_TXT_LOSS = IMG_TXT_LOSS + itc(img_a, txt_aligned, text_mask=text_mask)

                    IMG_TXT_LOSS = IMG_TXT_LOSS / len(img_feats_aligned)
                    IMG_TXT_LOSS = cfg.MODEL.text * IMG_TXT_LOSS

                loss = (
                    ID_LOSS + PART_ID_LOSS + TRI_LOSS +
                    PART_TRI_LOSS_LEFT + PART_TRI_LOSS_RIGHT +
                    TOKEN_CONTRAST_LOSS +
                    PART_TOKEN_ASSIGNMENT_LEFT_LOSS + PART_TOKEN_ASSIGNMENT_RIGHT_LOSS +
                    IMG_TXT_LOSS
                )

                return loss, (
                    ID_LOSS,
                    PART_ID_LOSS,
                    TRI_LOSS,
                    PART_TRI_LOSS_LEFT,
                    TOKEN_CONTRAST_LOSS,
                    IMG_TXT_LOSS
                )

            # =========================
            # Non label smoothing branch
            # =========================
            else:
                if isinstance(score, list):
                    ID_LOSS = sum([F.cross_entropy(scor, target) for scor in score[0:]])
                else:
                    ID_LOSS = F.cross_entropy(score, target)

                if isinstance(feat, list):
                    TRI_LOSS = sum([triplet(feats, target)[0] for feats in feat[0:]])
                else:
                    TRI_LOSS = triplet(feat, target)[0]

                loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                if i2tscore is not None:
                    I2TLOSS = F.cross_entropy(i2tscore, target)
                    loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss

                return loss

    else:
        print('expected sampler should be softmax, softmax_triplet '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
        loss_func = None

    return loss_func, center_criterion