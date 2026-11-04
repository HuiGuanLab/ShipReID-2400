import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import copy


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def clip_tokenize(texts, context_length=77, truncate=True):
    """
    texts: List[str] or str
    return: LongTensor [N, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot = _tokenizer.encoder["<|startoftext|>"]
    eot = _tokenizer.encoder["<|endoftext|>"]

    all_tokens = []
    for t in texts:
        tokens = [sot] + _tokenizer.encode(t) + [eot]
        if truncate and len(tokens) > context_length:
            tokens = tokens[:context_length]
            tokens[-1] = eot
        all_tokens.append(tokens)

    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    for i, tokens in enumerate(all_tokens):
        result[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return result


def _l2n(x, dim=-1, eps=1e-12):
    return F.normalize(x, dim=dim, eps=eps)


class MultiDimAlign(nn.Module):
    """
    Align multiple img_feat dims to a shared embedding dim.
    Example dims: [768, 768, 512] -> out_dim=512
    """
    def __init__(self, img_dims, txt_dim, out_dim=512, share_768=False):
        super().__init__()
        self.out_dim = out_dim

        if share_768:
            self.img_proj_768 = nn.Linear(768, out_dim, bias=False)
            self.img_proj_512 = nn.Linear(512, out_dim, bias=False)
        else:
            self.img_proj = nn.ModuleList([
                nn.Linear(img_dims[0], out_dim, bias=False),
                nn.Linear(img_dims[1], out_dim, bias=False),
                nn.Linear(img_dims[2], out_dim, bias=False),
            ])

        self.txt_proj = nn.Linear(txt_dim, out_dim, bias=False)

    def forward_img(self, img_feats):
        outs = []
        if hasattr(self, "img_proj"):
            for feat, proj in zip(img_feats, self.img_proj):
                outs.append(proj(feat))
        else:
            for feat in img_feats:
                if feat.size(-1) == 768:
                    outs.append(self.img_proj_768(feat))
                elif feat.size(-1) == 512:
                    outs.append(self.img_proj_512(feat))
                else:
                    raise ValueError(f"Unexpected img dim: {feat.size(-1)}")
        return outs

    def forward_txt(self, txt_feat):
        # txt_feat: (B,D) or (B,K,D)
        if txt_feat is None:
            return None
        if txt_feat.dim() == 2:
            return self.txt_proj(txt_feat)
        elif txt_feat.dim() == 3:
            B, K, D = txt_feat.shape
            return self.txt_proj(txt_feat.reshape(B * K, D)).reshape(B, K, self.out_dim)
        else:
            raise ValueError("txt_feat must be 2D or 3D")


class ITCLoss(nn.Module):
    """
    Image-Text Contrastive Loss with FIXED temperature (0.07 by default).
    No learnable logit_scale.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, image_feat, text_feat, text_mask=None):
        if image_feat is None or text_feat is None:
            return image_feat.new_tensor(0.0) if image_feat is not None else torch.tensor(0.0)

        B = image_feat.size(0)
        device = image_feat.device

        img = _l2n(image_feat)
        scale = (1.0 / self.temperature).to(device=device)

        # single-positive
        if text_feat.dim() == 2:
            txt = _l2n(text_feat)
            logits = scale * (img @ txt.t())
            labels = torch.arange(B, device=device)
            return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

        # multi-positive
        B2, K, D = text_feat.shape
        assert B2 == B
        txt = _l2n(text_feat)
        text_flat = txt.reshape(B * K, D)
        logits_i2t = scale * (img @ text_flat.t())

        if text_mask is None:
            text_mask = torch.ones((B, K), dtype=torch.bool, device=device)
        else:
            if text_mask.dim() == 1:
                text_mask = text_mask[:, None]
            text_mask = text_mask.to(device=device, dtype=torch.bool)

        valid_text_flat = text_mask.reshape(-1)

        pos = torch.zeros((B, B * K), dtype=torch.bool, device=device)
        for i in range(B):
            valid_k = text_mask[i].nonzero(as_tuple=False).view(-1)
            if valid_k.numel() == 0:
                continue
            pos[i, i * K + valid_k] = True

        logits_i2t = logits_i2t.masked_fill(~valid_text_flat[None, :], float("-inf"))
        log_prob = F.log_softmax(logits_i2t, dim=1)
        pos_logsumexp = torch.logsumexp(log_prob.masked_fill(~pos, float("-inf")), dim=1)

        valid_row = pos.any(dim=1)
        loss_i2t = -pos_logsumexp[valid_row].mean() if valid_row.any() else logits_i2t.new_tensor(0.0)

        logits_t2i = scale * (text_flat @ img.t())
        owner = torch.arange(B, device=device).repeat_interleave(K)
        loss_t2i = (
            F.cross_entropy(logits_t2i[valid_text_flat], owner[valid_text_flat])
            if valid_text_flat.any() else logits_t2i.new_tensor(0.0)
        )

        return 0.5 * (loss_i2t + loss_t2i)


# for Part_Attention
def generate_2d_mask(M=16, H=16, W=8, left=0, top=0, width=8, height=8, part=-1, cls_label=True, device='cuda'):
    H, W, left, top, width, height = int(H), int(W), int(left), int(top), int(width), int(height)
    if part > 3:
        width = 16
        height = 4
    assert left + width <= W and top + height <= H
    l, w = left, left + width
    t, h = top, top + height
    mask = torch.zeros([H, W], device=device)
    mask[t:h, l:w] = 1
    mask = mask.flatten(0)
    mask_ = torch.zeros([len(mask) + M], device=device)
    mask_[M:] = mask
    mask_[part] = 1
    mask_ = mask_.unsqueeze(1)  # N x 1
    mask_ = mask_ @ mask_.t()   # N x N
    return mask_


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.in_planes_proj = 512

        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)
        self.part_classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.part_classifier.apply(weights_init_classifier)
        self.part_classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.part_classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.part_bottleneck = nn.BatchNorm1d(self.in_planes)
        self.part_bottleneck.bias.requires_grad_(False)
        self.part_bottleneck.apply(weights_init_kaiming)
        self.part_bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.part_bottleneck_proj.bias.requires_grad_(False)
        self.part_bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]

        self.part_w = cfg.MODEL.PART_W
        self.part_h = cfg.MODEL.PART_H
        self.coords = self.generate_top_left_coords()
        self.M = len(self.coords)

        self.num_patches = self.h_resolution * self.w_resolution + self.M

        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")
        self.image_encoder = clip_model.visual

        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size, self.M)
        clip_model.to("cuda")
        self.part_image_encoder = clip_model.visual

        self.part_ratio = cfg.MODEL.PART_RATIO

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        # --- save RNG state ---
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all()
        np_rng = np.random.get_state()
        import random
        py_rng = random.getstate()

        # ============================================================
        # ✅ 关键改动：把 align / itc 作为 nn.Module 挂到模型里
        # 这样 optimizer(model.parameters()) 会自动包含它们 => 可训练
        # ============================================================
        self.itc_align = MultiDimAlign(img_dims=[768, 768, 512], txt_dim=512, out_dim=512, share_768=False)
        self.itc_loss = ITCLoss(temperature=0.07)

        # --- build text model ---
        clip_text_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_text_model.to("cuda")
        self.text_encoder = clip_text_model
        self.text_context_length = 77

        # --- restore RNG state ---
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(np_rng)
        random.setstate(py_rng)

        # default freeze text branch (more stable)
        self.freeze_text = True
        if self.freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def _encode_text_flat(self, flat_texts, device):
        tokens = clip_tokenize(flat_texts, context_length=self.text_context_length).to(device)
        txt = self.text_encoder.encode_text(tokens)
        txt = txt.float()
        txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-6)
        return txt

    def encode_texts(self, texts_list, device):
        if texts_list is None:
            return None, None

        # normalize to List[List[str]]
        if len(texts_list) > 0 and isinstance(texts_list[0], str):
            texts_list = [[t] for t in texts_list]

        B = len(texts_list)
        K = max(len(ts) for ts in texts_list) if B > 0 else 0

        flat_texts = []
        owners = []
        for i, ts in enumerate(texts_list):
            for t in ts:
                flat_texts.append(t)
                owners.append(i)

        if len(flat_texts) == 0:
            return None, None

        if self.freeze_text:
            txt_flat = self._encode_text_flat(flat_texts, device)
        else:
            tokens = clip_tokenize(flat_texts, context_length=self.text_context_length).to(device)
            txt_flat = self.text_encoder.encode_text(tokens)
            txt_flat = txt_flat.float()
            txt_flat = txt_flat / (txt_flat.norm(dim=-1, keepdim=True) + 1e-6)

        D = txt_flat.shape[-1]
        text_feat = txt_flat.new_zeros((B, K, D))
        text_mask = torch.zeros((B, K), dtype=torch.bool, device=device)

        cnt = [0] * B
        for j, i in enumerate(owners):
            k = cnt[i]
            text_feat[i, k] = txt_flat[j]
            text_mask[i, k] = True
            cnt[i] += 1

        if K == 1:
            return text_feat[:, 0], text_mask[:, 0]
        return text_feat, text_mask

    def forward(self, x, texts_list=None, cam_label=None, view_label=None):
        if cam_label is not None and view_label is not None:
            cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
        elif cam_label is not None:
            cv_embed = self.sie_coe * self.cv_embed[cam_label]
        elif view_label is not None:
            cv_embed = self.sie_coe * self.cv_embed[view_label]
        else:
            cv_embed = None

        B = x.size(0)
        image_features_last, image_features, image_features_proj, attn = self.image_encoder(x, cv_embed)
        img_feature_last = image_features_last[:, 0]
        img_feature = image_features[:, 0]
        img_feature_proj = image_features_proj[:, 0]

        mask = self.attn_mask_generate(self.num_patches, self.h_resolution, self.w_resolution,
                                       self.part_h, self.part_w, x.device.type)
        mask = mask.unsqueeze(0).repeat(B, 1, 1)
        feature_selection_mask = self.feature_selection(attn)
        mask[:, self.M:, self.M:] = mask[:, self.M:, self.M:] & feature_selection_mask
        mask = ~mask

        part_image_features_last, part_image_features, part_image_features_proj, _ = self.part_image_encoder(
            x, cv_embed, attn_mask=mask
        )
        part_img_feature_last = part_image_features_last[:, : self.M].mean(1)
        part_img_feature = part_image_features[:, : self.M].mean(1)
        part_img_feature_proj = part_image_features_proj[:, : self.M].mean(1)

        part_img_feature_last_left = part_image_features_last[:, :4].mean(1)
        part_img_feature_left = part_image_features[:, :4].mean(1)
        part_img_feature_proj_left = part_image_features_proj[:, :4].mean(1)

        part_img_feature_last_right = part_image_features_last[:, 4:8].mean(1)
        part_img_feature_right = part_image_features[:, 4:8].mean(1)
        part_img_feature_proj_right = part_image_features_proj[:, 4:8].mean(1)

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        part_feat = self.part_bottleneck(part_img_feature)
        part_feat_proj = self.part_bottleneck_proj(part_img_feature_proj)

        text_feat, text_mask = self.encode_texts(texts_list, device=x.device) if texts_list is not None else (None, None)

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            part_cls_score = self.part_classifier(part_feat)
            part_cls_score_proj = self.part_classifier_proj(part_feat_proj)

            part_features = [
                part_image_features_last[:, : self.M].contiguous(),
                part_image_features[:, : self.M].contiguous(),
                part_image_features_proj[:, : self.M].contiguous()
            ]

            return [cls_score, cls_score_proj], [part_cls_score, part_cls_score_proj], \
                   [img_feature_last, img_feature, img_feature_proj], \
                   [part_img_feature_last_left, part_img_feature_left, part_img_feature_proj_left], \
                   [part_img_feature_last_right, part_img_feature_right, part_img_feature_proj_right], \
                   part_features, text_feat, text_mask
        else:
            if self.neck_feat == 'after':
                return torch.cat([feat, feat_proj], dim=-1), \
                       torch.cat([part_feat, part_feat_proj], dim=-1), \
                       torch.cat([feat + part_feat, feat_proj + part_feat_proj], dim=-1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=-1), \
                       torch.cat([part_img_feature, part_img_feature_proj], dim=-1), \
                       torch.cat([img_feature + part_img_feature, img_feature_proj + part_img_feature_proj], dim=-1), \
                       torch.cat([part_image_features, part_image_features_proj], dim=-1)

    def attn_mask_generate(self, N=273, H=16, W=16, window_h=4, window_w=4, device='cuda'):
        mask = torch.ones(N, 1, device=device)
        M = len(self.coords)
        mask[:M, 0] = 0
        mask_ = (mask @ mask.t()).bool()
        for idx, (top, left) in enumerate(self.coords, start=0):
            mask_ |= generate_2d_mask(M, H, W, left, top, window_w, window_h, idx, False, device).bool()
        return mask_

    def generate_top_left_coords(self, device='cuda'):
        cols = torch.arange(0, self.w_resolution, self.part_w)
        rows = torch.arange(0, self.h_resolution, self.part_h)
        grid_rows, grid_cols = torch.meshgrid(rows, cols)
        coords = torch.stack((grid_rows.flatten(), grid_cols.flatten()), dim=-1).to(device)
        return torch.tensor([[0, 0], [0, 4], [0, 8], [0, 12],
                             [0, 0], [4, 0], [8, 0], [12, 0]], device=coords.device)

    def feature_selection(self, attn):
        B = attn.size(0)
        attn = attn[:, 0, 1:]
        k = int((attn.size(1) - 1) * self.part_ratio)
        attn_topk = attn.topk(dim=-1, k=k)[1]
        mask_ = torch.zeros(B, attn.size(1)).to(attn.device)
        mask_ = mask_.scatter(1, attn_topk, 1)
        mask_ = (mask_.unsqueeze(-1) @ mask_.unsqueeze(-2)).bool()
        return mask_

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size, M=None):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size, M)
    return model