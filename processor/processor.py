import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os.path as osp
from scipy.optimize import linear_sum_assignment


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             test_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_val_query, num_test_query, local_rank):
    writer = SummaryWriter(cfg.OUTPUT_DIR)

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None

    M = model.M

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  

    loss_meter = AverageMeter()
    token_contrast_meter = AverageMeter()
    id_loss_meter = AverageMeter()
    part_id_loss_meter = AverageMeter()
    tri_loss_meter = AverageMeter()
    part_tri_loss_meter = AverageMeter()
    part_token_assignment_loss = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_val_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    test_evaluator = R1_mAP_eval(num_test_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        token_contrast_meter.reset()
        id_loss_meter.reset()
        part_id_loss_meter.reset()
        tri_loss_meter.reset()
        part_tri_loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        test_evaluator.reset()
        part_token_assignment_loss.reset()

        scheduler.step()
        model.train()

        for n_iter, (img, texts_list, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            # edit by wtz
            img = img.to(device)
            target = vid.to(device)
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            with amp.autocast(enabled=True):
                score, part_score, feat, part_feat_left, part_feat_right, part_features,text_feat, text_mask = model(img, texts_list, cam_label=target_cam, view_label=target_view)
                loss, (ID_LOSS, PART_ID_LOSS, TRI_LOSS, PART_TRI_LOSS, TOKEN_CONTRAST_LOSS, PART_TOKEN_ASSIGNMENT_LOSS) = loss_fn(score, part_score, feat, part_feat_left, part_feat_right, part_features,text_feat, text_mask, target, target_cam, epoch, M=M)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            token_contrast_meter.update(TOKEN_CONTRAST_LOSS.item(), target.shape[0])
            loss_meter.update(loss.item(), target.shape[0])
            id_loss_meter.update(ID_LOSS.item(), target.shape[0])
            part_id_loss_meter.update(PART_ID_LOSS.item(), target.shape[0])
            tri_loss_meter.update(TRI_LOSS.item(), target.shape[0])
            part_tri_loss_meter.update(PART_TRI_LOSS.item(), target.shape[0])
            part_token_assignment_loss.update(PART_TOKEN_ASSIGNMENT_LOSS.item(), target.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)

        writer.add_scalar('acc/acc', acc_meter.avg, global_step=epoch)
        writer.add_scalar('loss/TOKEN_CONTRAST_LOSS', token_contrast_meter.avg, global_step=epoch)
        writer.add_scalar('loss/ID_LOSS', id_loss_meter.avg, global_step=epoch)
        writer.add_scalar('loss/PART_ID_LOSS', part_id_loss_meter.avg, global_step=epoch)
        writer.add_scalar('loss/TRI_LOSS', tri_loss_meter.avg, global_step=epoch)
        writer.add_scalar('loss/PART_TRI_LOSS', part_tri_loss_meter.avg, global_step=epoch)
        writer.add_scalar('loss/PART_TOK_ASSIGN_LOSS', part_token_assignment_loss.avg, global_step=epoch)


        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                pass
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                pass
            else:
                model.eval()
                for n_iter, (img, texts_list, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        if cfg.MODEL.SIE_CAMERA:
                            camids = camids.to(device)
                        else: 
                            camids = None
                        if cfg.MODEL.SIE_VIEW:
                            target_view = target_view.to(device)
                        else: 
                            target_view = None
                        feat, part_feat, two_feat,_ = model(img, texts_list, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, part_feat, two_feat, vid, camid))
                global_cmc, global_mAP, part_cmc, part_mAP, two_cmc, two_mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Val Global Branch Results ")
                logger.info("mAP: {:.1%}".format(global_mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, global_cmc[r - 1]))
                logger.info("Val Part Branch Results ")
                logger.info("mAP: {:.1%}".format(part_mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, part_cmc[r - 1]))     
                logger.info("Val Two Branch Results ")
                logger.info("mAP: {:.1%}".format(two_mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, two_cmc[r - 1]))                          

                for n_iter, (img, texts_list, vid, camid, camids, target_view, _) in enumerate(test_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        if cfg.MODEL.SIE_CAMERA:
                            camids = camids.to(device)
                        else: 
                            camids = None
                        if cfg.MODEL.SIE_VIEW:
                            target_view = target_view.to(device)
                        else: 
                            target_view = None
                        feat, part_feat, two_feat,_ = model(img, texts_list, cam_label=camids, view_label=target_view)
                        test_evaluator.update((feat, part_feat, two_feat, vid, camid))
                global_cmc, global_mAP, part_cmc, part_mAP, two_cmc, two_mAP, _, _, _, _, _ = test_evaluator.compute()
                logger.info("Test Global Branch Results ")
                logger.info("mAP: {:.1%}".format(global_mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, global_cmc[r - 1]))
                logger.info("Test Part Branch Results ")
                logger.info("mAP: {:.1%}".format(part_mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, part_cmc[r - 1]))     
                logger.info("Test Two Branch Results ")
                logger.info("mAP: {:.1%}".format(two_mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, two_cmc[r - 1]))                                                          
                torch.cuda.empty_cache()

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)

def do_inference(cfg,
                 model,
                 val_loader,
                 num_val_query,
                 test_loader,
                 num_test_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_val_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)
    test_evaluator = R1_mAP_eval(num_test_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, cfg=cfg)

    evaluator.reset()
    test_evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()

    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            # edit by wtz
            feat, part_feat, two_feat, part_features = model(img, cam_label=camids, view_label=target_view)
            # dist_mat = euclidean_dist(part_features.view(-1, part_features.size(-1)), part_features.view(-1, part_features.size(-1)))
            # part_features = get_part_features(dist_mat, part_features, torch.tensor(vid))
            # two_feat = feat + part_features
            evaluator.update((feat, part_feat, two_feat, vid, camid))
    global_cmc, global_mAP, part_cmc, part_mAP, two_cmc, two_mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Val Global Branch Results ")
    logger.info("mAP: {:.1%}".format(global_mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, global_cmc[r - 1]))
    logger.info("Val Part Branch Results ")
    logger.info("mAP: {:.1%}".format(part_mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, part_cmc[r - 1]))     
    logger.info("Val Two Branch Results ")
    logger.info("mAP: {:.1%}".format(two_mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, two_cmc[r - 1]))                          

    img_path_list = []

    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(test_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            # edit by wtz
            feat, part_feat, two_feat, part_features = model(img, cam_label=camids, view_label=target_view)
            # dist_mat = euclidean_dist(part_features.view(-1, part_features.size(-1)), part_features.view(-1, part_features.size(-1)))
            # part_features = get_part_features(dist_mat, part_features, torch.tensor(vid))
            # two_feat = feat + part_features
            test_evaluator.update((feat, part_feat, two_feat, vid, camid))
        img_path_list.extend(_)

    if cfg is not None and cfg.SAVE_NUMPY:
        save_dir = osp.join(cfg.OUTPUT_DIR, 'numpy')
        os.makedirs(save_dir, exist_ok=True)
        np.save(osp.join(save_dir, 'image_paths'), img_path_list)

    global_cmc, global_mAP, part_cmc, part_mAP, two_cmc, two_mAP, _, _, _, _, _ = test_evaluator.compute()
    logger.info("Test Global Branch Results ")
    logger.info("mAP: {:.1%}".format(global_mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, global_cmc[r - 1]))
    logger.info("Test Part Branch Results ")
    logger.info("mAP: {:.1%}".format(part_mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, part_cmc[r - 1]))     
    logger.info("Test Two Branch Results ")
    logger.info("mAP: {:.1%}".format(two_mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, two_cmc[r - 1]))                          
    return global_cmc[0], global_cmc[4]

# edit by wtz
def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n) #B, B
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def get_part_features(dist_mat, part_features, labels):
    B, P, D = part_features.shape  # B=48, P=4, D=1280
    result = torch.zeros(B, P, device=part_features.device)

    for i in range(B):
        indices = torch.arange(i * P, (i + 1) * P) # 当前小块在192中的索引

        # 找到同类的所有 index（包括自己）
        same_class_indices = (labels == labels[i]).nonzero(as_tuple=True)[0]
        same_class_indices = same_class_indices.repeat_interleave(P) * P + torch.arange(P).repeat(len(same_class_indices)).to(labels.device)
        dist = dist_mat[indices]  # shape [4, 192] → 只保留指定行
        dist = dist[:, same_class_indices]  # shape [4, 12] → 只保留指定列

        for j in range(int(dist.shape[1]/P)):
            # 取出第 i 个 4x4 块
            block = dist[:, j * P:(j + 1) * P]  # shape [4, 4]
            # 转 numpy
            block = block.detach().cpu().numpy()
            # 匈牙利匹配
            row_ind, col_ind = linear_sum_assignment(block)
            row = torch.tensor(block[row_ind, col_ind], device=dist_mat.device)
            # 累加匹配代价
            result[i] = result[i] + row
    K=2
    _, idx = torch.topk(-result, k=K, dim=1)
    # 创建 mask
    mask = torch.zeros_like(result, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    part_features = part_features[mask]
    part_features = part_features.view(B, K, D)  # [48, 2, 1280]
    # 取每组平均：
    part_features = part_features.mean(dim=1)

    return part_features