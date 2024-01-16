import os
import torch
import argparse
from tqdm import tqdm
import random
import numpy as np
from eval import eval_for_metric
from losses.get_losses import SelectLoss
from models.block.Drop import dropblock_step
from utils.dataloaders import get_loaders
from utils.common import check_dirs, init_seed, gpu_info, SaveResult, CosOneCycle, ScaleInOutput
from models.main_model import ChangeDetection, ModelEMA, ModelSWA


def train(opt):
    init_seed()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    gpu_info()  # 打印GPU信息
    save_path, best_ckp_save_path, best_ckp_file, result_save_path, every_ckp_save_path = check_dirs()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_results = SaveResult(result_save_path)
    save_results.prepare()

    train_loader, val_loader = get_loaders(opt)
    scale = ScaleInOutput(opt.input_size)

    model = ChangeDetection(opt).cuda()
    if torch.cuda.device_count() > 1:
        # model = torch.nn.DataParallel(model)
        model = torch.nn.DataParallel(model,device_ids = [0,1,2,3])
    criterion = SelectLoss(opt.loss)

    if opt.finetune:
        params = [{"params": [param for name, param in model.named_parameters()
                              if "backbone" in name], "lr": opt.learning_rate / 10}, 
                  {"params": [param for name, param in model.named_parameters()
                              if "backbone" not in name], "lr": opt.learning_rate}]
        print("Using finetune for model")
    else:
        params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=opt.learning_rate, weight_decay=0.01)
    #optimizer = torch.optim.SGD(params, lr=opt.learning_rate, weight_decay=0.5)
    if opt.pseudo_label:
        scheduler = CosOneCycle(optimizer, max_lr=opt.learning_rate/5, epochs=opt.epochs, up_rate=0)
    else:
        scheduler = CosOneCycle(optimizer, max_lr=opt.learning_rate, epochs=opt.epochs)

    best_metric = 0
    train_avg_loss = 0
    total_bs = 32
    accumulate_iter = max(round(total_bs / opt.batch_size), 1)
    print("Accumulate_iter={} batch_size={}".format(accumulate_iter, opt.batch_size))

    for epoch in range(opt.epochs):
        model.train()
        train_tbar = tqdm(train_loader)
        for i, (batch_img1, batch_img2, batch_label, batch_label2, _) in enumerate(train_tbar):
            train_tbar.set_description("epoch {}, train_loss {}".format(epoch, train_avg_loss))
            if epoch == 0 and i < 20:
                save_results.save_first_batch(batch_img1, batch_img2, batch_label, batch_label2, i)

            batch_img1 = batch_img1.float().cuda()
            batch_img2 = batch_img2.float().cuda()
            batch_label = batch_label.long().cuda()
            batch_label2 = batch_label2.long().cuda()

            batch_img1, batch_img2 = scale.scale_input((batch_img1, batch_img2))
            outs,mask1,mask2 = model(batch_img1, batch_img2)
            outs = scale.scale_output(outs)
            mask1 = scale.scale_output(mask1)
            mask2 = scale.scale_output(mask2)

            loss = criterion(outs, (batch_label,)) + 0.9*criterion(mask1, (batch_label,)) + 0.9*criterion(mask2, (batch_label,))
            # loss = criterion(outs, (batch_label,))
            train_avg_loss = (train_avg_loss * i + loss.cpu().detach().numpy()) / (i + 1)

            loss.backward()
            if ((i+1) % accumulate_iter) == 0:
                optimizer.step()
                optimizer.zero_grad()

            del batch_img1, batch_img2, batch_label, batch_label2

        scheduler.step()
        dropblock_step(model)

        p, r, f1, miou, oa, val_avg_loss = eval_for_metric(model, val_loader, criterion, input_size=opt.input_size)

        # refer_metric = f1
        refer_metric = f1
        underscore = "_"
        if refer_metric.mean() > best_metric and refer_metric.mean() < 0.9131:
            if best_ckp_file is not None:
                os.remove(best_ckp_file)
            best_ckp_file = os.path.join(
                best_ckp_save_path,
                underscore.join([opt.backbone, opt.neck, opt.head, 'epoch',
                                 str(epoch), str(round(float(refer_metric.mean()), 5))]) + ".pt")
            torch.save(model, best_ckp_file)
            best_metric = refer_metric.mean()

        # 写日志
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        save_results.show(p, r, f1, miou, oa, refer_metric, best_metric, train_avg_loss, val_avg_loss, lr, epoch)

def set_randomness():
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Change Detection train')


    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--neck", type=str, default="fpn+aspp+fuse+drop")
    parser.add_argument("--head", type=str, default="fcn")
    parser.add_argument("--loss", type=str, default="bce+dice")


    parser.add_argument("--pretrain", type=str,
                        default="") 
    parser.add_argument("--cuda", type=str, default="")
    parser.add_argument("--dataset-dir", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--dual-label", type=bool, default=False)
    parser.add_argument("--finetune", type=bool, default=True)
    parser.add_argument("--pseudo-label", type=bool, default=True)

    opt = parser.parse_args()
    print(opt)
    set_randomness()
    train(opt)
