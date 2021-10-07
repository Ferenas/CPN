import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
from tool import pyutils, imutils, torchutils,visualization
import argparse
import importlib
import torch.nn.functional as F
import os


torch.backends.cudnn.benchmark = True

def adaptive_min_pooling_loss(x):
    # This loss does not affect the highest performance, but change the optimial background score (alpha)
    n,c,h,w = x.size()
    k = h*w//4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n,-1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y)/(k*n)
    return loss



if __name__ == '__main__':
    session_name = 'CPN'
    results_path = 'results/' + session_name
    if not os.path.exists(results_path):
        os.makedirs(results_path)


    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--max_epoches", default=8, type=int)
    parser.add_argument("--network", default="network.resnet38_CPN", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--weights", required = True , type=str)  #path for the ilsvrc-cls_rna-a1_cls1000_ep-0001.pth
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--session_name", default=session_name, type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--voc12_root",  required=True, type=str)  #path for the VOC12, .../VOCdevkit/VOC2012

    args = parser.parse_args()

    model = getattr(importlib.import_module(args.network), 'Net')()

    pyutils.Logger(results_path+'/'+args.session_name + '.log')

    print(vars(args))




    train_dataset = voc12.data.VOC12ClsDataset(args.train_list, voc12_root=args.voc12_root,
                                               transform=transforms.Compose([
                        imutils.RandomResizeLong(448, 768),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                        np.asarray,
                        model.normalize,
                        imutils.RandomCrop(args.crop_size),
                        imutils.HWC_to_CHW,
                        torch.from_numpy,
                    ]))

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)

    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches


    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    if args.weights[-7:] == '.params':
        assert args.network == "network.resnet38_cls"
        import network.resnet38d
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    model.cuda()
    model.train()

    avg_meter1 = pyutils.AverageMeter('loss', 'loss_cls', 'loss_tcp', 'loss_cpcr')
    timer = pyutils.Timer("Session started: ")

    for ep in range(args.max_epoches):

        for iter, pack in enumerate(train_data_loader):

            img = pack[1].cuda()
            N, C, H, W = img.size()
            label = pack[2]
            bg_score = torch.ones((N,1))
            label = torch.cat((bg_score, label), dim=1)
            label = label.cuda().unsqueeze(2).unsqueeze(3)
            img1 = pack[3].cuda()
            img2 = pack[4].cuda()
            ctri = pack[5].view(N, 1, 1, 1).repeat(1, 21, 1, 1).type(torch.FloatTensor).cuda()
            uctri = pack[6].view(N, 1, 1, 1).repeat(1, 21, 1, 1).type(torch.FloatTensor).cuda()


            cam,cam_improved = model(img)
            l = F.adaptive_avg_pool2d(cam, (1, 1))
            loss_rvmin = adaptive_min_pooling_loss((cam_improved * label)[:, 1:, :, :])
            cam = visualization.max_norm(cam)*label
            cam_improved = visualization.max_norm(cam_improved)*label

            cam1,cam_improved1 = model(img1)
            l1 = F.adaptive_avg_pool2d(cam1, (1, 1))
            loss_rvmin1 = adaptive_min_pooling_loss((cam_improved1 * label)[:, 1:, :, :])
            cam1 = ctri*visualization.max_norm(cam1)*label
            cam_improved1 = ctri*visualization.max_norm(cam_improved1)*label


            cam2,cam_improved2 = model(img2)
            l2 = F.adaptive_avg_pool2d(cam2,(1,1))
            loss_rvmin2 = adaptive_min_pooling_loss((cam_improved2 * label)[:, 1:, :, :])
            cam2= uctri*visualization.max_norm(cam2) * label
            cam_improved2 = uctri*visualization.max_norm(cam_improved2) * label

            loss_cls = F.multilabel_soft_margin_loss(l[:,1:,:,:],label[:,1:,:,:])
            loss_cls1 = F.multilabel_soft_margin_loss(l1[:,1:,:,:],label[:,1:,:,:])
            loss_cls2 = F.multilabel_soft_margin_loss(l2[:,1:,:,:],label[:,1:,:,:])


            loss_cls = (loss_cls+loss_cls1+loss_cls2) / 3 + (loss_rvmin+loss_rvmin1+loss_rvmin2) / 3


            loss_tcp = torch.mean(torch.abs((cam2[:,1:,:,:].detach() + cam1[:,1:,:,:].detach()) - cam[:,1:,:,:]))
            loss_tcp_improved = torch.mean(torch.abs(cam_improved[:,1:,:,:]-cam_improved1[:,1:,:,:]-cam_improved2[:,1:,:,:]))
            loss_tcp_all = loss_tcp+loss_tcp_improved


            ns, cs, hs, ws = cam.size()
            cam[:, 0, :, :] = 1 - torch.max(cam[:, 1:, :, :], dim=1)[0]
            cam1[:, 0, :, :] = 1 - torch.max(cam1[:, 1:, :, :], dim=1)[0]
            cam2[:, 0, :, :] = 1 - torch.max(cam2[:, 1:, :, :], dim=1)[0]
            cam_improved[:, 0, :, :] = 1 - torch.max(cam_improved[:, 1:, :, :], dim=1)[0]
            cam_improved1[:, 0, :, :] = 1 - torch.max(cam_improved1[:, 1:, :, :], dim=1)[0]
            cam_improved2[:, 0, :, :] = 1 - torch.max(cam_improved2[:, 1:, :, :], dim=1)[0]
            loss_cpcr1 = torch.abs(imutils.max_onehot(cam.detach()-cam1.detach())-cam_improved2)
            loss_cpcr2 = torch.abs(imutils.max_onehot(cam.detach()-cam2.detach())-cam_improved1)



            loss_cpcr1_ohem = torch.mean(torch.topk(loss_cpcr1.view(ns,-1), k=(int)(21*hs*ws*0.2), dim=-1)[0]) #OHEM is 20%
            loss_cpcr2_ohem = torch.mean(torch.topk(loss_cpcr2.view(ns,-1), k=(int)(21*hs*ws*0.2), dim=-1)[0])
            #
            loss_cpcr =  loss_cpcr1_ohem + loss_cpcr2_ohem



            loss = loss_cls+loss_tcp_all+loss_cpcr
            avg_meter1.add({'loss': loss.item(),'loss_cls':loss_cls.item(),'loss_tcp':loss_tcp_all.item(),
                            'loss_cpcr':loss_cpcr.item()
                            })

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if (optimizer.global_step-1)%1 == 0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'loss:%.4f %.4f %.4f %.4f' % avg_meter1.get('loss', 'loss_cls', 'loss_tcp', 'loss_cpcr'),
                      'imps:%.1f' % ((iter+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

                avg_meter1.pop()

        else:
            timer.reset_stage()

    torch.save(model.state_dict(), results_path+'/'+args.session_name +'.pth')
