import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
import voc12.data
import scipy.misc
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils, pyutils
import argparse
from PIL import Image
import torch.nn.functional as F
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="results/CPN/CPN.pth", type=str) #results/ss-sp38-try3-0.2-nconvex-atten3/ss-sp38-try3-0.2-nconvex-atten3-7.pth
    parser.add_argument("--network", default="network.resnet38_CPN", type=str)
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--voc12_root", default ="/home/szbl/zf/data/VOCdevkit/VOC2012", type=str)  #The root path for VOC12    '.../VOCdevkit/VOC2012'
    parser.add_argument("--out_cam", default='out_cam', type=str) #contains the foreground pixels for CAM

    args = parser.parse_args()

    if not os.path.exists(args.out_cam):
        os.mkdir(args.out_cam)

    model = getattr(importlib.import_module(args.network), 'Net')()
    model.load_state_dict(torch.load(args.weights),strict = False)

    model.eval()
    model.cuda()

    infer_dataset = voc12.data.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                                   scales=[0.5,1.0,1.5,2.0],
                                                   inter_transform=torchvision.transforms.Compose(
                                                       [np.asarray,
                                                        model.normalize,
                                                        imutils.HWC_to_CHW,
                                                    ]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)


    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):


            img_name = img_name[0]; label = label[0]
            img_path = voc12.data.get_img_path(img_name, args.voc12_root)
            orig_img = np.asarray(Image.open(img_path))
            orig_img_size = orig_img.shape[:2]

            def _work(i, img):
                with torch.no_grad():
                        _,cam = model.forward(img.cuda())
                        cam = F.upsample(cam[:,1:,:,:], orig_img_size, mode='bilinear', align_corners=False)[0]
                        cam = cam.cpu().numpy() * label.clone().view(20, 1, 1).numpy()
                        if i % 2 == 1:
                            cam = np.flip(cam, axis=-1)
                        return cam
            #
            thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list)),
                                                batch_size=12, prefetch_size=0, processes=args.num_workers)

            cam_list = thread_pool.pop_results()
            sum_cam = np.sum(cam_list, axis=0)
            sum_cam[sum_cam < 0] = 0
            cam_max = np.max(sum_cam, (1,2), keepdims=True)
            cam_min = np.min(sum_cam, (1,2), keepdims=True)
            sum_cam[sum_cam < cam_min+1e-5] = 0
            norm_cam = (sum_cam-cam_min-1e-5) / (cam_max - cam_min + 1e-5)

            #This compresses all the foregrond seed values so that saves time for the best background searching
            norm_cam = norm_cam / 2

            cam_dict = {}
            for i in range(20):
                if label[i] > 1e-5:
                    cam_dict[i] = norm_cam[i]

            if args.out_cam is not None:
                np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)



            print(iter)

