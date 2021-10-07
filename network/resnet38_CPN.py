import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
import numpy as np
np.set_printoptions(threshold=np.inf)
import network.resnet38d
from network.da_atten import SpatialGather_Module


class Net(network.resnet38d.Net):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout7 = torch.nn.Dropout2d(0.5)

        self.fc8 = nn.Conv2d(4096, 21, 1, bias=False)

        self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        self.f9 = torch.nn.Conv2d(192+3, 192, 1, bias=False)
        self.f10 = torch.nn.Conv2d(192+3, 192, 1, bias=False)

        self.spatial_gather=SpatialGather_Module()
        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)
        torch.nn.init.xavier_uniform_(self.f10.weight, gain=4)
        self.from_scratch_layers = [self.f8_3, self.f8_4, self.f9, self.fc8,self.f10]+[l for l in self.spatial_gather.modules()] #+ [l for l in self.p2.modules()] +
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]

    def forward(self, x):
        N, C, H, W = x.size()
        d = super().forward_as_dict(x)
        cam = self.fc8(self.dropout7(d['conv6']))
        n,c,h,w = cam.size()
        with torch.no_grad():
            cam_d = F.relu(cam.detach())
            cam_d_max = torch.max(cam_d.view(n,c,-1), dim=-1)[0].view(n,c,1,1)+1e-5
            cam_d_norm = F.relu(cam_d-1e-5)/cam_d_max
            cam_d_norm[:,0,:,:] = 1-torch.max(cam_d_norm[:,1:,:,:], dim=1)[0]
            cam_max = torch.max(cam_d_norm[:, 1:, :, :], dim=1, keepdim=True)[0]
            cam_d_norm[:, 1:, :, :][cam_d_norm[:, 1:, :, :] < cam_max] = 0


        f8_33 = d['conv4']
        f8_44 = d['conv5']
        f8_3 = F.relu(self.f8_3(f8_33.detach()), inplace=True)
        f8_4 = F.relu(self.f8_4(f8_44.detach()), inplace=True)
        x_s = F.interpolate(x,(h,w),mode='bilinear',align_corners=True)
        f = torch.cat([x_s, f8_3, f8_4], dim=1)

        fea_p = self.PRCM(self.spatial_gather(f,cam_d_norm),f)
        fea_final =  self.PCM(cam_d_norm,f)  + fea_p*cam_d_norm


        cam_rv = F.interpolate(fea_final, (H,W), mode='bilinear', align_corners=True)
        cam = F.interpolate(cam, (H,W), mode='bilinear', align_corners=True)
        return cam,cam_rv

    def PCM(self, cam, f):
        n, c, h, w = f.size()
        cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h * w)
        f = self.f9(f)
        f = f.view(n, -1, h * w)
        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5)  # 1-Norm of f

        aff = F.relu(torch.matmul(f.transpose(1, 2), f), inplace=True)
        aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-5)
        cam_rv = torch.matmul(cam, aff).view(n, -1, h, w)

        return cam_rv

    def PRCM(self, cam, f):
        n,c,h,w = f.size()
        f = F.relu(self.f9(f))
        f = f.view(n,-1,h*w)
        key = F.relu(self.f10(cam)).view(n, 21, -1)
        sim_map = torch.matmul(key,f)
        sim_map = (21**-.5) * sim_map
        cam_rv= F.softmax(sim_map, dim=-1).view(n,-1,h,w)


        return cam_rv

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups

