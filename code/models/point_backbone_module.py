import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule

class Pointnet2Backbone(nn.Module):
    def __init__(self, input_feature_dim=0):
        super(Pointnet2Backbone, self).__init__()

        self.SA1 = PointnetSAModuleVotes(
            npoint=1024,
            radius=0.1,
            nsample=32,
            mlp=[input_feature_dim, 32, 32, 64],
            use_xyz=True,
            normalize_xyz=True
        )

        self.SA2 = PointnetSAModuleVotes(
            npoint=512,
            radius=0.2,
            nsample=32,
            mlp=[64, 64, 64, 128],
            use_xyz=True,
            normalize_xyz=True
        )

        self.SA3 = PointnetSAModuleVotes(
            npoint=256,
            radius=0.4,
            nsample=16,
            mlp=[128, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.SA4 = PointnetSAModuleVotes(
            npoint=64,
            radius=0.1,
            nsample=32,
            mlp=[256, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.FP1 = PointnetFPModule(mlp=[256+256, 256, 256])
        self.FP2 = PointnetFPModule(mlp=[128+256, 128, 256])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud:torch.cuda.FloatTensor, end_points=None):

        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)

        xyz, features, fps_inds = self.SA1(xyz, features)
        end_points['SA1_inds'] = fps_inds
        end_points['SA1_xyz'] = xyz
        end_points['SA1_features'] = features

        xyz, features, fps_inds = self.SA2(xyz, features)
        end_points['SA2_inds'] = fps_inds
        end_points['SA2_xyz'] = xyz
        end_points['SA2_features'] = features

        xyz, features, fps_inds = self.SA3(xyz, features)
        end_points['SA3_xyz'] = xyz
        end_points['SA3_features'] = features

        xyz, features, fps_inds = self.SA4(xyz, features)
        end_points['SA4_xyz'] = xyz
        end_points['SA4_features'] = features

        features = self.FP1(end_points['SA3_xyz'], end_points['SA4_xyz'], end_points['SA3_features'], end_points['SA4_features'])
        features = self.FP2(end_points['SA2_xyz'], end_points['SA3_xyz'], end_points['SA2_features'], features)
        end_points['FP2_features'] = features
        end_points['FP2_xyz'] = end_points['SA2_xyz']
        num_seed = end_points['FP2_xyz'].shape[1]
        end_points['FP2_inds'] = end_points['SA1_inds'][:,0:num_seed]
        return end_points


if __name__=='__main__':
    backbone_net = Pointnet2Backbone(input_feature_dim=3).cuda()
    print(backbone_net)
    backbone_net.eval()
    out = backbone_net(torch.rand(16, 20000, 6).cuda())
    for key in sorted(out.keys()):
        print(key, '\t', out[key].shape)










