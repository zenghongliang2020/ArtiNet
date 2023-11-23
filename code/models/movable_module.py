import torch
import torch.nn as nn
import torch.nn.functional as F

from rgb_backbone_module import CNN_Backbone
from point_backbone_module import Pointnet2Backbone

class movable_net(nn.Module):
    def __init__(self, input_feature_dim=0, sampling='vote_fps'):
        super(movable_net, self).__init__()

        self.input_feature_dim = input_feature_dim
        self.sampling = sampling

        self.rgb_backbone = CNN_Backbone()
        self.point_backbone = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        self.fc = nn.Linear(256 * 512, 256)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 + 256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, input):
        img = input['img']
        point_cloud = input['point_clouds']
        img_feature = self.rgb_backbone(img)
        end_points = {}
        end_points = self.point_backbone(point_cloud, end_points)
        point_feature = self.fc(self.flatten(end_points['FP2_features']))
        features = torch.cat((img_feature, point_feature), dim=1)
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__=='__main__':
    model = movable_net(1).cuda()
    input = {'img': torch.rand(2, 4, 256, 256).cuda(), 'point_clouds': torch.rand(2, 100, 3).cuda()}
    print(input['point_clouds'].dtype)
    out = model(input)
    print(out)

