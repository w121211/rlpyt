import torch
import torch.nn as nn
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dHeadModel


class ConvEncoder(nn.Module):
    def __init__(self, image_shape=(1, 128, 128), z_size=128):
        super(ConvEncoder, self).__init__()
        c, h, w = image_shape
        self.output_size = z_size

        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, 4, 2, 1),  # 16,32,32
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1),  # 32,16,16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),  # 32,16,16 -> 64,8,8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 64,8,8 -> 128,4,4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.AvgPool2d(kernel_size=2),
        )
        out_size = w // 2 ** 4
        self.fc = nn.Linear(128 * out_size ** 2, z_size)

    def forward(self, x):
        N = x.size(0)
        x = self.conv(x).view(N, -1)
        x = self.fc(x)
        return x


class MyModel(nn.Module):
    def __init__(self, image_shape, output_size):
        super().__init__()
        self.conv = ConvEncoder(image_shape=image_shape)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.conv.output_size * 2, 128),
            # torch.nn.Linear(self.conv.output_size, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(inplace=True),
        )
        self.pi = torch.nn.Linear(128, output_size)
        self.value = torch.nn.Linear(128, 1)

    def forward(self, obs, prev_action, prev_reward):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""
        # print(obs.shape)
        # print(obs.target_im.shape)
        # print(obs.cur_im.shape)
        # x1 = self.conv(obs.target_im)
        # x2 = self.conv(obs.cur_im)

        # obs.cur_coord
        lead_dim, T, B, img_shape = infer_leading_dims(obs.target_im, 3)

        x1 = self.conv(obs.target_im.view(T * B, *img_shape))
        x2 = self.conv(obs.cur_im.view(T * B, *img_shape))
        x = torch.cat((x1, x2), dim=1)

        # x = self.conv(obs.view(T * B, *img_shape))
        x = self.fc(x.view(T * B, -1))

        pi = F.softmax(self.pi(x), dim=-1)
        v = self.value(x).squeeze(-1)
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)

        return pi, v

