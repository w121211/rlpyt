import os
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
import atari_py
import cv2

from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.int_box import IntBox
from rlpyt.spaces.float_box import FloatBox
from rlpyt.spaces.composite import Composite
from rlpyt.utils.quick_args import save__init__args
from rlpyt.samplers.collections import TrajInfo


W, H = (80, 104)  # Crop two rows, then downsample by 2x (fast, clean image).

ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}

ACTION_INDEX = {v: k for k, v in ACTION_MEANING.items()}


# EnvInfo = namedtuple("EnvInfo", ["game_score", "traj_done"])
EnvInfo = namedtuple("EnvInfo", [])
Obs = namedtuple("Obs", ["a", "b"])

from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.models.pg.atari_ff_model import AtariFfModel
from rlpyt.models.pg.atari_lstm_model import AtariLstmModel


class MyModel(torch.nn.Module):
    def __init__(
        self,
        # image_shape,
        output_size,
        # fc_sizes=512,
        # use_maxpool=False,
        # channels=None,  # None uses default.
        # kernel_sizes=None,
        # strides=None,
        # paddings=None,
    ):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1, 16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(inplace=True),
        )
        self.pi = torch.nn.Linear(64, output_size)
        self.value = torch.nn.Linear(64, 1)

        # self.conv = Conv2dHeadModel(
        #     image_shape=image_shape,
        #     channels=channels or [16, 32],
        #     kernel_sizes=kernel_sizes or [8, 4],
        #     strides=strides or [4, 2],
        #     paddings=paddings or [0, 1],
        #     use_maxpool=use_maxpool,
        #     hidden_sizes=fc_sizes,  # Applies nonlinearity at end.
        # )
        # self.pi = torch.nn.Linear(self.conv.output_size, output_size)
        # self.value = torch.nn.Linear(self.conv.output_size, 1)

    def forward(self, x, prev_action, prev_reward):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""
        print(x)
        # img = image.type(torch.float)  # Expect torch.uint8 inputs
        # img = img.mul_(1.0 / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        # lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        # fc_out = self.conv(img.view(T * B, *img_shape))
        # pi = F.softmax(self.pi(fc_out), dim=-1)
        # v = self.value(fc_out).squeeze(-1)
        # print(x)
        # print(x.shape)
        fc_out = self.fc(x)
        pi = F.softmax(self.pi(fc_out), dim=-1)
        v = self.value(fc_out).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        # pi, v = restore_leading_dims((pi, v), lead_dim, T, B)

        return pi, v


class MyMixin:
    def make_env_to_model_kwargs(self, env_spaces):
        return dict(
            # image_shape=env_spaces.observation.shape,
            output_size=env_spaces.action.n
        )


class MyAgent(MyMixin, CategoricalPgAgent):
    def __init__(self, ModelCls=MyModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class MyEnv(Env):
    def __init__(self):
        self.end_pos = 10
        self.cur_pos = 0
        # self._action_space = IntBox(low=0, high=2, shape=(2,))
        self._action_space = IntBox(low=0, high=2)
        self._observation_space = Composite(
            [FloatBox(low=0, high=self.end_pos), FloatBox(low=0, high=self.end_pos)],
            Obs,
        )

    def reset(self):
        self._step_counter = 0
        self.cur_pos = 0
        # return [self.cur_pos]
        # return {"a": [self.cur_pos], "b": [self.cur_pos]}
        return self.get_obs()

    def step(self, action):
        """
        Returns:
            obs
            reward
            done
            log
        """
        print(type(action))

        # assert action in [0, 1], action
        # if action[0] == 0 and self.cur_pos > 0:
        #     self.cur_pos -= 1
        # elif action[0] == 1:
        #     self.cur_pos += 1
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        done = self.cur_pos >= self.end_pos

        # info = EnvInfo(game_score=game_score, traj_done=game_over)
        info = None
        reward = 1 if done else 0
        self._step_counter += 1
        return EnvStep(self.get_obs(), reward, done, info)

    def get_obs(self):
        # return self._obs.copy()
        # return np.array([self.cur_pos], dtype=np.float32)
        return Obs(
            a=np.array([self.cur_pos], dtype=np.float32),
            b=np.array([self.cur_pos], dtype=np.float32),
        )
