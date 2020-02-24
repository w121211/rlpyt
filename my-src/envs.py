import os
from collections import namedtuple

import numpy as np
from PIL import Image, ImageDraw

from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.int_box import IntBox
from rlpyt.spaces.float_box import FloatBox
from rlpyt.spaces.composite import Composite
from rlpyt.utils.quick_args import save__init__args
from rlpyt.samplers.collections import TrajInfo

CANVAS_WIDTH = 64
OBJ_WIDTH = 10
N_ITEMS = 1
MAX_STEP = 100
ACTION_MOVE = np.array([0, -8, 8, -1, 1, -8, 8, -1, 1], dtype=np.float32) / CANVAS_WIDTH
HOR_MOVE = np.array([0, -8, 8, -1, 1, 0, 0, 0, 0], dtype=np.float32) / CANVAS_WIDTH
VER_MOVE = np.array([0, 0, 0, 0, 0, -8, 8, -1, 1], dtype=np.float32) / CANVAS_WIDTH
N_ACTIONS = len(ACTION_MOVE)

EnvInfo = namedtuple("EnvInfo", [])
Obs = namedtuple("Obs", ["target_im", "cur_im", "cur_coord"])


class CanvasEnv(Env):
    def __init__(self):
        self.cur_step = 0
        self.width = CANVAS_WIDTH
        self.obj_wh = np.array([OBJ_WIDTH, OBJ_WIDTH], dtype=np.float32) / self.width
        self.n_items = N_ITEMS

        self._action_space = IntBox(low=0, high=N_ACTIONS)
        self._observation_space = Composite(
            [
                FloatBox(low=-1, high=1, shape=(self.width, self.width, 1)),
                FloatBox(low=-1, high=1, shape=(self.width, self.width, 1)),
                FloatBox(low=-10, high=10, shape=(self.n_items, 4)),
            ],
            Obs,
        )
        # self._observation_space = FloatBox(
        #     low=-1, high=1, shape=(self.width, self.width, 1)
        # )
        
        self.target_im = np.zeros(shape=(self.width, self.width, 1), dtype=np.float32)
        self.target_coord = None  # (n_obj, 4=(x0, y0, x1, y1))
        self.cur_im = None
        self.cur_coord = None  # (n_obj, 4=(x0, y0, x1, y1))
        self.item = None

    def reset(self):
        self.cur_step = 0
        xy0 = np.random.rand(self.n_items, 2)
        self.target_coord = np.concatenate(
            [xy0, xy0 + np.tile(self.obj_wh, (self.n_items, 1))], axis=1
        )
        self.cur_coord = np.tile(
            np.array([0, 0, *tuple(self.obj_wh)], dtype=np.float32), (self.n_items, 1)
        )
        self.target_im = self._render(self.target_coord)
        self.cur_im = self._render(self.cur_coord)
        return self._obs()

    def step(self, action: np.ndarray):
        """
        Args:
            action: [int, int]
        Return:
            obs: target_im (H, W, C), cur_im (H, W, C), field_info (x0, y0)
        """
        # print(self.action_map[action[0]], self.action_map[action[1]])
        # idx = action[0]
        # print(action)
        i_item = 0
        # action = action.reshape((N_ACTIONS / 2, 2))
        dmove = np.array([HOR_MOVE[action], VER_MOVE[action]], dtype=np.float32)
        xy0 = self.cur_coord[i_item, :2] + dmove
        self.cur_coord[i_item] = np.concatenate((xy0, xy0 + self.obj_wh), axis=0)
        self.cur_im = self._render(self.cur_coord)

        reward = self._reward(self.cur_coord, self.target_coord)
        done = self.cur_step >= MAX_STEP
        info = EnvInfo()
        self.cur_step += 1

        # return self._obs(), reward, done, {}
        return EnvStep(self._obs(), reward, done, info)

    def _obs(self):
        # return np.transpose(self.target_im.copy(), (2, 0, 1))
        return Obs(
            target_im=np.transpose(self.target_im.copy(), (2, 0, 1)),
            cur_im=np.transpose(self.cur_im.copy(), (2, 0, 1)),
            cur_coord=self.cur_coord.copy(),
        )

    def _render(self, coord: np.ndarray):
        """
        Args: coord: (n_items, 4)
        """
        coord = (coord * self.width).astype(np.int16)
        im = Image.new("L", (self.width, self.width))
        draw = ImageDraw.Draw(im)
        for i, c in enumerate(coord):
            if i == 0:
                draw.rectangle(tuple(c), fill=255)
            else:
                draw.ellipse(tuple(c), fill=255)
        x = np.array(im, dtype=np.float32) / 255.0  # normalize
        x = np.expand_dims(x, axis=-1)  # (H, W, C=1)
        return x

    def _reward(self, xy_a: np.array, xy_b: np.array):
        dist = np.linalg.norm(xy_a - xy_b, axis=1)
        r = -1 * dist / 2 + 1
        r = np.clip(r, -1, None)
        r = np.sum(r)
        #     elif r > 0:
        #         r *= 0.05 ** self.cur_step  # 衰退因子
        return r

    def _denorm(self, a: np.array):
        return (a * self.width).astype(np.int16)



