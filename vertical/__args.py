"""
@Author: Conghao Wong
@Date: 2022-06-23 10:16:04
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-13 10:23:27
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""
import time
import os
from codes.args import BaseArgTable
from codes.utils import dir_check

TIME = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))

class VArgs(BaseArgTable):
    def __init__(self, terminal_args: list[str] = None) -> None:
        super().__init__(terminal_args)

    @property
    def key_points(self) -> str:
        """
        A list of key-time-steps to be predicted in the agent model.
        For example, `'0_6_11'`.
        """
        return self._get('key_points', '0_6_11', argtype='static')

    @property
    def Kc(self) -> int:
        """
        Number of hidden categories used in alpha model.
        """
        return self._get('Kc', 20, argtype='static')

    @property
    def points(self) -> int:
        """
        Controls number of points (representative time steps) input to the beta model.
        It only works when training the beta model only.
        """
        return self._get('points', 1, argtype='static')

    @property
    def feature_dim(self) -> int:
        """
        Feature dimension used in most layers.
        """
        return self._get('feature_dim', 128, argtype='static')

    @property
    def depth(self) -> int:
        """
        Depth of the random noise vector (for random generation).
        """
        return self._get('depth', 16, argtype='static')

    @property
    def K(self) -> int:
        """
        Number of multiple generations when evaluating.
        The number of trajectories predicted for one agent
        is calculated by `N = args.K * args.Kc`,
        where `Kc` is the number of style channels.
        """
        return self._get('K', 1, argtype='dynamic')

    @property
    def K_train(self) -> int:
        """
        Number of multiple generations when training.
        """
        return self._get('K_train', 1, argtype='static')

    @property
    def preprocess(self) -> str:
        """
        Controls if running any preprocess before model inference.
        Accept a 3-bit-like string value (like `'111'`):
        - the first bit: `MOVE` trajectories to (0, 0);
        - the second bit: re-`SCALE` trajectories;
        - the third bit: `ROTATE` trajectories.
        """
        return self._get('preprocess', '111', argtype='static')

    @property
    def keypoints_loss_type(self):
        """
        Type of loss function for key points (VA).
        """
        return self._get('keypoints_loss_type', 'ade', argtype='static')

    @property
    def loss_weights_b(self) -> list[float]:
        """
        Type of loss function for key points (VA).
        """
        value = self._get('loss_weights_b', [0.0], argtype='static')
        return value

    @property
    def loss_weights_a(self) -> list[float]:
        """
        Type of loss function for key points (VA).
        """
        value = list(map(float, self._get('loss_weights_a', '1.0', argtype='static').split(',')))
        return value

    @property
    def log_dir(self) -> str:
        """
        Folder to save training logs and models. If set to `null`,
        logs will save at `args.save_base_dir/current_model`.
        """
        loss_weights = (('_a-' + '_'.join(map(str, self.loss_weights_a))) if self.loss_weights_a is not None else "") + (
                ('_b-' + '_'.join(map(str, self.loss_weights_b))) if self.loss_weights_b is not None else '' )
        log_dir_current = (TIME +
                           self.model_name +
                           self.model +
                           self.test_set +
                           loss_weights)
        default_log_dir = os.path.join(dir_check(self.save_base_dir),
                                       log_dir_current)
        return self._get('log_dir', dir_check(default_log_dir), argtype='static')
