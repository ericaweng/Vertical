"""
@Author: Conghao Wong
@Date: 2022-06-20 16:27:21
@LastEditors: Conghao Wong
@LastEditTime: 2022-07-07 09:11:48
@Description: Structures to train trajectory prediction models.
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ..__base import BaseObject
from ..args import BaseArgTable
from ..basemodels import Model
from ..dataset import Agent, Dataset, DatasetManager, get_inputs_by_type
from ..utils import dir_check
from . import __loss as losslib
from .__vis import Visualization


class Structure(BaseObject):

    def __init__(self, terminal_args: list[str]):
        super().__init__()

        self.args = BaseArgTable(terminal_args)
        self.model: Model = None
        self.important_args = ['lr', 'test_set']

        self.set_gpu()
        self.optimizer = self.set_optimizer()

        self.set_inputs('obs')
        self.set_labels('pred')

        self.set_loss('ade')
        self.set_loss_weights(1.0)
        self.set_metrics('ade', 'fde')
        self.set_metrics_weights(1.0, 0.0)

    def set_inputs(self, *args):
        """
        Set inputs to the model.
        Accept keywords:
        ```python
        historical_trajectory = ['traj', 'obs']
        groundtruth_trajectory = ['pred', 'gt']
        context_map = ['map']
        context_map_paras = ['para', 'map_para']
        destination = ['des', 'inten']
        ```

        :param input_names: type = `str`, accept several keywords
        """
        self.model_inputs = []
        for item in args:
            if 'traj' in item or \
                    'obs' in item:
                self.model_inputs.append('TRAJ')

            elif 'para' in item or \
                    'map_para' in item:
                self.model_inputs.append('MAPPARA')

            elif 'context' in item or \
                    'map' in item:
                self.model_inputs.append('MAP')

            elif 'des' in item or \
                    'inten' in item:
                self.model_inputs.append('DEST')

            elif 'gt' in item or \
                    'pred' in item:
                self.model_inputs.append('GT')
        self.model_inputs.append('SEQNAME')
        self.model_inputs.append('FRAMEID')
        self.model_inputs.append('PEDID')

    def set_labels(self, *args):
        """
        Set ground truths of the model.
        Accept keywords:
        ```python
        groundtruth_trajectory = ['traj', 'pred', 'gt']
        destination = ['des', 'inten']
        ```

        :param input_names: type = `str`, accept several keywords
        """
        self.model_labels = []
        for item in args:
            if 'traj' in item or \
                'gt' in item or \
                    'pred' in item:
                self.model_labels.append('GT')

            elif 'des' in item or \
                    'inten' in item:
                self.model_labels.append('DEST')

    def set_loss(self, *args):
        self.loss_list = [arg for arg in args]

    def set_loss_weights(self, *args: list[float]):
        self.loss_weights = [arg for arg in args]

    def set_metrics(self, *args):
        self.metrics_list = [arg for arg in args]

    def set_metrics_weights(self, *args: list[float]):
        self.metrics_weights = [arg for arg in args]

    def set_optimizer(self, *args, **kwargs) -> tf.keras.optimizers.Optimizer:
        return tf.keras.optimizers.Adam(learning_rate=self.args.lr)

    def set_gpu(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu.replace('_', ',')
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    def load_model_weights(self, weights_path: str,
                           *args, **kwargs) -> Model:
        """
        Load already trained model weights from checkpoint files.
        Additional parameters will be fed to `create_model` method.

        :param model_path: path of model
        :return model: loaded model
        """
        model = self.create_model(*args, **kwargs)
        model.load_weights(weights_path)
        return model

    def load_best_model(self, model_path: str,
                        *args, **kwargs) -> Model:
        """
        Load already trained models from saved files.
        Additional parameters will be fed to `create_model` method.

        :param model_path: target dir where your model puts in
        :return model: model loaded
        """

        dir_list = os.listdir(model_path)
        save_format = '.tf'
        try:
            name_list = [item.split(save_format)[0].split(
                '_epoch')[0] for item in dir_list if save_format in item]
            model_name = max(name_list, key=name_list.count)
            base_path = os.path.join(model_path, model_name + '{}')

            if 'best_ade_epoch.txt' in dir_list:
                best_epoch = np.loadtxt(os.path.join(model_path, 'best_ade_epoch.txt'))[
                    1].astype(int)
                model = self.load_model_weights(base_path.format(
                    '_epoch{}{}'.format(best_epoch, save_format)),
                    *args, **kwargs)
            else:
                model = self.load_model_weights(base_path.format(save_format),
                                                *args, **kwargs)

        except:
            model_name = name_list[0]
            base_path = os.path.join(model_path, model_name + save_format)
            model = self.load_model_weights(base_path, *args, **kwargs)

        self.model = model
        return model

    def load_dataset(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Load training and val dataset.

        :return dataset_train: train dataset, type = `tf.data.Dataset`
        :return dataset_val: val dataset, type = `tf.data.Dataset`
        """
        train_agents, test_agents = DatasetManager.load(self.args,
                                                        dataset='auto',
                                                        mode='train')

        train_data = self.load_inputs_from_agents(train_agents)
        test_data = self.load_inputs_from_agents(test_agents)

        train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_data)

        train_dataset = train_dataset.shuffle(len(train_dataset),
                                              reshuffle_each_iteration=True)

        return train_dataset, test_dataset

    def load_test_dataset(self, agents: list[Agent]) -> tf.data.Dataset:
        """
        Load test dataset.
        """
        test_data = self.load_inputs_from_agents(agents)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
        return test_dataset

    def load_inputs_from_agents(self, agents: list[Agent]) -> list[tf.Tensor]:
        """
        Load model inputs and labels from agents.

        :param agents: a list of `Agent` objects        
        """
        inputs = [get_inputs_by_type(agents, T) for T in self.model_inputs]
        labels = [get_inputs_by_type(agents, T) for T in self.model_labels][0]

        inputs.append(labels)
        return tuple(inputs)

    def create_model(self, *args, **kwargs) -> Model:
        """
        Create models.
        Please *rewrite* this when subclassing new models.

        :return model: created model
        :return optimizer: training optimizer
        """
        raise NotImplementedError('MODEL is not defined!')

    def save_model_weights(self, save_path: str):
        """
        Save trained model to `save_path`.

        :param save_path: where model saved.
        """
        self.model.save_weights(save_path)

    def loss(self, outputs: list[tf.Tensor],
             labels: tf.Tensor,
             *args, **kwargs) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
        """
        Train loss, use ADE (point-wise l2 loss) by default.

        :param outputs: model's outputs
        :param labels: groundtruth labels

        :return loss: sum of all single loss functions
        :return loss_dict: a dict of all losses
        """
        return losslib.apply(self.loss_list,
                             outputs,
                             labels,
                             self.loss_weights,
                             mode='loss',
                             *args, **kwargs)

    def metrics(self, outputs: list[tf.Tensor],
                labels: tf.Tensor) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
        """
        Metrics, use [ADE, FDE] by default.
        Use ADE as the validation item.

        :param outputs: model's outputs, a list of tensor
        :param labels: groundtruth labels

        :return loss: sum of all single loss functions
        :return loss_dict: a dict of all losses
        """
        return losslib.apply(self.metrics_list,
                             outputs,
                             labels,
                             self.metrics_weights,
                             mode='m')

    def gradient_operations(self, inputs: list[tf.Tensor],
                            labels: tf.Tensor,
                            loss_move_average: tf.Variable,
                            *args, **kwargs) -> tuple[tf.Tensor, dict[str, tf.Tensor], tf.Tensor]:
        """
        Run gradient dencent once during training.

        :param inputs: model inputs, a list of tensors
        :param labels: ground truth
        :param loss_move_average: moving average loss

        :return loss: sum of all single loss functions
        :return loss_dict: a dict of all loss functions
        :return loss_move_average: moving average loss
        """

        with tf.GradientTape() as tape:
            outputs = self.model.forward(inputs, training=True)
            loss, loss_dict = self.loss(outputs, labels,
                                        inputs=inputs, *args, **kwargs)

            loss_move_average = 0.7 * loss + 0.3 * loss_move_average

        grads = tape.gradient(loss_move_average,
                              self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,
                                           self.model.trainable_variables))

        return loss, loss_dict, loss_move_average

    def model_validate(self, inputs: list[tf.Tensor],
                       labels: tf.Tensor,
                       training=None) -> tuple[list[tf.Tensor], tf.Tensor, dict[str, tf.Tensor]]:
        """
        Run one step of forward and calculate metrics.

        :param inputs: model inputs
        :param labels :ground truth

        :return model_output: model output
        :return metrics: weighted sum of all loss 
        :return loss_dict: a dict contains all loss
        """

        outpus = self.model.forward(inputs, training)
        metrics, metrics_dict = self.metrics(outpus, labels)

        return outpus, metrics, metrics_dict

    def train_or_test(self):
        """
        Load args, load datasets, and start training or test.
        """
        # start training if not loading any model weights
        if self.args.load == 'null':
            self.model = self.create_model()

            # restore weights before training (optional)
            if self.args.restore != 'null':
                self.load_best_model(self.args.restore)

            self.log('Start training with args = {}'.format(self.args))
            self.__train()

        # prepare test
        else:
            self.log('Start test `{}`'.format(self.args.load))
            self.load_best_model(self.args.load)
            self.run_test()

    def run_test(self):
        """
        Run test accoding to arguments.
        """

        if self.args.test_mode == 'one':
            try:
                agents = DatasetManager.load(self.args,
                                             ds := self.args.force_set,
                                             mode='test')
            except:
                ds = Dataset(self.args.test_set).test_sets[0]
                agents = DatasetManager.load(self.args, ds, mode='test')

            self.__test(agents=agents, dataset=ds)
            return

        test_sets = Dataset(self.args.test_set).test_sets
        if self.args.test_mode == 'all':
            for ds in test_sets:
                agents = DatasetManager.load(self.args, ds, mode='test')
                self.__test(agents, ds)

        elif self.args.test_mode == 'mix':
            agents = []
            for ds in test_sets:
                agents += DatasetManager.load(self.args, ds, mode='test')

            self.__test(agents, dataset=self.args.test_set)

        else:
            raise ValueError(self.args.test_mode)

    def __train(self):
        """
        Training
        """

        # print training infomation
        self.print_dataset_info()
        self.__print_train_info()

        # make log directory and save current args
        self.args._save_as_json(dir_check(self.args.log_dir))

        # open tensorboard
        tb = tf.summary.create_file_writer(self.args.log_dir)

        # init variables for training
        best_metrics = 10000.0
        best_epoch = 0
        loss_move = tf.Variable(0, dtype=tf.float32)
        loss_dict = {}
        test_epochs = []

        # Load dataset
        ds_train, ds_val = self.load_dataset()
        train_number = len(ds_train)

        # divide with batch size
        ds_train = ds_train.repeat(
            self.args.epochs).batch(self.args.batch_size)

        # start training
        timebar = self.log_timebar(inputs=ds_train,
                                   text='Training...',
                                   return_enumerate=False)
        for batch_id, dat in enumerate(timebar):

            epoch = (batch_id * self.args.batch_size) // train_number
            loss, loss_dict, loss_move = self.gradient_operations(
                # inputs=dat[:-1]
                inputs=dat[:-4],  # -4 bc don't want seq_name, frame_id, ped_id, and labels
                labels=dat[-1],
                loss_move_average=loss_move,
                epoch=epoch,
            )

            # Check if `nan` in loss dictionary
            if tf.math.is_nan(loss):
                self.log(e := 'Find `nan` values in the loss dictionary, stop training...',
                         level='error')
                raise ValueError(e)

            # Run validation
            if ((epoch >= self.args.start_test_percent * self.args.epochs)
                    and ((epoch - 1) % self.args.test_step == 0)
                    and (not epoch in test_epochs)
                    and (epoch > 0)):

                metrics, metrics_dict = self.__test_on_dataset(
                    ds=ds_val,
                    return_results=False,
                    show_timebar=False
                )

                test_epochs.append(epoch)

                # Save model
                if metrics <= best_metrics:
                    best_metrics = metrics
                    best_epoch = epoch

                    self.save_model_weights(os.path.join(
                        self.args.log_dir, '{}_epoch{}.tf'.format(
                            self.args.model_name,
                            epoch)))

                    np.savetxt(os.path.join(self.args.log_dir, 'best_ade_epoch.txt'),
                               np.array([best_metrics, best_epoch]))

            # Update time bar
            step_dict = dict(zip(['epoch', 'best'], [epoch, best_metrics]))
            try:
                loss_dict = dict(
                    step_dict, **dict(loss_dict, **metrics_dict))
            except UnboundLocalError as e:
                loss_dict = dict(step_dict, **loss_dict)

            for key in loss_dict:
                if issubclass(type(loss_dict[key]), tf.Tensor):
                    loss_dict[key] = loss_dict[key].numpy()
            timebar.set_postfix(loss_dict)

            # Write tensorboard
            with tb.as_default():
                for loss_name in loss_dict:
                    value = loss_dict[loss_name]
                    tf.summary.scalar(loss_name, value, step=epoch)

        self.print_train_results(best_epoch=best_epoch,
                                 best_metric=best_metrics)

    def __test(self, agents: list[Agent], dataset: str):
        """
        Test
        """

        # Print test information
        self.__print_test_info()

        # Load dataset
        ds_test = self.load_test_dataset(agents)

        # Run test
        outputs, labels, obses, frame_ids, ped_ids, seq_names, metrics, metrics_dict = self.__test_on_dataset(
            ds=ds_test,
            return_results=True,
            show_timebar=True,
        )
        # import time
        # t_s = time.time()
        # metrics, metrics_dict = self.__test_on_dataset(
        #     ds=ds_test,
        # )
        # dt = time.time() - t_s
        # print(f"dt: {dt:0.3f} seconds")
        # import ipdb; ipdb.set_trace()

        # Write test results
        self.print_test_results(metrics_dict, dataset)

        # model_inputs_all = list(ds_test.as_numpy_iterator())
        outputs = stack_results(outputs)
        labels = stack_results(labels)
        obses = stack_results(obses)
        frame_ids = stack_results(frame_ids)
        ped_ids = stack_results(ped_ids)
        seq_names = stack_results(seq_names)

        self.write_test_results(outputs=outputs,
                                agents=agents,
                                dataset=dataset)

        # sanity checks for same dataset size as the standard
        dset_to_num_peds = {
            'eth': 364,
            'hotel': 1197, # but has 1075
            'univ': 14295 + 10039,
            'zara1': 2356,
            'zara2': 5910,
            'sdd': 2829,
        }
        dset_to_num_frames = {
            'eth': 253,
            'hotel': 445, # but has 1075
            'univ': 425 + 522,
            'zara1': 705,
            'zara2': 998,
            'sdd': 1999,
        }
        SEQUENCE_NAMES = {
            'eth': ['biwi_eth'],
            'hotel': ['biwi_hotel'],
            'zara1': ['crowds_zara01'],
            'zara2': ['crowds_zara02'],
            'univ': ['students001', 'students003'],
            'sdd': [
                'coupa_0', 'coupa_1', 'gates_2', 'hyang_0', 'hyang_1', 'hyang_3',
                'hyang_8', 'little_0', 'little_1', 'little_2', 'little_3',
                'nexus_5', 'nexus_6', 'quad_0', 'quad_1', 'quad_2', 'quad_3',
            ],
        }
        assert dset_to_num_peds[dataset] == len(outputs[0]), \
            f'{dataset} should have {dset_to_num_peds[dataset]} peds but has {len(outputs[0])}'

        vv_to_standard_seq_name = {
            'biwi_eth': 'eth',
            'biwi_hotel': 'hotel',
            'crowds_zara01': 'zara1',
            'crowds_zara02': 'zara2',
            'students001': 'univ',
            'students003': 'univ3',
            **dict(zip(SEQUENCE_NAMES['sdd'], [seq.replace('_', '') for seq in SEQUENCE_NAMES['sdd']]))
        }
        assert len(seq_names) == len(ped_ids) == len(frame_ids) == len(outputs) == len(labels) == len(obses) == 1, \
            f'len(seq_names) ({len(seq_names)}) != len(ped_ids) ({len(ped_ids)}) != len(frame_ids) ({len(frame_ids)})' \
            f' != len(outputs) ({len(outputs)}) != len(labels) ({len(labels)}) != len(obses) ({len(obses)}) != 1'
        assert len(outputs) == len(labels) == len(obses) == len(frame_ids) == len(ped_ids) == len(seq_names) == 1, \
            "all should be of length 1"
        seq_names = seq_names[0]
        ped_ids = ped_ids[0]
        frame_ids = frame_ids[0]
        outputs = outputs[0]
        labels = labels[0]
        obses = obses[0]

        seq_name_to_frames = {}
        for seq in SEQUENCE_NAMES[dataset]:
            idxs = np.where(seq_names == vv_to_standard_seq_name[seq])[0]
            out = tf.gather(outputs, idxs)
            lbl = tf.gather(labels, idxs)
            obs = tf.gather(obses, idxs)
            fram = tf.gather(frame_ids, idxs)
            ped = tf.gather(ped_ids, idxs)
            # gather same frames together
            unique_final_obs_frames_sorted = sorted(np.unique(fram[:, 7].numpy()))
            frames_this_seq = []
            for frame in unique_final_obs_frames_sorted:
                idxs = np.where(fram[:, 7] == frame)[0]  # 7 is the last obs step
                assert len(idxs) > 0
                outp = tf.gather(out, idxs)
                labl = tf.gather(lbl, idxs)
                obse = tf.gather(obs, idxs)
                fra = tf.gather(fram, idxs)
                pede = tf.gather(ped, idxs)
                frames_this_seq.append([outp, labl, obse, fra, pede])
            seq_name_to_frames[seq] = frames_this_seq
            # save_seqs.append(sorted(list(zip(out, lbl, obs, fram, ped)), key=lambda x: (x[-1], x[-2][-1])))
            # seq_name_to_sorted_items[seq] = sorted(list(zip(out, lbl, obs, fram, ped)), key=lambda x: (x[-1], x[-2][-1]))  # get the last frame

        # save trajectories to standard format
        save_path = f'../../results/trajectories/{self.args.save_traj_dir}'
        print("check save path:", save_path)
        if dataset == 'sdd':
            save_path += '/trajnet_sdd'

        for seq_name, frames in seq_name_to_frames.items():
            for out, lbl, obs, frame_ids, ped_ids in tqdm(frames, desc=f'Saving trajs in {seq_name}...', total=len(frames)):
                assert np.all([np.all(frame_ids[0].numpy() == f for f in frame_ids)]), \
                    f'all 20 frame_ids should be the same for all peds but are {frame_ids}'
                frame_ids = frame_ids[0].numpy()
                dset_outputs = out
                dset_labels = lbl
                if dataset == 'sdd':
                    obs = obs * 100
                    dset_outputs = dset_outputs * 100
                    dset_labels = dset_labels * 100

                # check that the seqs metadata and agents match
                pred = np.flip(dset_outputs.numpy().transpose(1,0,2,3), -1)  # --> (20, num_peds, 12, 2) + flip x and y
                flattened_gt = self.flatten_scene(np.flip(dset_labels.numpy(), -1),  # --> (num_peds, 12, 2)
                                                  frame_ids[self.args.obs_frames:], ped_ids)
                flattened_obs = self.flatten_scene(np.flip(obs.numpy(), -1),  # --> (num_peds, 8, 2)
                                                   frame_ids[:self.args.obs_frames], ped_ids)

                # todo save correct ped_ids and frame_ids
                for sample_i, sample in enumerate(pred):
                    flattened_peds = self.flatten_scene(sample, frame_ids[self.args.obs_frames:], ped_ids)
                    self.save_trajectories(flattened_peds, save_path, seq_name, frame_ids[self.args.obs_frames-1],
                                           suffix=f'/sample_{sample_i:03d}')
                self.save_trajectories(flattened_gt, save_path, seq_name, frame_ids[self.args.obs_frames-1], suffix='/gt')
                self.save_trajectories(flattened_obs, save_path, seq_name, frame_ids[self.args.obs_frames - 1], suffix='/obs')
        print(f"saved trajectories to {save_path}")

    @staticmethod
    def flatten_scene(trajs_arr, frame_nums=None, ped_nums=None, frame_skip=10):
        """flattens a 3d scene of shape (num_peds, ts, 2) into a 2d array of shape (num_peds, ts x 4)
        ped_nums (optional): list of ped numbers to assign, length == num_peds
        frame_nums (optional): list of frame numbers to assign to the resulting, length == ts
        """
        if ped_nums is None:
            ped_nums = np.arange(0, trajs_arr.shape[1])
        if frame_nums is None:
            frame_nums = np.arange(0, trajs_arr.shape[0] * frame_skip, frame_skip)
        ped_ids = np.tile(np.array(ped_nums).reshape(-1, 1), (1, trajs_arr.shape[1])).reshape(-1, 1)
        frame_ids = np.tile(np.array(frame_nums).reshape(1, -1), (trajs_arr.shape[0], 1)).reshape(-1, 1)
        trajs_arr = np.concatenate([frame_ids, ped_ids, trajs_arr.reshape(-1, 2)], -1)
        return trajs_arr

    @staticmethod
    def save_trajectories(trajectory, save_dir, seq_name, frame, suffix=''):
        """Save trajectories in a text file.
        Input:
            trajectory: (np.array/torch.Tensor) Predcited trajectories with shape
                        of (future_timesteps * n_pedestrian, 4). The last element is
                        [frame_id, track_id, x, y] where each element is float.
            save_dir: (str) Directory to save into.
            seq_name: (str) Sequence name (e.g., eth_biwi, coupa_0)
            frame: (num) Frame ID.
            suffix: (str) Additional suffix to put into file name.
        """

        fname = f"{save_dir}/{seq_name}/frame_{int(frame):06d}{suffix}.txt"
        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        np.savetxt(fname, trajectory, fmt="%s")

    def __test_on_dataset(self, ds: tf.data.Dataset,
                          return_results=False,
                          show_timebar=False):

        # init variables for test
        outputs_all = []
        obs_all = []
        labels_all = []
        metrics_all = []
        frame_ids_all, ped_ids_all, seq_names_all = [], [], []
        metrics_dict_all = {}

        # divide with batch size
        ds_batch = ds.batch(self.args.batch_size)

        timebar = self.log_timebar(
            ds_batch,
            text='Test...',
            return_enumerate=False) if show_timebar else ds_batch

        test_numbers = []
        import time
        ds = time.time()
        i = 0
        for dat in timebar:
            outputs, metrics, metrics_dict = self.model_validate(
                # inputs=dat[:-1],  # -4 bc don't want seq_name, frame_id, ped_id, and labels
                inputs=dat[:-4],  # -4 bc don't want seq_name, frame_id, ped_id, and labels
                labels=dat[-1],
                training=False,
            )
            i += 1
            if i % 20 == 0:
                print(f"Tested {i} batches in {time.time() - ds} seconds")
            test_numbers.append(outputs[0].shape[0])

            if return_results:
                seq_names_all = append_results_to_list(dat[-4:-3], seq_names_all)
                frame_ids_all = append_results_to_list(dat[-3:-2], frame_ids_all)
                ped_ids_all = append_results_to_list(dat[-2:-1], ped_ids_all)
                obs_all = append_results_to_list(dat[:1], obs_all)
                outputs_all = append_results_to_list(outputs, outputs_all)
                labels_all = append_results_to_list(dat[-1:], labels_all)

            # add metrics to metrics dict
            metrics_all.append(metrics)
            for key, value in metrics_dict.items():
                if not key in metrics_dict_all.keys():
                    metrics_dict_all[key] = []
                metrics_dict_all[key].append(value)

        # calculate average metric
        weights = tf.cast(tf.stack(test_numbers), tf.float32)
        metrics_all = \
            (tf.reduce_sum(tf.stack(metrics_all) * weights) /
             tf.reduce_sum(weights)).numpy()

        for key in metrics_dict_all:
            metrics_dict_all[key] = \
                (tf.reduce_sum(tf.stack(metrics_dict_all[key]) * weights) /
                 tf.reduce_sum(weights)).numpy()

        if return_results:
            return outputs_all, labels_all, obs_all, frame_ids_all, ped_ids_all, seq_names_all, metrics_all, metrics_dict_all
        else:
            return metrics_all, metrics_dict_all

    def __get_important_args(self):
        args_dict = dict(zip(
            self.important_args,
            [getattr(self.args, n) for n in self.important_args]))
        return args_dict

    def __print_train_info(self):
        args_dict = self.__get_important_args()
        self.print_parameters(title='Training Options',
                              model_name=self.args.model_name,
                              batch_size=self.args.batch_size,
                              epoch=self.args.epochs,
                              gpu=self.args.gpu,
                              **args_dict)
        self.print_train_info()

    def __print_test_info(self):
        args_dict = self.__get_important_args()
        self.print_parameters(title='Test Options',
                              model_name=self.args.model_name,
                              batch_size=self.args.batch_size,
                              gpu=self.args.gpu,
                              **args_dict)
        self.print_test_info()

    def print_dataset_info(self):
        self.print_parameters(title='dataset options')

    def print_train_info(self):
        """
        Information to show (or to log into files) before training
        """
        pass

    def print_test_info(self):
        """
        Information to show (or to log into files) before testing
        """
        pass

    def print_train_results(self, best_epoch: int, best_metric: float):
        """
        Information to show (or to log into files) after training
        """
        self.log('Training done.')
        self.log('During training, the model reaches the best metric ' +
                 '`{}` at epoch {}.'.format(best_metric, best_epoch))

        self.log(('Tensorboard file is saved at `{}`. ' +
                  'To open this log file, please use `tensorboard ' +
                  '--logdir {}`').format(self.args.log_dir,
                                         self.args.log_dir))
        self.log(('Trained model is saved at `{}`. ' +
                  'To re-test this model, please use ' +
                  '`python main.py --load {}`.').format(self.args.log_dir,
                                                        self.args.log_dir))

    def print_test_results(self, loss_dict: dict[str, float], dataset: str):
        """
        Information to show (or to log into files) after testing
        """
        self.print_parameters(title='Test Results',
                              dataset=dataset,
                              **loss_dict)
        self.log('{}, {}, {}\n'.format(
                 self.args.load,
                 dataset,
                 loss_dict))

    def write_test_results(self, outputs: list[tf.Tensor],
                           agents: dict[str, list[Agent]],
                           dataset: str):

        if self.args.draw_results and (not dataset.startswith('mix')):
            # draw results on video frames
            tv = Visualization(dataset=dataset)
            save_base_path = dir_check(self.args.log_dir) \
                if self.args.load == 'null' \
                else self.args.load

            save_format = os.path.join(dir_check(os.path.join(
                save_base_path, 'VisualTrajs')), '{}_{}.{}')

            self.log('Start saving images at {}'.format(
                os.path.join(save_base_path, 'VisualTrajs')))

            for index, agent in self.log_timebar(agents, 'Saving...'):
                # write traj
                output = outputs[0][index].numpy()
                agents[index].pred = output

                # draw as one image
                tv.draw(agents=[agent],
                        frame_name=float(
                            agent.frames[self.args.obs_frames]),
                        save_path=save_format.format(
                            dataset, index, 'jpg'),
                        show_img=False,
                        draw_distribution=self.args.draw_distribution)

                # # draw as one video
                # tv.draw_video(
                #     agent,
                #     save_path=save_format.format(index, 'avi'),
                #     draw_distribution=self.args.draw_distribution,
                # )

            self.log('Prediction result images are saved at {}'.format(
                os.path.join(save_base_path, 'VisualTrajs')))


def append_results_to_list(results: list[tf.Tensor], target: list):
    if not len(target):
        [target.append([]) for _ in range(len(results))]
    [target[index].append(results[index]) for index in range(len(results))]
    return target


def stack_results(results: list[tf.Tensor]):
    for index, tensor in enumerate(results):
        results[index] = tf.concat(tensor, axis=0)
    return results
