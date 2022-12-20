import os

import torch
import torch.optim as optim

import dataloader.file_io.get_path as get_path
from models.sgdepth import SGDepth


class ModelContext(object):
    def __init__(self, model, mode):
        self.model = model
        self.mode_wanted = mode

    def _set_mode(self, mode):
        if mode == 'train':
            self.model.train()
        elif mode == 'eval':
            self.model.eval()

    def __enter__(self):
        self.mode_was = 'train' if self.model.training else 'eval'
        self._set_mode(self.mode_wanted)

        return self.model

    def __exit__(self, *_):
        self._set_mode(self.mode_was)


class ModelManager(object):
    def __init__(self, model):
        self.model = model

    def get_eval(self):
        return ModelContext(self.model, 'eval')

    def get_train(self):
        return ModelContext(self.model, 'train')


class StateManager(object):
    def __init__(self, experiment_class, model_name, device, split_pos, num_layers,
                 grad_scale_depth, grad_scale_seg,
                 weights_init, resolutions_depth, num_layers_pose,
                 learning_rate, weight_decay, scheduler_step_size):

        self.device = device

        path_getter = get_path.GetPath()
        self.log_base = path_getter.get_checkpoint_path()
        self.log_path = os.path.join(self.log_base, experiment_class, model_name)

        self._init_training()
        self._init_model(
            split_pos, num_layers, grad_scale_depth, grad_scale_seg, weights_init, resolutions_depth,
            num_layers_pose
        )
        self._init_optimizer(learning_rate, weight_decay, scheduler_step_size)

    def _init_training(self):
        self.epoch = 0
        self.step = 0

    def _init_model(self, split_pos, num_layers, grad_scale_depth, grad_scale_seg, weights_init, resolutions_depth,
                    num_layers_pose
                    ):

        model = SGDepth(split_pos, num_layers, grad_scale_depth, grad_scale_seg, weights_init,
                        resolutions_depth, num_layers_pose
                        )

        # noinspection PyUnresolvedReferences
        model = model.to(self.device)
        self.model_manager = ModelManager(model)

    def _init_optimizer(self, learning_rate, weight_decay, scheduler_step_size):
        with self.model_manager.get_train() as model:
            self.optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)

        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, scheduler_step_size, learning_rate
        )

    def _state_dir_paths(self, state_dir):
        return {
            'model': os.path.join(self.log_base, state_dir, "model.pth"),
            'optimizer': os.path.join(self.log_base, state_dir, "optim.pth"),
            'scheduler': os.path.join(self.log_base, state_dir, "scheduler.pth"),
            'train': os.path.join(self.log_base, state_dir, "train.pth"),
        }

    def store_state(self, state_dir):
        print(f"Storing model state to {state_dir}:")
        os.makedirs(state_dir, exist_ok=True)

        paths = self._state_dir_paths(state_dir)

        with self.model_manager.get_train() as model:
            torch.save(model.state_dict(), paths['model'])

        torch.save(self.optimizer.state_dict(), paths['optimizer'])
        torch.save(self.lr_scheduler.state_dict(), paths['scheduler'])

        state_train = {
            'step': self.step,
            'epoch': self.epoch,
        }

        torch.save(state_train, paths['train'])

    def store_checkpoint(self):
        state_dir = os.path.join(self.log_path, "checkpoints", f"epoch_{self.epoch}")
        self.store_state(state_dir)

    # Idea: log the model every batch to see how the training of the statistic parameters of the BN layers for the
    #       shared encoder effect the validation
    def store_batch_checkpoint(self, batch_idx):
        state_dir = os.path.join(self.log_path, "checkpoints", f"batch_{batch_idx}")
        self.store_state(state_dir)

    def _load_model_state(self, path):
        with self.model_manager.get_train() as model:
            state = model.state_dict()
            to_load = torch.load(path, map_location=self.device)

            for (k, v) in to_load.items():
                if k not in state:
                    print(f"    - WARNING: Model file contains unknown key {k} ({list(v.shape)})")

            for (k, v) in state.items():
                if k not in to_load:
                    print(f"    - WARNING: Model file does not contain key {k} ({list(v.shape)})")

                else:
                    state[k] = to_load[k]

            model.load_state_dict(state)

    def _load_optimizer_state(self, path):
        state = torch.load(path, map_location=self.device)
        self.optimizer.load_state_dict(state)

    def _load_scheduler_state(self, path):
        state = torch.load(path)
        self.lr_scheduler.load_state_dict(state)

    def _load_training_state(self, path):
        state = torch.load(path)
        self.step = state['step']
        self.epoch = state['epoch']

    def load(self, state_dir, disable_lr_loading=False):
        """Load model(s) from a state directory on disk
        """

        print(f"Loading checkpoint from {os.path.join(self.log_base, state_dir)}:")

        paths = self._state_dir_paths(state_dir)

        print(f"  - Loading model state from {paths['model']}:")
        try:
            self._load_model_state(paths['model'])
        except FileNotFoundError:
            print("   - Could not find model state file")
        if not disable_lr_loading:
            print(f"  - Loading optimizer state from {paths['optimizer']}:")
            try:
                self._load_optimizer_state(paths['optimizer'])
            except FileNotFoundError:
                print("   - Could not find optimizer state file")
            except ValueError:
                print("   - Optimizer state file is incompatible with current setup")

            print(f"  - Loading scheduler state from {paths['scheduler']}:")
            try:
                self._load_scheduler_state(paths['scheduler'])
            except FileNotFoundError:
                print("   - Could not find scheduler state file")

            print(f"  - Loading training state from {paths['train']}:")
            try:
                self._load_training_state(paths['train'])
            except FileNotFoundError:
                print("   - Could not find training state file")
