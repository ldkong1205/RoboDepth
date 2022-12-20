from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

DEFAULT_IMAGES_KITTI = 57874
DEFAULT_DEPTH_BATCH_SIZE = 6
DEFAULT_SEG_BATCH_SIZE = 6
DEFAULT_BATCHES_PER_EPOCH = DEFAULT_IMAGES_KITTI // DEFAULT_DEPTH_BATCH_SIZE


class ArgumentsBase(object):
    DESCRIPTION = 'SGDepth Arguments'

    def __init__(self):
        self.ap = ArgumentParser(
            description=self.DESCRIPTION,
            formatter_class=ArgumentDefaultsHelpFormatter
        )

    def _harness_init_system(self):
        self.ap.add_argument(
            '--sys-cpu', default=False, action='store_true',
            help='Disable Hardware acceleration'
        )

        self.ap.add_argument(
            '--sys-num-workers', type=int, default=3,
            help='Number of worker processes to spawn per DataLoader'
        )

        self.ap.add_argument(
            '--sys-best-effort-determinism', default=False, action='store_true',
            help='Try and make some parts of the training/validation deterministic'
        )

    def _harness_init_model(self):

        self.ap.add_argument(
            '--model-num-layers', type=int, default=18, choices=(18, 34, 50, 101, 152),
            help='Number of ResNet Layers in the depth and segmentation encoder'
        )

        self.ap.add_argument(
            '--model-num-layers-pose', type=int, default=18, choices=(18, 34, 50, 101, 152),
            help='Number of ResNet Layers in the pose encoder'
        )

        self.ap.add_argument(
            '--model-split-pos', type=int, default=1, choices=(0, 1, 2, 3, 4),
            help='Position in the decoder to split from common to separate depth/segmentation decoders'
        )

        self.ap.add_argument(
            '--model-depth-min', type=float, default=0.1,
            help='Depth Estimates are scaled according to this min/max',
        )

        self.ap.add_argument(
            '--model-depth-max', type=float, default=100.0,
            help='Depth Estimates are scaled according to this min/max',
        )

        self.ap.add_argument(
            '--model-depth-resolutions', type=int, default=4, choices=(1, 2, 3, 4),
            help='Number of depth resolutions to generate in the network'
        )

        self.ap.add_argument(
            '--experiment-class', type=str, default='sgdepth_eccv_test',
            help='A nickname for the experiment folder'
        )

        self.ap.add_argument(
            '--model-name', type=str, default='sgdepth_base',
            help='A nickname for this model'
        )

        self.ap.add_argument(
            '--model-load', type=str, default=None,
            help='Load a model state from a state directory containing *.pth files'
        )

        self.ap.add_argument(
            '--model-disable-lr-loading', default=False, action='store_true',
            help='Do not load the learning rate scheduler if you load a checkpoint'
        )

    def _harness_init_depth(self):
        self.ap.add_argument(
            '--depth-validation-resize-height', type=int, default=192,
            help='Depth images are resized to this height'
        )

        self.ap.add_argument(
            '--depth-validation-resize-width', type=int, default=640,
            help='Depth images are resized to this width'
        )

        self.ap.add_argument(
            '--depth-validation-crop-height', type=int, default=192,
            help='Segmentation validation images are cropped to this height'
        )

        self.ap.add_argument(
            '--depth-validation-crop-width', type=int, default=640,
            help='Segmentation validation images are cropped to this width'
        )

        self.ap.add_argument(
            '--depth-validation-loaders', type=str, default='kitti_kitti_validation',
            help='Comma separated list of depth dataset loaders from loaders/depth.py to use for validation'
        )

        self.ap.add_argument(
            '--depth-validation-batch-size', type=int, default=1,
            help='Batch size for depth validation'
        )

        self.ap.add_argument(
            '--depth-validation-fixed-scaling', type=float, default=0,
            help='Use this fixed scaling ratio (from another run) for validation outputs'
        )

        self.ap.add_argument(
            '--depth-ratio-on-validation', default=False, action='store_true',
            help='Determines the ratios only on the first quarter of the data'
        )

    def _harness_init_pose(self):
        self.ap.add_argument(
            '--pose-validation-resize-height', type=int, default=192,
            help='Depth images are resized to this height'
        )

        self.ap.add_argument(
            '--pose-validation-resize-width', type=int, default=640,
            help='Depth images are resized to this width'
        )

        self.ap.add_argument(
            '--pose-validation-loaders', type=str, default='',
            help='Comma separated list of depth dataset loaders from loaders/depth.py to use for validation'
        )

        self.ap.add_argument(
            '--pose-validation-batch-size', type=int, default=1,
            help='Batch size for depth validation'
        )

        self.ap.add_argument(
            '--pose-validation-fixed-scaling', type=float, default=0,
            help='Use this fixed scaling ratio (from another run) for validation outputs'
        )

    def _harness_init_segmentation(self):
        self.ap.add_argument(
            '--segmentation-validation-resize-height', type=int, default=512,
            help='Segmentation images are resized to this height prior to cropping'
        )

        self.ap.add_argument(
            '--segmentation-validation-resize-width', type=int, default=1024,
            help='Segmentation images are resized to this width prior to cropping'
        )

        self.ap.add_argument(
            '--segmentation-validation-loaders', type=str, default='cityscapes_validation',
            help='Comma separated list of segmentation dataset loaders from loaders/segmentation.py to '
                 'use for validation'
        )

        self.ap.add_argument(
            '--segmentation-validation-batch-size', type=int, default=1,
            help='Batch size for segmentation validation'
        )

    def _training_init_train(self):
        self.ap.add_argument(
            '--train-batches-per-epoch', type=int, default=DEFAULT_BATCHES_PER_EPOCH,
            help='Number of batches we consider an epoch'
        )

        self.ap.add_argument(
            '--train-num-epochs', type=int, default=20,
            help='Number of epochs to train for'
        )

        self.ap.add_argument(
            '--train-checkpoint-frequency', type=int, default=5,
            help='Number of epochs between model checkpoint dumps'
        )

        self.ap.add_argument(
            '--train-tb-frequency', type=int, default=500,
            help='Number of steps between each info dump to tensorboard'
        )

        self.ap.add_argument(
            '--train-print-frequency', type=int, default=2500,
            help='Number of steps between each info dump to stdout'
        )

        self.ap.add_argument(
            '--train-learning-rate', type=float, default=1e-4,
            help='Initial learning rate to train with',
        )

        self.ap.add_argument(
            '--train-scheduler-step-size', type=int, default=15,
            help='Number of epochs between learning rate reductions',
        )

        self.ap.add_argument(
            '--train-weight-decay', type=float, default=0.0,
            help='Weight decay to train with',
        )

        self.ap.add_argument(
            '--train-weights-init', type=str, default='pretrained', choices=('pretrained', 'scratch'),
            help='Initialize the encoder networks with Imagenet pretrained ResNets oder start from scratch'
        )

        self.ap.add_argument(
            '--train-depth-grad-scale', type=float, default=0.9,
            help='How much are depth gradients scaled on their way into the common network parts'
        )

        self.ap.add_argument(
            '--train-segmentation-grad-scale', type=float, default=0.1,
            help='How much are segmentation gradients scaled on their way into the common network parts'
        )

    def _training_init_depth(self):
        self.ap.add_argument(
            '--depth-training-loaders', type=str, default='kitti_kitti_train',
            help='Comma separated list of depth dataset loaders from loaders/depth.py to use for training'
        )

        self.ap.add_argument(
            '--depth-training-batch-size', type=int, default=DEFAULT_DEPTH_BATCH_SIZE,
            help='Batch size for depth training'
        )

        self.ap.add_argument(
            '--depth-resize-height', type=int, default=192,
            help='Depth images are resized to this height'
        )

        self.ap.add_argument(
            '--depth-resize-width', type=int, default=640,
            help='Depth images are resized to this width'
        )

        self.ap.add_argument(
            '--depth-crop-height', type=int, default=192,
            help='Segmentation images are cropped to this height'
        )

        self.ap.add_argument(
            '--depth-crop-width', type=int, default=640,
            help='Segmentation images are cropped to this width'
        )

        self.ap.add_argument(
            '--depth-disparity-smoothness', type=float, default=1e-3,
            help='Scaling factor for the disparity smoothness component of the depth loss'
        )

        self.ap.add_argument(
            '--depth-min-sampling-res', type=int, default=10000,
            help='Smallest max(x,y) image dimension at which to multi resolution sampling should '
            'continue to downsample. Set this to >= max(--depth-resize-height,--depth-resize-width)'
            'to disable multi resolution sampling all together'
        )

        self.ap.add_argument(
            '--depth-avg-reprojection', action='store_true',
            help='Use average reprojection loss instead of minimum reprojection loss'
        )

        self.ap.add_argument(
            '--depth-disable-automasking', action='store_true',
            help='Disable automasking with the unwarped input frames'
        )

    def _training_init_segmentation(self):
        self.ap.add_argument(
            '--segmentation-training-loaders', type=str, default='cityscapes_train',
            help='Comma separated list of segmentation dataset loaders from loaders/segmentation.py to use for training'
        )

        self.ap.add_argument(
            '--segmentation-training-batch-size', type=int, default=DEFAULT_SEG_BATCH_SIZE,
            help='Batch size for segmentation training'
        )

        self.ap.add_argument(
            '--segmentation-resize-height', type=int, default=512,
            help='Segmentation images are resized to this height prior to cropping'
        )

        self.ap.add_argument(
            '--segmentation-resize-width', type=int, default=1024,
            help='Segmentation images are resized to this width prior to cropping'
        )

        self.ap.add_argument(
            '--segmentation-crop-height', type=int, default=192,
            help='Segmentation images are cropped to this height'
        )

        self.ap.add_argument(
            '--segmentation-crop-width', type=int, default=640,
            help='Segmentation images are cropped to this width'
        )

    def _training_init_masking(self):
        self.ap.add_argument(
            '--masking-enable', action='store_true',
            help='if set uses segmentation mask to mask moving objects'
        )

        self.ap.add_argument(
            '--masking-from-epoch', type=int, default=15,
            help='defines at which epoch the mask is applied for the first time'
        )

        self.ap.add_argument(
            '--moving-mask-percent', type=float, default=0.1,
            help='Percentage of moving objects with worst iou should be masked, if --linear'
                 'is set than this will be the percentage at the end of training'
        )

        self.ap.add_argument(
            '--masking-linear-increase', action='store_true',
            help='if set first mask out all objects and then increases the percentage of allowed images linear'
        )

    def _eval_init_logging(self):
        self.ap.add_argument(
            '--eval-num-images', type=int, default=20,
            help='Number of generated images to store to disk'
        )

    def _inference(self):
        self.ap.add_argument(
            '--image-path', type=str,
            help='Path to image directory'
        )

        self.ap.add_argument(
            '--output-path', type=str,
            help='Path to output directory'
        )

        self.ap.add_argument(
            '--model-path', type=str,
            help='Path to model.pth'
        )

        self.ap.add_argument(
            '--inference-resize-height', type=int, default=192,
            help='Segmentation images are resized to this height'
        )

        self.ap.add_argument(
            '--inference-resize-width', type=int, default=640,
            help='Segmentation images are resized to this width'
        )

        self.ap.add_argument(
            '--output-format', type=str, default='.jpg',
            help='format the results will be saved in. Everything that OpenCV supports fe.: .jpg or .png'
        )

    def _parse(self):
        return self.ap.parse_args()


class TrainingArguments(ArgumentsBase):
    DESCRIPTION = 'SGDepth training arguments'

    def __init__(self):
        super().__init__()

        self._harness_init_system()
        self._harness_init_model()
        self._harness_init_depth()
        self._harness_init_segmentation()
        self._training_init_train()
        self._training_init_depth()
        self._training_init_segmentation()
        self._training_init_masking()

    def parse(self):
        opt = self._parse()

        # This option is only useful for evaluation
        # but required in the harness
        opt.eval_avg_with_flipped = False

        return opt


class DepthEvaluationArguments(ArgumentsBase):
    DESCRIPTION = 'SGDepth Depth Evaluation'

    def __init__(self):
        super().__init__()

        self._harness_init_system()
        self._harness_init_model()
        self._harness_init_depth()
        self._eval_init_logging()

    def parse(self):
        opt = self._parse()

        # These options are required by the StateManager
        # but are effectively ignored when evaluating so
        # they can be initialized to arbitrary values
        opt.train_learning_rate = 0
        opt.train_scheduler_step_size = 1000
        opt.train_weight_decay = 0
        opt.train_weights_init = 'scratch'
        opt.train_depth_grad_scale = 0
        opt.train_segmentation_grad_scale = 0

        return opt


class SegmentationEvaluationArguments(ArgumentsBase):
    DESCRIPTION = 'SGDepth Segmentation Evaluation'

    def __init__(self):
        super().__init__()

        self._harness_init_system()
        self._harness_init_model()
        self._harness_init_segmentation()
        self._eval_init_logging()

    def parse(self):
        opt = self._parse()

        # These options are required by the StateManager
        # but are effectively ignored when evaluating so
        # they can be initialized to arbitrary values
        opt.train_learning_rate = 0
        opt.train_scheduler_step_size = 1000
        opt.train_weight_decay = 0
        opt.train_weights_init = 'scratch'
        opt.train_depth_grad_scale = 0
        opt.train_segmentation_grad_scale = 0

        return opt


class PoseEvaluationArguments(ArgumentsBase):
    DESCRIPTION = 'SGDepth Depth Evaluation'

    def __init__(self):
        super().__init__()

        self._harness_init_system()
        self._harness_init_model()
        self._harness_init_pose()
        self._eval_init_logging()

    def parse(self):
        opt = self._parse()

        # These options are required by the StateManager
        # but are effectively ignored when evaluating so
        # they can be initialized to arbitrary values
        opt.train_learning_rate = 0
        opt.train_scheduler_step_size = 1000
        opt.train_weight_decay = 0
        opt.train_weights_init = 'scratch'
        opt.train_depth_grad_scale = 0
        opt.train_segmentation_grad_scale = 0

        return opt


class InferenceEvaluationArguments(ArgumentsBase):
    DESCRIPTION = 'SGDepth Segmentation Inference'

    def __init__(self):
        super().__init__()

        self._harness_init_system()
        self._harness_init_model()
        # self._harness_init_segmentation()
        # self._eval_init_logging()
        self._inference()

    def parse(self):
        opt = self._parse()

        # These options are required by the StateManager
        # but are effectively ignored when evaluating so
        # they can be initialized to arbitrary values
        opt.train_learning_rate = 0
        opt.train_scheduler_step_size = 1000
        opt.train_weight_decay = 0
        opt.train_weights_init = 'scratch'
        opt.train_depth_grad_scale = 0
        opt.train_segmentation_grad_scale = 0

        return opt