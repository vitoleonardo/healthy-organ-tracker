# This class contains all the configuration parameters for the model
class CFG:

    def __init__(self,img_dims=(128,128,3), model='UNet', batch_size=16, epochs=50, kaggle=True, use_fold_csv=True, backbone='efficientnetb0', lr_patience=5, semi3d_data=True, internet=False):
        self.img_dims             = img_dims
        self.height               = img_dims[0]
        self.width                = img_dims[1]
        self.model                = model
        self.batch_size           = batch_size
        self.epochs               = epochs
        self.kaggle               = kaggle
        self.lr                   = 5e-4
        self.n_splits             = 5
        self.selected_fold        = 1
        self.folds                = range(1, self.n_splits + 1)
        self.classes              = 3
        self.path_prefix          =  '../input/' if kaggle else 'input/'
        self.base_path            = self.path_prefix + 'uw-madison-gi-tract-image-segmentation/'
        self.train_dir            = self.base_path + 'train'
        self.train_csv            = self.base_path + 'train.csv'
        self.seed                 = 42
        self.use_fold_csv         = use_fold_csv
        self.backbone             = backbone
        self.lr_patience          = lr_patience
        self.semi3d_data          = semi3d_data
        self.internet             = internet

    # Method which returns path according to key from encoder_weights_path
    def get_encoder_weights_path(self, key):
        return self.encoder_weights_path[key]

    # A dictionary of the encoder weights path.
    encoder_weights_path = {
        'efficientnetb0': 'input/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
        'efficientnetb5': 'input/efficientnet-b5_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
        'efficientnetb7': 'input/efficientnet-b7_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
        'inceptionresnetv2': 'input/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
        'inceptionv3': 'input/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
        'resnet50': 'input/resnet50_imagenet_1000_no_top.h5',
        'resnext50': 'input/resnext50_imagenet_1000_no_top.h5',
        'resnext101': 'input/resnext101_imagenet_1000_no_top.h5',
        'seresnext50': 'input/seresnext50_imagenet_1000_no_top.h5',
        'seresnext101': 'input/seresnext101_imagenet_1000_no_top.h5',
        }