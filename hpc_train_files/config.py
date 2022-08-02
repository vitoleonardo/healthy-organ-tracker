class CFG:
    def __init__(self,img_dims=(128,128,3), model='UNet', batch_size=16, epochs=50, kaggle=True, use_fold_csv=True, encoder_weights_path=encoder_weights_path):
        self.img_dims             = img_dims
        self.height               = img_dims[0]
        self.width                = img_dims[1]
        self.model                = model
        self.batch_size           = batch_size  # train batch_size
        self.epochs               = epochs
        self.lr                   = 5e-4 # learning rate
        self.n_splits             = 5
        self.selected_fold        = 1
        self.folds                = range(1, self.n_splits + 1)
        self.classes              = 3
        self.path_prefix          = '../' if kaggle else ''
        self.base_path            = self.path_prefix + 'input/uw-madison-gi-tract-image-segmentation/'
        self.train_dir            = self.base_path + 'train'
        self.train_csv            = self.base_path + 'train.csv'
        self.seed                 = 42
        self.use_fold_csv         = use_fold_csv
        self.encoder_weights_path = encoder_weights_path

encoder_weights_path = {
    'efficientnetb0': cfg.base_path + 'efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
    'efficientnetb5': cfg.base_path + 'efficientnet-b5_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
    'efficientnetb7': cfg.base_path + 'efficientnet-b7_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
    'inceptionresnetv2': cfg.base_path + 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
    'inceptionv3': cfg.base_path + 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
    'resnet50': cfg.base_path + 'resnet50_imagenet_1000_no_top.h5',
    'resnext50': cfg.base_path + 'resnext50_imagenet_1000_no_top.h5',
    'resnext101': cfg.base_path + 'resnext101_imagenet_1000_no_top.h5',
    'seresnext50': cfg.base_path + 'seresnext50_imagenet_1000_no_top.h5',
    'seresnext101': cfg.base_path + 'seresnext101_imagenet_1000_no_top.h5',
    }