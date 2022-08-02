import tensorflow as tf

# Data Generator
# Inspired by https://www.kaggle.com/code/samuelcortinhas/uwmgi-segmentation-unet-keras-train
class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, df, batch_size=16,  height=HEIGHT, width=WIDTH, subset="train", shuffle=False, two_five_dim=True):
        """_summary_

        Args:
            df (DataFrame): pandas dataframe
            batch_size (integer): default is set to 16. Increase if you have a lot of memory
            height (integer): height which image ins resized to. Should be a multiple of 32
            width (integer): width which image ins resized to. Should be a multiple of 32
            subset (str): set subset to "train" or "test". Default is train
            shuffle (bool): defaults to False.
            two_five_dim (bool): Specify if generator should load 2.5D data. Default is True
        """
        super().__init__()
        self.df           = df
        self.shuffle      = shuffle
        self.subset       = subset
        self.batch_size   = batch_size
        self.height       = height
        self.width        = width
        self.two_five_dim = two_five_dim
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        """
        The `on_epoch_end` function is called at the end of each epoch. It shuffles the dataframe indexes if
        `shuffle` is set to `True` which is the default.
        """
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """
        It takes in a dataframe, and returns a generator that can be used to iterate over the dataframe in
        batches
        
        :param index: the index of the batch
        :return: X is the image, y is the mask
        """
        
        X = np.empty((self.batch_size,self.width,self.height,3))
        y = np.empty((self.batch_size,self.width,self.height,3))
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        for i,image_paths in enumerate(self.df['image_paths'].iloc[indexes]):
            h=self.df['height'].iloc[indexes[i]]
            w=self.df['width'].iloc[indexes[i]]

            img = self.__load_grayscale(image_paths[0])

            if self.two_five_dim:
                # 2.5D Data (Stack 3 slices together)
                img2 = self.__load_grayscale(image_paths[1])
                img3 = self.__load_grayscale(image_paths[2])
                img = np.stack([img1,img2,img3], axis=2)

            X[i,] = img
            
            if self.subset == 'train':
                for k,j in enumerate(["large_bowel","small_bowel","stomach"]):
                    
                    # Get RLE encoded string of class, decode and create mask
                    rles = self.df[j].iloc[indexes[i]]
                    masks = rle_decode(rles, shape=(h, w, 1))
                    masks = cv2.resize(masks, (self.height, self.width))
                    
                    y[i,:,:,k] = masks
        if self.subset == 'train': return X, y
        else: return X
        
    def __load_grayscale(self, img_path):
        """
        It loads the image, resizes it to the desired size, and converts it to a float32 array
        
        :param img_path: the path to the image
        :return: The image is being returned.
        """
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        dsize = (self.height, self.width)
        img = cv2.resize(img, dsize)
        img = img.astype(np.float32) / 255.
        if not self.two_five_dim:
            img = np.expand_dims(img, axis=-1)

        return img