import tensorflow as tf
import numpy as np
import cv2

from utility import rle_encode, rle_decode, build_masks

# Data Generator
# Inspired by https://www.kaggle.com/code/samuelcortinhas/uwmgi-segmentation-unet-keras-train
class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, df, semi3d_data, batch_size=16,  height=128, width=128, subset="train", shuffle=False, use_crop_data=False):
        """_summary_

        Args:
            df (DataFrame): pandas dataframe
            batch_size (integer): default is set to 16. Increase if you have a lot of memory
            height (integer): height which image ins resized to. Should be a multiple of 32
            width (integer): width which image ins resized to. Should be a multiple of 32
            subset (str): set subset to "train" or "test". Default is train
            shuffle (bool): defaults to False.
            semi3d_data (bool): Specify if generator should load 2.5D data. Default is True
        """
        super().__init__()
        self.df           = df
        self.semi3d_data  = semi3d_data
        self.shuffle      = shuffle
        self.subset       = subset
        self.batch_size   = batch_size
        self.height       = height
        self.width        = width
        self.use_crop_data= use_crop_data
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
                
        for i,path in enumerate(self.df['path'].iloc[indexes]):
            h=self.df['height'].iloc[indexes[i]]
            w=self.df['width'].iloc[indexes[i]]
            
            col = self.df.loc[self.df['path'] == path]

            img = self.__load_grayscale(col['path00'].values[0])

            if self.semi3d_data:
                # 2.5D Data (Stack 3 slices together)
                img2 = self.__load_grayscale(col['path01'].values[0])
                img3 = self.__load_grayscale(col['path02'].values[0])
                img = np.stack([img,img2,img3], axis=2)

            X[i,] = img
            
            if self.subset == 'train':
                for k,j in enumerate(["large_bowel","small_bowel","stomach"]):
                    
                    # Get RLE encoded string of class, decode and create mask
                    rles = self.df[j].iloc[indexes[i]]
                    
                    masks = rle_decode(rles, shape=(h, w, 1))
                    if self.use_crop_data:
                        masks = self.__crop_image(masks, col['path00'].values[0])
                        
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

        if self.use_crop_data:
            img = self.__crop_image(img, img_path)
            
        dsize = (self.height, self.width)
        img = cv2.resize(img, dsize)
        img = img.astype(np.float32) / 255.
        
        if not self.semi3d_data:
            img = np.expand_dims(img, axis=-1)

        return img

    def __crop_image(self, img, img_path):
        
        col = self.df.loc[self.df['path'] == img_path]
        
        rs = col['rs'].values[0]
        re = col['re'].values[0]
        cs = col['cs'].values[0]
        ce = col['ce'].values[0]
        
        return img[rs:re, cs:ce]