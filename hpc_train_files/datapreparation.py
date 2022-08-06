import pandas as pd
import numpy as np
from glob import glob
import re
import os


def extract_metadata(df, train_dir, remove_faulty_cases=True):
    """
    > It takes the dataframe from the train.csv file and merges it with the metadata from the slice meta
    info
    
    :param df: the dataframe containing the segmentation data
    :param train_dir: The directory where the training images are stored
    :param remove_faulty_cases: This is a boolean parameter that determines whether or not to remove the
    faulty cases, defaults to True
    :return: A dataframe with the following columns:
        id, large_bowel, small_bowel, stomach, case, day, slice, path, width, height, pixel_x, pixel_y,
    count
    """
    # Get data from train.csv
    df_train = pd.DataFrame({'id':df['id'][::3]})
    df_train['large_bowel'] = df['segmentation'][::3].values
    df_train['small_bowel'] = df['segmentation'][1::3].values
    df_train['stomach'] = df['segmentation'][2::3].values
    df_train.reset_index(inplace=True,drop=True)

    df_train["case"] = df_train["id"].apply(lambda x: re.findall("\d+",x)[0])
    df_train["day"] = df_train["id"].apply(lambda x: re.findall("\d+",x)[1])
    df_train["slice"] = df_train["id"].apply(lambda x: re.findall("\d+",x)[2])

    # Get data from slice meta info
    path_df = pd.DataFrame(glob(os.path.join(train_dir, "**", "*.png"), recursive=True),columns=['path'])
    path_df['case'] = path_df['path'].apply(lambda x: re.findall("\d+",x)[1])
    path_df['day'] = path_df['path'].apply(lambda x: re.findall("\d+",x)[2])
    path_df['slice'] = path_df['path'].apply(lambda x: re.findall("\d+",x)[3])
    path_df['width'] = path_df['path'].apply(lambda x: re.findall("\d+",x)[4]).astype('int64')
    path_df['height'] = path_df['path'].apply(lambda x: re.findall("\d+",x)[5]).astype('int64')

    path_df['pixel_x'] = path_df['path'].apply(lambda x: re.findall("\d+\.\d+",x)[0])
    path_df['pixel_y'] = path_df['path'].apply(lambda x: re.findall("\d+\.\d+",x)[1])

    # Merge dataframes 
    df_train = df_train.merge(path_df, how='left', on=['case','day','slice'])
    df_train.fillna('',inplace=True);
    df_train['count'] = np.sum(df_train.iloc[:,1:4]!='',axis=1).values
    df_train.sort_values(by=['case','day','slice'],ignore_index=True, inplace=True)

    df_train = _get_paths_array(df_train)

    print("Frame merged. Shape: {}".format(df_train.shape))
    print("Remove faulty cases: {}".format(remove_faulty_cases))

    if remove_faulty_cases:
        df_train = _remove_faulties(df_train)
        print("Sucess. Shape: {}".format(df_train.shape))

    return df_train

def _remove_faulties(df_train):
    fault1 = 'case7_day0'
    fault2 = 'case81_day30'
    df_train = df_train[~df_train['id'].str.contains(fault1) & ~df_train['id'].str.contains(fault2)].reset_index(drop=True)
    return df_train

def _get_paths_array(df_train, channels=3, stride=2):
    """
    > For each channel, we shift the path column by a certain stride, and then fill the missing values
    with the previous value
    
    :param df: the dataframe
    :param channels: number of images to be used in the model, defaults to 3 (optional)
    :param stride: how many days to skip between each image, defaults to 2 (optional)
    :return: A list of lists of image paths. (3 paths are expected)
    """
    for i in range(channels):
        df_train[f'path{i:02}'] = df_train.groupby(['case','day'])['path'].shift(-i*stride).fillna(method="ffill")

    df_train['image_paths'] = df_train[[f'path{i:02d}' for i in range(channels)]].values.tolist()
    return df_train