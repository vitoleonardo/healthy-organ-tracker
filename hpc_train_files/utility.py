import numpy as np

# HELPER FUNCTIONS
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros((shape[0] * shape[1], shape[2]), dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = color
    return img.reshape(shape)


def build_masks(labels,input_shape, colors=True):
    """
    It takes a list of RLEs, decodes them, and returns a mask
    
    :param labels: a list of RLE strings
    :param input_shape: the shape of the image that will be fed into the model
    :param colors: If True, the mask will be a 3-channel mask, otherwise it will be a 1-channel mask,
    defaults to True (optional)
    :return: The mask is being returned.
    """
    height, width = input_shape
    if colors:
        mask = np.zeros((height, width, 3))
        for label in labels:
            mask += rle_decode(label, shape=(height,width , 3), color=np.ra4ndom.rand(3))
    else:
        mask = np.zeros((height, width, 1))
        for label in labels:
            mask += rle_decode(label, shape=(height, width, 1))
    mask = mask.clip(0, 1)
    return mask

def fix_empty_slices(_row):
    """
    If the slice_id is in the list of slices to remove, then set the predicted value to an empty string.
    
    :param _row: the row of the dataframe
    :return: the row with the predicted column being empty if the slice_id is in the remove_seg_slices
    list.
    """
    if int(_row["slice_id"].rsplit("_", 1)[-1]) in remove_seg_slices[_row["class"]]:
        _row["predicted"] = ""
    return _row

def is_isolated(_row):
    """
    > If the current row's predicted value is not empty, and the previous and next rows' predicted
    values are empty, then return True
    
    :param _row: the row of the dataframe that we're currently looking at
    :return: A boolean value.
    """
    return (_row["predicted"]!="" and _row["prev_predicted"]=="" and _row["next_predicted"]=="")

def fix_nc_slices(_row):
    """
    If the row has a segmented isolated mask, then set the predicted value to an empty string
    
    :param _row: the row of the dataframe that is being processed
    :return: A dataframe with the predicted and actual values for each row.
    """
    if _row["seg_isolated"]:
        _row["predicted"] = ""
    return _row

remove_seg_slices = {
    "large_bowel": [1, 138, 139, 140, 141, 142, 143, 144],
    "small_bowel": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 138, 139, 140, 141, 142, 143, 144],
    "stomach": [1, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144],
}