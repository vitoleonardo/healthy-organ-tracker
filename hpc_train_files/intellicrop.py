def __crop_image(img, rs, re, cs , ce):
    """
    It takes an image and returns a cropped version of the image
    
    :param img: the image to be cropped
    :param rs: row start
    :param re: row end
    :param cs: column start
    :param ce: column end
    :return: The image is being cropped to the specified rows and columns.
    """
    return img[rs:re, cs:ce]

def __get_crop_area_from_image(slice_, rs, re, cs , ce, tol=0.5):        
    """
    It takes a 2D array and returns the smallest rectangle that contains all non-zero elements
    
    :param slice_: the image to be cropped
    :param rs: row start
    :param re: row end
    :param cs: column start
    :param ce: column end
    :param tol: tolerance for the masking
    :return: The crop area of the image.
    """
    slice_ = median_filter(slice_, size=15, mode='constant', cval=0)

    slice_mask = slice_ > tol
    m,n = slice_mask.shape
    mask0,mask1 = slice_mask.any(0),slice_mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()

    rs = min(rs, row_start)
    re = max(re, row_end)
    cs = min(cs, col_start)
    ce = max(ce, col_end)

    return rs, re, cs, ce

def __get_crop_area(df):
    """
    For each case and day, find the crop area by finding the crop area from each image and then taking
    the minimum of the crop areas
    
    :param df: dataframe with the paths to the images
    :return: The crop area for each case and day.
    """
    d = pd.DataFrame()
    
    for case in df['case'].unique():
        for day in df[df['case']==case]['day'].unique():
            print("Looking for area for day {} and case {}".format(day, case))
            rs = 10000
            re = 10000
            cs = 0
            ce = 0
            

            for path in df.loc[(df['case'] == case) & (df['day'] == day)]['path']:
                image = cv2.imread(path, -1)
                image = np.array(image, dtype=int)
                image = image.astype(np.float32) / np.percentile(image, 99)
                rs, re, cs, ce = __get_crop_area_from_image(image, rs, re, cs, ce)
                
            data = [{'case' : case, 'day':day, 'rs':rs,'re':re,'cs':cs,'ce':ce}]
            d = d.append(data)
            print("Area: {} {} {} {}".format(rs, re, cs, ce))
        
    return d

