"""
Common utility functions and classes.
"""

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import jaccard
from skimage.feature import peak_local_max
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.segmentation import relabel_sequential
from skimage.morphology import watershed
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops_table
import pandas as pd
import tensorflow as tf
import SimpleITK as sitk

############################################################
#  Analyze regions and return labels
############################################################

def label_mask(mask, threshold=0.5, min_pixel=15, do_watershed=False, exclude_border=False):
    'Analyze regions and return labels'
    if mask.ndim == 3:
        mask = np.squeeze(mask, axis=2)

    # apply threshold to mask
    # bw = closing(mask > threshold, square(2))
    bw = (mask > threshold).astype(int)

    # label image regions
    label_image = label(bw, connectivity=2) # Falk p.13, 8-“connectivity”.

    # Watershed: Separates objects in image by generate the markers
    # as local maxima of the distance to the background
    if do_watershed:
        distance = ndimage.distance_transform_edt(bw)
        # Minimum number of pixels separating peaks in a region of `2 * min_distance + 1`
        # (i.e. peaks are separated by at least `min_distance`)
        min_distance = int(np.ceil(np.sqrt(min_pixel / np.pi)))
        local_maxi = peak_local_max(distance, indices=False, exclude_border=False,
                                    min_distance=min_distance, labels=label_image)
        markers = label(local_maxi)
        label_image = watershed(-distance, markers, mask=bw)

    # remove artifacts connected to image border
    if exclude_border:
        label_image = clear_border(label_image)
    
    # remove areas < min pixel   
    unique, counts = np.unique(label_image, return_counts=True)
    label_image[np.isin(label_image, unique[counts<min_pixel])] = 0
    
    # re-label image
    label_image, _ , _ = relabel_sequential(label_image, offset=1)

    return (label_image)


############################################################
#  IOU
############################################################

def iou(a,b,threshold=0.5):
    'Compute IoU'
    a = a > threshold
    b = b > threshold
    overlap = a*b # Logical AND
    union = a+b # Logical OR
    return np.count_nonzero(overlap)/np.count_nonzero(union)
    # return overlap.sum()/float(union.sum())

############################################################
#  Compare masks using ROI-wise IoU
############################################################

def get_candidates(labels_a, labels_b):
    'Get candidates to compare masks for ROI-wise IoU'
    
    label_stack = np.dstack((labels_a, labels_b))
    cadidates = np.unique(label_stack.reshape(-1, label_stack.shape[2]), axis=0)
    # Remove Zero Entries
    cadidates = cadidates[np.prod(cadidates, axis=1) > 0]
    return(cadidates)
    

############################################################
#  Compare masks using ROI-wise Jaccard Similarity
############################################################

def iou_mapping(labels_a, labels_b, min_roi_size=30): 
    'Compare masks using ROI-wise Jaccard Similarity'

    candidates = get_candidates(labels_a, labels_b)
    
    if candidates.size > 0:
        # create a similarity matrix
        dim_a = np.max(candidates[:,0])+1
        dim_b = np.max(candidates[:,1])+1
        similarity_matrix = np.zeros((dim_a, dim_b))

        for x,y in candidates:
            roi_a = (labels_a == x).astype(np.uint8).flatten()
            roi_b = (labels_b == y).astype(np.uint8).flatten()
            similarity_matrix[x,y] = 1-jaccard(roi_a, roi_b)

        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        
        return(similarity_matrix[row_ind,col_ind], 
               row_ind, col_ind,
               np.max(labels_a), 
               np.max(labels_b)
               )
    else:
        return([], 
               np.nan, np.nan,
               np.max(labels_a), 
               np.max(labels_b)
               )

############################################################
# STAPLE: Simultaneous Truth and Performance Level Estimation
# with simple ITK
############################################################
def staple(segmentations, foregroundValue = 1, threshold = 0.5):
    'STAPLE: Simultaneous Truth and Performance Level Estimation with simple ITK'

    segmentations = [sitk.GetImageFromArray(x) for x in segmentations]
    STAPLE_probabilities = sitk.STAPLE(segmentations) 
    STAPLE = STAPLE_probabilities > threshold
    return sitk.GetArrayViewFromImage(STAPLE)


############################################################
#  Measure regions and return region props
############################################################

def measure_rois(mask, image,file_id, threshold=0.5, min_pixel=30, 
                 properties = ['mean_intensity', 'label', 'area']):
    'Measure regions and return region properties'
    
    if mask.ndim == 3:
        mask = np.squeeze(mask, axis=2)

    # apply threshold to mask
    bw = (mask > threshold).astype(int)

    # label image regions
    label_image = label(bw, connectivity=2) # Falk p.13, 8-“connectivity”.

    # remove areas < min pixel   
    unique, counts = np.unique(label_image, return_counts=True)
    label_image[np.isin(label_image, unique[counts<min_pixel])] = 0
    
    # re-label image
    label_image, _ , _ = relabel_sequential(label_image, offset=1)  
    
    # measure region props
    if label_image.max()>0:
        props_inner = regionprops_table(label_image, image, properties=properties)
        df = pd.DataFrame(props_inner)
    else:
        df = pd.DataFrame(np.nan,index=[0],columns=properties)
         
    df['Nummer'] = file_id 
    return(df)