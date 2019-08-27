import numpy as np
import SimpleITK as sitk


def GetImageFromArray(array):
    """
    GetImageFromArray converts a numpy array to an ITK image, while ensuring
    the orientation is kept the same.

    Args:
        array (numpy array): 2D or 3D array of image

    Returns:
        ITK image

    """

    if len(array.shape) == 3:
        array = np.transpose(array, [2, 1, 0])
    elif len(array.shape) == 2:
        array = np.transpose(array, [1, 0])
    return sitk.GetImageFromArray(array)


def GetArrayFromImage(image):
    """
    GetArrayFromImage converts an ITK image to a numpy array, while ensuring
    the orientation is kept the same.

    Args:
        image (ITK image): 2D or 3D ITK image

    Returns:
        numpy array

    """

    array = sitk.GetArrayFromImage(image)
    if len(array.shape) == 3:
        array = np.transpose(array, [2, 1, 0])
    elif len(array.shape) == 2:
        array = np.transpose(array, [1, 0])
    return array
