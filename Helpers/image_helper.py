import numpy as np
import sitk_helper as sitkh


def get_masked_slices_image(image_array, mask_array):
    mask_array = mask_array.astype(np.bool)

    mask_slices = np.any(mask_array, axis=(0, 1))
    image_array = image_array[:, :, mask_slices]
    mask_array = mask_array[:, :, mask_slices]

    return image_array, mask_array


def get_masked_voxels(image_array, mask_array):
    mask_array = mask_array.astype(np.bool)

    mask_array = mask_array.flatten()
    image_array = image_array.flatten()

    masked_voxels = image_array[mask_array]

    return masked_voxels


def get_masked_slices_mask(mask_image):
    mask_array = sitkh.GetArrayFromImage(mask_image)
    # Filter out slices where there is no mask (need actual index here)
    mask_slices = np.flatnonzero(np.any(mask_array, axis=(0, 1)))

    mask_sliced = mask_image[:, :, mask_slices[0]:mask_slices[-1]]

    return mask_sliced
