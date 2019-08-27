import SimpleITK as sitk
import pandas as pd


def get_location_features(mask, ROI):
    LabelShapeStatMask = sitk.LabelShapeStatisticsImageFilter()
    LabelShapeStatMask.Execute(mask)
    mask_centroid = LabelShapeStatMask.GetCentroid(1)

    LabelShapeStatROI = sitk.LabelShapeStatisticsImageFilter()
    LabelShapeStatROI.Execute(ROI)
    ROI_centroid = LabelShapeStatROI.GetCentroid(1)

    x_diff = mask_centroid[0] - ROI_centroid[0]
    y_diff = mask_centroid[1] - ROI_centroid[1]
    z_diff = mask_centroid[2] - ROI_centroid[2]

    panda_labels = ['x_position', 'y_position', 'z_position']

    location_features = [x_diff, y_diff, z_diff]


    pandas_dict = dict(zip(panda_labels, location_features))
    location_features = pd.Series(pandas_dict)

    return location_features
