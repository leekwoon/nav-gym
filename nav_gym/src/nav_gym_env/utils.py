import numpy as np


# rescale angle to be in [-pi, pi]
def angle_correction(angle):
    angle = np.arctan2(
        np.sin(angle), np.cos(angle)
    ) 
    return angle


def translation_matrix_from_xyz(
    x, y, z
):
    return np.array([
        [1., 0., 0., x],
        [0., 1., 0., y],
        [0., 0., 1., z],
        [0., 0., 0., 1.]
    ])


def quaternion_matrix_from_yaw(
    yaw
):
    return np.array([
        [np.cos(yaw), -np.sin(yaw), 0., 0.],
        [np.sin(yaw), np.cos(yaw), 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ])


def transform_xy(
    translation_matrix,
    quaternion_matrix,
    xy # (2)
):
    result = transform_xys(
        translation_matrix,
        quaternion_matrix,
        xy[None]
    )
    return result[0]


def transform_xys(
    translation_matrix,
    quaternion_matrix,
    xys # (batch, 2) 
):
    mat = np.dot(translation_matrix, quaternion_matrix)

    padded_xys = np.zeros((xys.shape[0], xys.shape[1] + 2))
    padded_xys[:, :2] = xys
    padded_xys[:, 2] = 0.
    padded_xys[:, 3] = 1.

    result = np.dot(mat, padded_xys.T).T
    result = result[:, :2]
    return result


