import os
import yaml

import numpy as np


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary"""

    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            try:
                key, value = line.split(':', 1)
            except ValueError:
                key, value = line.split(' ', 1)
            # the only non-float values in these files are dates, which we don't care about anyway

            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def load_calib(calib_filepath: str):
    filedata = read_calib_file(calib_filepath)

    data = {}

    # Create 3x4 projection matrices
    P_rect_00 = np.reshape(filedata['P0'], (3, 4))
    P_rect_10 = np.reshape(filedata['P1'], (3, 4))
    P_rect_20 = np.reshape(filedata['P2'], (3, 4))
    P_rect_30 = np.reshape(filedata['P3'], (3, 4))

    data['P_rect_00'] = P_rect_00
    data['P_rect_10'] = P_rect_10
    data['P_rect_20'] = P_rect_20
    data['P_rect_30'] = P_rect_30

    # compute the rectified extrinsics from cam0 to camN
    T1 = np.eye(4)
    T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
    T2 = np.eye(4)
    T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
    T3 = np.eye(4)
    T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 3]

    # compute the camera intrinsics
    data['K_cam0'] = P_rect_00[0:3, 0:3]
    data['K_cam1'] = P_rect_10[0:3, 0:3]
    data['K_cam2'] = P_rect_20[0:3, 0:3]
    data['K_cam3'] = P_rect_30[0:3, 0:3]

    print("P_rect_00:\n{}".format(P_rect_00))
    print("K_cam0:\n{}".format(data['K_cam0']))

    print("P_rect_10:\n{}".format(P_rect_10))
    print("K_cam1:\n{}".format(data['K_cam1']))

    print("P_rect_20:\n{}".format(P_rect_20))
    print("K_cam2:\n{}".format(data['K_cam2']))

    print("P_rect_30:\n{}".format(P_rect_30))
    print("K_cam3:\n{}".format(data['K_cam3']))


# P_rect_00:
# [[718.856    0.     607.1928   0.    ]
#  [  0.     718.856  185.2157   0.    ]
# [  0.       0.       1.       0.    ]]
# K_cam0:
# [[718.856    0.     607.1928]
#  [  0.     718.856  185.2157]
# [  0.       0.       1.    ]]
# P_rect_10:
# [[ 718.856     0.      607.1928 -386.1448]
#  [   0.      718.856   185.2157    0.    ]
# [   0.        0.        1.        0.    ]]
# K_cam1:
# [[718.856    0.     607.1928]
#  [  0.     718.856  185.2157]
# [  0.       0.       1.    ]]
# P_rect_20:
# [[ 7.188560e+02  0.000000e+00  6.071928e+02  4.538225e+01]
#  [ 0.000000e+00  7.188560e+02  1.852157e+02 -1.130887e-01]
# [ 0.000000e+00  0.000000e+00  1.000000e+00  3.779761e-03]]
# K_cam2:
# [[718.856    0.     607.1928]
#  [  0.     718.856  185.2157]
# [  0.       0.       1.    ]]
# P_rect_30:
# [[ 7.188560e+02  0.000000e+00  6.071928e+02 -3.372877e+02]
#  [ 0.000000e+00  7.188560e+02  1.852157e+02  2.369057e+00]
# [ 0.000000e+00  0.000000e+00  1.000000e+00  4.915215e-03]]
# K_cam3:
# [[718.856    0.     607.1928]
#  [  0.     718.856  185.2157]
# [  0.       0.       1.    ]]

if __name__ == '__main__':
    calib_filepath = "/mnt/c/Users/minxuan/Dataset/00/calib.txt"
    load_calib(calib_filepath)
