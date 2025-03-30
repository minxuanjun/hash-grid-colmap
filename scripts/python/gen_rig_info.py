# https://cau-git.rz.uni-kiel.de/inf-ag-koeser/colmap_underwater/-/blob/2.1/src/exe/rig_bundle_adjuster.cc

# Read the configuration of the camera rigs from a JSON file. The input images
# of a camera rig must be named consistently to assign them to the appropriate
# camera rig and the respective snapshots.

# An example configuration of a single camera rig:
# [
# {
# "ref_camera_id": 1,
# "cameras":
# [
# {
# "camera_id": 1,
# "image_prefix": "left1_image"
# },
# {
# "camera_id": 2,
# "image_prefix": "left2_image"
# },
# {
# "camera_id": 3,
# "image_prefix": "right1_image"
# },
# {
# "camera_id": 4,
# "image_prefix": "right2_image"
# }
# ]
# }
# ]

# This file specifies the configuration for a single camera rig and that you
# could potentially define multiple camera rigs. The rig is composed of 4
# cameras: all images of the first camera must have "left1_image" as a name
# prefix, e.g., "left1_image_frame000.png" or "left1_image/frame000.png".
# Images with the same suffix ("_frame000.png" and "/frame000.png") are
# assigned to the same snapshot, i.e., they are assumed to be captured at the
# same time. Only snapshots with the reference image registered will be added
# to the bundle adjustment problem. The remaining images will be added with
# independent poses to the bundle adjustment problem. The above configuration
# could have the following input image file structure:

# /path/to/images/...
# left1_image/...
# frame000.png
# frame001.png
# frame002.png
# ...
# left2_image/...
# frame000.png
# frame001.png
# frame002.png
# ...
# right1_image/...
# frame000.png
# frame001.png
# frame002.png
# ...
# right2_image/...
# frame000.png
# frame001.png
# frame002.png
# ...

# TODO: Provide an option to manually / explicitly set the relative extrinsics
# of the camera rig. At the moment, the relative extrinsics are automatically
# inferred from the reconstruction.

def gen_reg_info(ref_cam, cameras, camera_name2_camera_id_dict, rig_info_path):
    rig_info = {}
    rig_info["ref_camera_id"]
    rig_info["cameras"] = []
    # ref_cam
    camera_dict = {}
    camera_dict["camera_id"]
    camera_dict["image_prefix"]
    camera_dict["rel_tvec"]
    camera_dict["ref_qvec"]
    rig_info["cameras"].append(camera_dict)

    # add other camera
    pass