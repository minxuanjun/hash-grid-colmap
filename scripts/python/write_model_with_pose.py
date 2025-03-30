import numpy as np
import os
from read_write_model import  read_model, write_model, CameraModel, Camera, BaseImage, Point3D, CAMERA_MODEL_IDS, CAMERA_MODEL_NAMES, CAMERA_MODELS
from database import COLMAPDatabase

cameras, images, point_3ds = read_model(path='/home/minxuan/Data/ImageDataset_SceauxCastle/sparse', ext='.bin')

  # Open the database.
out_db_path = "/home/minxuan/Data/ImageDataset_SceauxCastle/database_with_pose_prior.db"

db = COLMAPDatabase.connect(out_db_path)

# For convenience, try creating all the tables upfront.

db.create_tables()

# Create dummy cameras.

model1, width1, height1, params1 = (
    0,
    1024,
    768,
    np.array((1024.0, 512.0, 384.0)),
)
model2, width2, height2, params2 = (
    2,
    1024,
    768,
    np.array((1024.0, 512.0, 384.0, 0.1)),
)

for camera_id, camera in cameras.items():
    print(f'{CAMERA_MODEL_NAMES[camera.model].model_id} {camera.model}')
    camera_id1 = db.add_camera(CAMERA_MODEL_NAMES[camera.model].model_id, camera.width, camera.height, camera.params, True, camera_id=camera.id)

new_images = {}
for image_id, image in images.items():
    # image.xys = np.array([])
    image = BaseImage(id=image.id, qvec= image.qvec, tvec=image.tvec,  camera_id= image.camera_id, name=image.name, xys=np.array([]), point3D_ids=np.array([]))
    new_images[image_id] = image

    db.add_image(name= image.name, camera_id=image.camera_id, prior_q= image.qvec, prior_t= image.tvec, image_id= image.id, )
db.commit()

db.close()


out_path = '/home/minxuan/Data/ImageDataset_SceauxCastle/sparse1'

os.makedirs(out_path, exist_ok=True)

write_model(cameras, new_images, {}, out_path, ext='.txt')

# ./colmap point_triangulator  --database_path  /home/minxuan/Data/ImageDataset_SceauxCastle/database_with_pose_prior.db --image_path /home/minxuan/Data/ImageDataset_SceauxCastle/images --input_path /home/minxuan/Data/ImageDataset_SceauxCastle/sparse1  --output_path /home/minxuan/Data/ImageDataset_SceauxCastle/triangulated/sparse1


