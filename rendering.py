import open3d as o3d
from typing import NamedTuple
import numpy as np


class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

from threestudio.systems.GaussianDreamer import GaussianDreamer
from threestudio.data.uncond import RandomCameraDataModule

point_cloud = o3d.io.read_point_cloud("/home/xueyingjia/threestudio/outputs/gs-sds-generation/a_staircase_in_a_small_white_building,_featuring_a_white_pedestal,_a_white_box_on_a_table,_and_recessed_ceiling_lights.@20240425-230712/save/point_cloud.ply")
colors = np.asarray(point_cloud.colors)
points = np.asarray(point_cloud.points)
normals = np.asarray(point_cloud.normals)

pcd = BasicPointCloud(points, colors, normals)


dreamer = GaussianDreamer(GaussianDreamer.Config())
dreamer.configure()
dreamer.gaussian.create_from_pcd(pcd)
###### or #####
# dreamer.gaussian.load_ply("....saved gaussian .ply file")

loader = RandomCameraDataModule().val_dataloader()
for batch in loader:
    images = dreamer.forward(batch)['comp_rgb']
    print(images)
    # visualize / save images here
