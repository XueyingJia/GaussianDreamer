import open3d as o3d
from typing import NamedTuple
import numpy as np
from dataclasses import asdict


class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

from threestudio.systems.GaussianDreamer import GaussianDreamer
from threestudio.data.uncond import RandomCameraDataModule

system_config = {
    "load_type": 0,  # 0: load from shape, 1: load from smpl
    "load_path": "./load/shapes/stand.obj",  # only used when load_type == 1
    "radius": 4.0,
    "sh_degree": 0,
    "prompt_processor_type": "stable-diffusion-prompt-processor",
    "prompt_processor": {
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-2-1-base",
        "prompt": "a fox",  # Placeholder for the actual prompt
        "negative_prompt": ("ugly, bad anatomy, blurry, pixelated, obscure, unnatural colors, "
                            "poor lighting, dull, and unclear, cropped, lowres, low quality, "
                            "artifacts, duplicate, morbid, mutilated, poorly drawn face, "
                            "deformed, dehydrated, bad proportions, unfocused")
    },
    "guidance_type": "stable-diffusion-guidance",
    "guidance": {
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-2-1-base",
        "guidance_scale": 100.0,
        "weighting_strategy": "sds",
        "min_step_percent": 0.02,
        "max_step_percent": 0.98,
        "grad_clip": [0, 1.5, 2.0, 1000]
    },
    "loggers": {
        "wandb": {
            "enable": False,
            "project": "threestudio",
            "name": None
        }
    },
    "loss": {
        "lambda_sds": 1.0,
        "lambda_sparsity": 1.0,
        "lambda_opaque": 0.0
    },
    "optimizer": {
        "name": "Adam",
        "args": {
            "lr": 0.001,
            "betas": [0.9, 0.99],
            "eps": 1.e-15
        }
    }
}


point_cloud = o3d.io.read_point_cloud("/home/xueyingjia/threestudio/outputs/gs-sds-generation/a_staircase_in_a_small_white_building,_featuring_a_white_pedestal,_a_white_box_on_a_table,_and_recessed_ceiling_lights.@20240425-230712/save/point_cloud.ply")
colors = np.asarray(point_cloud.colors)
points = np.asarray(point_cloud.points)
normals = np.asarray(point_cloud.normals)

pcd = BasicPointCloud(points, colors, normals)

dreamer = GaussianDreamer(system_config)
dreamer.configure()
dreamer.gaussian.create_from_pcd(pcd)
###### or #####
# dreamer.gaussian.load_ply("....saved gaussian .ply file")

loader = RandomCameraDataModule().val_dataloader()
for batch in loader:
    images = dreamer.forward(batch)['comp_rgb']
    print(images)
    # visualize / save images here
