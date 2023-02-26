import os
import numpy as np
from pathlib import Path

trajnet_path = '../AgentFormerSDD/datasets/trajnet_sdd_ye/'
for trajnet_scene_path in Path(trajnet_path).glob('t*/*.txt'):
    print("trajnet_scene_path:", trajnet_scene_path)
    scene_name, video_num = str(trajnet_scene_path).split('.txt')[0].split('/')[-1].split('_')
    # print(f"scene_name: {scene_name}_{video_num}")
    og = np.loadtxt(str(trajnet_scene_path), delimiter=' ')
    vv_path = f'../Vertical/data/sdd/{scene_name}/video{video_num}/true_pos_.csv'
    vv = np.loadtxt(vv_path, delimiter=',')
    # print("vv.shape:", vv.shape)
    new_vv = og.T
    # print("new_vv.shape:", new_vv.shape)
    # np.savetxt(vv_path, new_vv, delimiter=',')
