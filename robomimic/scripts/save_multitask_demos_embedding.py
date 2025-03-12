import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import h5py
import torch

import os
import json
import h5py
import argparse
import imageio
import numpy as np
import time
import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.envs.env_base import EnvBase, EnvType
import natsort

from robomimic.algo.algo import RolloutPolicy

from scipy.spatial.distance import cdist
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import gc
import sys

torch.backends.cudnn.enabled = False


def get_raw_obs_at_idx(obs, i):
    d = dict()
    for key in obs:
        d[key] = obs[key][i].copy()
    return d

def process_batch(batch, policy):
    batch = TensorUtils.to_batch(batch)
    batch = TensorUtils.to_tensor(batch)
    batch = TensorUtils.to_device(batch, policy.nets["policy"].device)
    batch = TensorUtils.to_float(batch)
    return batch

def downscale_img(obs, policy):
    try:
        downscale_img = policy.algo_config.dyn.downscale_img
    except:
        downscale_img = False
    if downscale_img:
        policy._scaled_img_size = policy.algo_config.dyn.scaled_img_size
        obs = policy._downscale_img(obs)
    for k in obs:
        if "image" in k:
            obs[k] = ObsUtils.process_obs(obs[k], obs_modality='rgb')
    return obs

def replace_home_path(path):
    return path.replace("~", "/home/anonymous")


def get_embedding(f, demo, policy, save_future, obs_key="robot0_agentview_left_image"):
    embeddings = []
    demos = f["data/{}".format(demo)]
    demos_obs = f["data/{}".format(demo)]["obs"]
    traj_len = len(demos_obs[obs_key])
    step = 10 if not save_future else 9
    with torch.no_grad():
        for i in range(0, traj_len, step):
            if not save_future:
                index = slice(i, min(i+10, traj_len))
            else:
                if i+20 >= traj_len:
                    break
                index = slice(i, i+20)
            obs = get_raw_obs_at_idx(demos_obs, index)
            obs = process_batch(obs, policy.policy)
            actions = demos["actions"][index]
            
            obs = downscale_img(obs, policy.policy)
            if not save_future:
                embedding = policy.policy.nets['policy'].get_curr_latent_features({'obs':obs}).cpu().numpy()
            else:
                batch = {'obs':obs, 'actions':actions}
                embedding = policy.policy.nets['policy'].imagine(batch).cpu().numpy()
            embeddings.append(embedding.reshape(-1, embedding.shape[-1]))
            # breakpoint()
            
        gc.collect()
        torch.cuda.empty_cache()
    return embeddings

def main(args):   
    json_path, ckpt_path, save_path, save_future = args.json_path, args.ckpt_path, args.save_path, args.save_future
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump({"json_path": json_path, "ckpt_path": ckpt_path}, f, indent=4)
        
         
    embed_all = []
    with open(json_path, "r") as f:
        config = json.load(f)
        data_configs = config["train"]["data"]
            
    device = torch.device("cuda:2")

    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
    policy.policy.nets['policy'].eval()
    

    for data_config in data_configs:
        
        if args.env is not None and args.env not in data_config["path"]:
            continue

        print("Processing data config: ", data_config["path"])
        path = replace_home_path(data_config["path"])
        f = h5py.File(path, "r")
        demos = list(f["data"].keys())
        demos = natsort.natsorted(demos)
        
        test_embed = get_embedding(f, demos[0], policy, save_future) #test
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(get_embedding, f, demo, policy, save_future) for demo in demos]
            for i,future in enumerate(futures):
                print("at demo: ", i)
                embedding = future.result()
                embed_all.extend(embedding)
                
        f.close()
        del f
    embed_all = np.concatenate(embed_all, axis=0).reshape(-1, 1, 1, embed_all[0].shape[-1])
    print("Embedding shape: ", embed_all.shape)
    
    npy_path = os.path.join(save_path, args.save_name)

    np.save(npy_path, embed_all, allow_pickle=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="sirius-fleet/robomimic/exps/dataset_human.json")
    parser.add_argument("--ckpt_path", type=str, default="world_model.pth")
    parser.add_argument("--save_path", type=str, default="ood_embeddings")
    parser.add_argument("--save_name", type=str, default="embeddings.npy")
    parser.add_argument("--save_future", action="store_true")
    parser.add_argument("--env", default=None) # by default, generate all the embeddings

    args = parser.parse_args()
    
    main(args)
