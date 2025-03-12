import argparse
import json
import numpy as np
import time
import os
import shutil
import psutil
import sys
import socket
import traceback

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.lang_utils as LangUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings

from types import SimpleNamespace

from argparse import ArgumentParser

def main_eval(args):

    #_, config, algo_name, env_meta, shape_meta 
    model, config, algo_name, env_meta, shape_meta, action_normalization_stats = torch.load(args.checkpoint, map_location=torch.device('cpu')).values()
    
    action_normalization_stats["actions"]["scale"] = np.array(action_normalization_stats["actions"]["scale"])
    action_normalization_stats["actions"]["offset"] = np.array(action_normalization_stats["actions"]["offset"])

    obs_normalization_stats = None

    ext_cfg = json.loads(config)

    config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)
    
    """ Overriding the number of rollouts and parallel envs """
    assert args.num_rollouts > 0
    assert args.num_parallels > 0
    
    if args.num_parallels > 1:
        config.experiment.rollout.batched = True
        config.experiment.rollout.num_batch_envs = args.num_parallels
    
    config.experiment.rollout.n = args.num_rollouts
    
    log_dir, ckpt_dir, video_dir, vis_dir = TrainUtils.get_exp_dir(config)
    
    ObsUtils.initialize_obs_utils_with_config(config)
    
    # example checkpoint : /home/anonymous/expdata/robocasa/im/bc_xfmr/02-18-single_stage_mg_human/seed_1_ds_mg-and-human/20240219195516/models/model_epoch_100.pth
    epoch = args.checkpoint.split("/")[-1].split("_")[-1].split(".")[0]
    try:
        epoch = int(epoch)
    except: 
        print("Epoch is not an integer"); exit()
    
    # setup for a new training run
    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    
    # extract the metadata and shape metadata across all datasets
    env_meta_list = []
    shape_meta_list = []
    for dataset_cfg in config.train.data:
        dataset_path = os.path.expanduser(dataset_cfg["path"])
        ds_format = config.train.data_format
        if not os.path.exists(dataset_path):
            raise Exception("Dataset at provided path {} not found!".format(dataset_path))

        # load basic metadata from training file
        print("\n============= Loaded Environment Metadata =============")
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path, ds_format=ds_format)
        if "MG_" in env_meta["env_name"]:
            env_meta["env_name"] = env_meta["env_name"].replace("MG_", "")

        # populate language instruction for env in env_meta
        env_meta["env_lang"] = dataset_cfg.get("lang", None)

        # update env meta if applicable
        from robomimic.utils.script_utils import deep_update
        deep_update(env_meta, dataset_cfg.get("env_meta_update_dict", {}))
        deep_update(env_meta, config.experiment.env_meta_update_dict)
        env_meta_list.append(env_meta)

        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=dataset_path,
            action_keys=config.train.action_keys,
            all_obs_keys=config.all_obs_keys,
            ds_format=ds_format,
            verbose=True
        )
        shape_meta_list.append(shape_meta)

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    eval_env_meta_list = []
    eval_shape_meta_list = []
    eval_env_name_list = []
    eval_env_horizon_list = []
    for (dataset_i, dataset_cfg) in enumerate(config.train.data):
        do_eval = dataset_cfg.get("do_eval", True)
        
        if "TurnOffSinkFaucet" not in dataset_cfg["path"]:
            do_eval = False
        
        if do_eval is not True:
            continue
        eval_env_meta_list.append(env_meta_list[dataset_i])
        eval_shape_meta_list.append(shape_meta_list[dataset_i])
        eval_env_name_list.append(env_meta_list[dataset_i]["env_name"])
        horizon = dataset_cfg.get("horizon", config.experiment.rollout.horizon)
        eval_env_horizon_list.append(horizon)
    
    # create environments
    def env_iterator():
        for (env_meta, shape_meta, env_name) in zip(eval_env_meta_list, eval_shape_meta_list, eval_env_name_list):
            def create_env_helper(env_i=0):
                env_kwargs = dict(
                    env_meta=env_meta,
                    env_name=env_name,
                    render=False,
                    render_offscreen=config.experiment.render_video,
                    use_image_obs=shape_meta["use_images"],
                    seed=config.train.seed * 1000 + env_i,
                )
                env = EnvUtils.create_env_from_metadata(**env_kwargs)
                # handle environment wrappers
                env = EnvUtils.wrap_env_from_config(env, config=config)  # apply environment warpper, if applicable

                return env

            if config.experiment.rollout.batched:
                from tianshou.env import SubprocVectorEnv
                env_fns = [lambda env_i=i: create_env_helper(env_i) for i in range(config.experiment.rollout.num_batch_envs)]
                env = SubprocVectorEnv(env_fns)
                # env_name = env.get_env_attr(key="name", id=0)[0]
            else:
                env = create_env_helper()
                # env_name = env.name
            print(env)
            yield env

    # do rollouts at fixed rate or if it's time to save a new ckpt
    video_paths = None
    
    lang_encoder = LangUtils.LangEncoder(
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    rollout_model_policy = FileUtils.policy_from_checkpoint(ckpt_path=args.checkpoint)[0].policy
    rollout_model = RolloutPolicy(
                rollout_model_policy,
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
                lang_encoder=lang_encoder,
            )
    
    best_return = {k: -np.inf for k in eval_env_name_list} 
    best_success_rate = {k: -1. for k in eval_env_name_list}
    
    epoch_ckpt_name = "model_epoch_{}".format(epoch)
    
    num_episodes = config.experiment.rollout.n
    
    all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
        policy=rollout_model,
        envs=env_iterator(),
        horizon=eval_env_horizon_list,
        use_goals=config.use_goals,
        num_episodes=num_episodes,
        render=False,
        video_dir=video_dir if config.experiment.render_video else None,
        epoch=epoch,
        video_skip=config.experiment.get("video_skip", 5),
        terminate_on_success=config.experiment.rollout.terminate_on_success,
        del_envs_after_rollouts=True,
        data_logger=data_logger,
    )

    # checkpoint and video saving logic
    updated_stats = TrainUtils.should_save_from_rollout_logs(
        all_rollout_logs=all_rollout_logs,
        best_return=best_return,
        best_success_rate=best_success_rate,
        epoch_ckpt_name=epoch_ckpt_name,
        save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
        save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
    )
    best_return = updated_stats["best_return"]
    best_success_rate = updated_stats["best_success_rate"]
    epoch_ckpt_name = updated_stats["epoch_ckpt_name"]

    # Specify the file name
    file_name = args.checkpoint.replace(".pth", ".json")

    # Writing the dictionary to a JSON file
    with open(file_name, 'w') as json_file:
        json.dump(all_rollout_logs, json_file, indent=4)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num-rollouts', default=25, type=int)
    parser.add_argument('--num-parallels', default=1, type=int)
    args = parser.parse_args()
    main_eval(args)