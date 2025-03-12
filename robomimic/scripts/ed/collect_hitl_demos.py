"""Teleoperate robot with keyboard or SpaceMouse. """

import argparse
import numpy as np
import pandas as pd
import os
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper
import time
import numpy as np
import json
from robosuite.scripts.collect_human_demonstrations import gather_demonstrations_as_hdf5
import robomimic
import cv2
import robomimic.utils.obs_utils as ObsUtils
import copy

from collect_playback_utils import reset_to
import h5py

import colorama
from colorama import Fore, Style

import threading
import subprocess

import robosuite
is_v1 = (robosuite.__version__.split(".")[0] == "1")

from collect_hitl_demos_helper import *

from error_detectors import *

import torch


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

from PIL import Image

from robomimic.utils.obs_utils import TU, unprocess_frame

import random

def load_detector_config(detector_type):
    if True:
        config_path = os.path.join(
            os.path.dirname(__file__), "detector_configs/{}.json".format(detector_type)
        )
        with open(config_path) as f:
            detector_config = json.load(f)

        if 'CONFIG_OVERRIDE' in os.environ:
            config_override = json.loads(os.environ['CONFIG_OVERRIDE'])
            detector_config.update(config_override)
    else:
        detector_config = None
        
    return detector_config


def detector_from_config(detector_type, detector_checkpoints, args):

    try:
        detector_config = load_detector_config(detector_type)
    except:
        pass

    if detector_type == "siriusfleet_failure":

        eval_idx = slice(7, 10)
        detector = SiriusFleet_Failure(checkpoint=detector_checkpoints[0], 
                                     threshold=0.2, 
                                     threhold_history=3, 
                                     threshold_count=2, 
                                     eval_method="idx", 
                                     eval_idx=eval_idx,
                                     use_prob=False, 
                                     pred_future=True, 
                                     compile=False,
                                     use_intv=False
                                     )
        
    elif detector_type == "siriusfleet_ood":
        
        detector = SiriusFleet_OOD(checkpoint=detector_checkpoints[0], 
                                 threshold=None, 
                                 demos_embedding_path=args.demos_embedding_path, 
                                 eval_method="weighted", 
                                 num_future=20, 
                                 pred_future=True, 
                                 dist_metric="kmeans", 
                                 train_kmeans=False, 
                                 percentile=args.percentile)
        
    elif detector_type == "siriusfleet_combined":
        
        eval_idx = slice(7, 10)
        failure_detector = SiriusFleet_Failure(checkpoint=detector_checkpoints[0], 
                                     threshold=0.2, 
                                     threhold_history=3, 
                                     threshold_count=2, 
                                     eval_method="idx", 
                                     eval_idx=eval_idx,
                                     use_prob=False, 
                                     pred_future=True, 
                                     compile=False,
                                     use_intv=False
                                     )
        
        ood_detector = SiriusFleet_OOD(checkpoint=detector_checkpoints[0], 
                                 threshold=None, 
                                 demos_embedding_path=args.demos_embedding_path, 
                                 eval_method="weighted", 
                                 num_future=20, 
                                 pred_future=True, 
                                 dist_metric="kmeans", 
                                 train_kmeans=False, 
                                 percentile=args.percentile)
        
        detector = SiriusFleetCombined(
            ood_detector=ood_detector,
            failure_detector=failure_detector,
            )

    return detector



def print_color(message, color):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m'
    }
    print(f"{colors.get(color, '')}{message}\033[0m")  # Default to no color if invalid color is provided

class RandomPolicy:
    def __init__(self, env):
        self.env = env
        self.low, self.high = env.action_spec

    def get_action(self, obs):
        return np.random.uniform(self.low, self.high) / 2


def is_empty_input_spacemouse(action):
    empty_input = np.array([0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 1.000])
    if np.array_equal(np.abs(action), empty_input):
        return True
    return False

def terminate_condition_met(time_success, timestep_count, term_cond):
    assert term_cond in ["fixed_length", "success_count", "stop"]
    if term_cond == "fixed_length":
        return timestep_count >= GOOD_EPISODE_LENGTH and time_success > 0
    elif term_cond == "success_count":
        return time_success == SUCCESS_HOLD
    elif term_cond == "stop":
        return timestep_count >= MAX_EPISODE_LENGTH

def post_process_spacemouse_action(action, grasp, last_grasp):
    """ Fixing Spacemouse Action """
    last_grasp = grasp

    env_robosuite = env.env.env.env.env

    if is_v1:
        env_action_dim = env_robosuite.action_dim
    else:
        env_action_dim = 7

    # Fill out the rest of the action space if necessary
    rem_action_dim = env_action_dim - action.size
    if rem_action_dim > 0:
        # Initialize remaining action space
        rem_action = np.zeros(rem_action_dim)
        # This is a multi-arm setting, choose which arm to control and fill the rest with zeros
        if args.arm == "right":
            action = np.concatenate([action, rem_action])
        elif args.arm == "left":
            action = np.concatenate([rem_action, action])
        else:
            # Only right and left arms supported
            print("Error: Unsupported arm specified -- "
                  "must be either 'right' or 'left'! Got: {}".format(args.arm))
    elif rem_action_dim < 0:
        # We're in an environment with no gripper action space, so trim the action space to be the action dim
        action = action[:env_action_dim]

    """ End Fixing Spacemouse Action """
    return action, last_grasp

def process_obs_robosuite(obs):
    obs = copy.deepcopy(obs)
    di = obs

    ret = {}
    for k in di:
        if "image" in k:
            ret[k] = di[k][::-1]
            ret[k] = ObsUtils.process_obs(ret[k], obs_modality='rgb')
    obs.update(ret)
    obs.pop('frame_is_assembled', None)
    obs.pop('tool_on_frame', None)
    return obs


class Renderer:
    def __init__(self, env, render_onscreen):
        
        self.env = env
        self.render_onscreen = render_onscreen

        if (is_v1 is False) and self.render_onscreen:
            self.env.viewer.set_camera(camera_id=2)

    def render(self, obs):
        if is_v1:
            
            ## if using data collection wrapper
            vis_env = self.env.env
            robosuite_env = self.env.env.env.env.env
            robosuite_env.visualize(vis_settings=vis_env._vis_settings)
        else:
            robosuite_env = self.env.env

        if self.render_onscreen:
            self.env.render()
        else:

            img0 = robosuite_env.sim.render(height=500, width=550, camera_name="robot0_agentview_right")
            img0 = img0[:,:,::-1]
            img0 = np.flip(img0, axis=0)

            img1 = robosuite_env.sim.render(height=500, width=550, camera_name="robot0_agentview_left")
            img1 = img1[:,:,::-1]
            img1 = np.flip(img1, axis=0)

            img2 = robosuite_env.sim.render(height=500, width=550, camera_name="robot0_eye_in_hand")
            img2 = img2[:,:,::-1]
            img2 = np.flip(img2, axis=0)

            # concat two images
            img = np.concatenate((img0, img1, img2), axis=1)
            
            cv2.imshow('offscreen render', img)
            cv2.waitKey(1)

        if is_v1:
            robosuite_env.visualize(vis_settings=dict(
                env=False,
                grippers=False,
                robots=False,
            ))



# Change later
GOOD_EPISODE_LENGTH = None
MAX_EPISODE_LENGTH = None
SUCCESS_HOLD = None

def collect_trajectory(env, device, args):
    
    ## if using data collection wrapper
    robosuite_env = env.env.env.env.env
    robomimic_env = env.env.env.env
    
    robosuite_env.translucent_robot = False
    robomimic_env.translucent_robot = False
    
    time0 = time.time()
    
    renderer = Renderer(env, args.render_onscreen)

    time1 = time.time()

    obs = env.reset()
    
    time2 = time.time()
    
    renderer.render(obs)
    
    time3 = time.time()
    
    detector.reset()
    
    print("Time to initialize the renderer: ", time1 - time0)
    print("Time to do env.reset(): ", time2 - time1)
    print("Time to render: ", time3 - time2)
    
    ep_meta = env.get_ep_meta()

    lang = ep_meta.get("lang", None)
    print_color("\nTask Goal: " + lang + " \n", "yellow")
    
    if not args.all_demos:
        policy.start_episode(robomimic_env._ep_lang_str)

    # Initialize variables that should the maintained between resets
    last_grasp = 0

    # Initialize device control
    device.start_control()

    time_success = 0
    timestep_count = 0
    num_human_samples = 0
    nonzero_ac_seen = False
    discard_traj = False
    success_at_time = -1

    obs_buffer = []

    """ Runtime monitoring variables """
    HUMAN_CONTROL = False # by default, no human control
    segment_first_intv = True # by default is the first intv place
    
    while True:
        
        if_sleep = args.sleep_time > 0

        if if_sleep and not time_success:
            if is_v1:
                time.sleep(args.sleep_time)
            else:
                time.sleep(0.02)

        if is_v1:
            # Set active robot
            active_robot = robosuite_env.robots[0] if args.config == "bimanual" else robosuite_env.robots[args.arm == "left"]
        else:
            active_robot = None

        # Get the newest action
        action, grasp, _ = input2action(
            device=device,
            robot=active_robot,
            active_arm=args.arm,
            env_configuration=args.config,
            mirror_actions=True,
        )

        if action is None:
            discard_traj = True
            break

        action, last_grasp = post_process_spacemouse_action(action, grasp, last_grasp)
        action_mode = None
        
        """ Process input for policy """
        for k in obs:
            if "image" in k and obs[k].shape[1] != 3:
                # need to permute
                obs[k] = np.transpose(obs[k], (0, 3, 1, 2))
        
        policy_action = policy(obs)


        """ Process input for runtime monitoring """
        obs_copy_for_rm = copy.deepcopy(obs)
        obs_single = {}
        for k in obs_copy_for_rm:
            obs_single[k] = obs_copy_for_rm[k][-1]
        
        for k in obs_single:
            if "image" not in k or "right" in k:
                continue
            if True:
                obs_single[k] = TU.to_uint8(unprocess_frame(frame=obs_single[k], channel_dim=3, scale=255.))

        """ Runtime monitoring results """
        if args.detector_type == "siriusfleet_combined":
            ood_result, failure_result, dist, prob = detector.human_intervene_frame({'obs':obs_single, 'actions':policy_action})
        
            if ood_result or failure_result:
                human_intervene_true = True
            else:
                human_intervene_true = False
            
            if human_intervene_true:
                print(Fore.RED + "Human Intervene " + f"OOD dist: {dist:.2f}" + f"   failure prob: {prob:.2f}" \
                      + Fore.LIGHTBLUE_EX + "                   Task Goal: " + lang + Style.RESET_ALL)    
            else:
                print(Fore.GREEN + "No Intervene " + f"OOD dist: {dist:.2f}" + f"   failure prob: {prob:.2f}" \
                      + Fore.LIGHTBLUE_EX + "                   Task Goal: " + lang + Style.RESET_ALL)  
            
            print()
            if ood_result:
                print(Fore.YELLOW + "Type: OOD Intervene" + Style.RESET_ALL)
            if failure_result:
                print(Fore.YELLOW + "Type: Failure Intervene" + Style.RESET_ALL)
            print()

        else:

            human_intervene_true, prob = detector.human_intervene_frame({'obs':obs_single, 'actions':policy_action})
        
            if human_intervene_true:
                print(Fore.RED + "Human Intervene " + f"   failure prob: {prob:.2f}" \
                    )
            else:
                print(Fore.GREEN + "No Intervene " + f"   failure prob: {prob:.2f}" \
                    )
            print(Style.RESET_ALL)

        if args.human_always_override: # human still makes the decision when to intervene, just for testing
            
            if is_empty_input_spacemouse(action):
                if args.all_demos:
                    if not nonzero_ac_seen: # if have not seen nonzero action, should not be zero action
                        continue # if all demos, no action
                    # else: okay to be zero action afterwards
                    num_human_samples += 1
                    action_mode = -1
                else:
                    action = policy_action
                    action_mode = 0
            else:
                nonzero_ac_seen = True
                num_human_samples += 1
                if args.all_demos:
                    action_mode = -1 # iter 0 is viewed as non-intervention
                else:
                    action_mode = 1

        else: # Actual runtime monitoring

            """ Human Intervention Decision Part """
            if human_intervene_true: # if it asks human to intervene
                print('\033[91m ############# HUMAN INTERVENE!!!! ############# \033[0m')
                
                user_intv_here = False
                # at first intv place, human decide whether to intervene
                if segment_first_intv:
                    segment_first_intv = False
                    user_choice = input("Intervene here? y - Yes, n - No").replace('\r', '')
                    while user_choice not in ["y", "n"]:
                        user_choice = input("Intervene here? y - Yes, n - No").replace('\r', '')
                    if user_choice == "y": # choose to intervene
                        user_intv_here = True
                    else:
                        user_intv_here = False

                # if human does not intervene at first intv place, robot will continue to rollout
                if not user_intv_here:
                    if is_empty_input_spacemouse(action):
                        action = policy_action
                        action_mode = 0
                        
                    else:
                        print('\033[92m HUMAN CONTROL \033[0m')
                        HUMAN_CONTROL = True

                        num_human_samples += 1
                        action_mode = 1
                
                # if human decides to intervene at first intv place: 
                else:    
                    print('\033[92m HUMAN CONTROL \033[0m')
                    HUMAN_CONTROL = True
                    
                    # wait for human to provide input
                    while is_empty_input_spacemouse(action):
                        # Get the newest action
                        action, grasp, _ = input2action(
                            device=device,
                            robot=active_robot,
                            active_arm=args.arm,
                            env_configuration=args.config
                        )
                        if action is None:
                            break
                        action, last_grasp = post_process_spacemouse_action(action, grasp, last_grasp)
                    
                    action_mode = 1
            
            else: # if did not asks human to intervene
                print("Normal rollout")

                # if human control is on, and the human decides to go back to robot control
                if HUMAN_CONTROL:
                    if is_empty_input_spacemouse(action):
                        HUMAN_CONTROL = False
                        print('\033[93m ROBOT CONTROL BACK \033[0m')
                        action = policy_action
                        action_mode = 0
                        
                    else:
                        num_human_samples += 1
                        if args.all_demos:
                            action_mode = -1 # iter 0 is viewed as non-intervention
                        else:
                            action_mode = 1

                # if human control is off: normal rollout case
                else:
                    segment_first_intv = True
                    action = policy_action
                    action_mode = 0
                
        assert action_mode is not None

        obs, _, _, _ = env.step(action, 
                                action_mode=action_mode,
                                rm_dict={
                                    "human_intervene": human_intervene_true,
                                    "failure_score": prob, 
                                    "failure_pred": failure_result,
                                    "ood_score": dist,
                                    "ood_pred": ood_result,
                                    "ood_threshold": detector.ood_detector.threshold
                                }
                                )
        timestep_count += 1

        renderer.render(obs)

        if env._check_success():
            time_success += 1
            if time_success == 1:
                print("Success length: ", timestep_count)
                success_at_time = timestep_count

        if terminate_condition_met(time_success=time_success,
                                   timestep_count=timestep_count,
                                   term_cond=args.term_condition):
            break

        if timestep_count > MAX_EPISODE_LENGTH:
            print("Agent Fails!")
            # discard_traj = True
            break

    ep_directory = env.ep_directory
    env.close()
    return ep_directory, timestep_count, num_human_samples, success_at_time, discard_traj


def obtain_model_and_env(args):

    try:
        model, config, algo_name, env_meta, shape_meta, action_normalization_stats = torch.load(args.checkpoint).values()
    except:
        model, config, algo_name, env_meta, shape_meta = torch.load(args.checkpoint).values()
        action_normalization_stats = OrderedDict([('actions', {'scale': [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], 
                                                               'offset': [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]})])

    action_normalization_stats["actions"]["scale"] = np.array(action_normalization_stats["actions"]["scale"])
    action_normalization_stats["actions"]["offset"] = np.array(action_normalization_stats["actions"]["offset"])

    obs_normalization_stats = None

    ext_cfg = json.loads(config)

    config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked():
        config.update(ext_cfg)
    
    """ Overriding the number of rollouts and parallel envs """
    assert args.num_rollouts > 0
    
    config.experiment.rollout.n = args.num_rollouts
    
    log_dir, ckpt_dir, video_dir, vis_dir = TrainUtils.get_exp_dir(config)
    
    ObsUtils.initialize_obs_utils_with_config(config)
    
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
    
    _, config_data, _, _, _, _ = torch.load("model_epoch_1600.pth", map_location=torch.device('cpu')).values()
    config_data = json.loads(config_data)
    config_ = config_factory(config_data["algo_name"])
    with config_.values_unlocked():
        config_.update(config_data)
    config_data = config_
    
    # extract the metadata and shape metadata across all datasets
    env_meta_list = []
    shape_meta_list = []
    for dataset_cfg in config_data.train.data:
        dataset_path = os.path.expanduser(dataset_cfg["path"])
        ds_format = config_data.train.data_format
        if not os.path.exists(dataset_path):
            raise Exception("Dataset at provided path {} not found!".format(dataset_path))

        # load basic metadata from training file
        print("\n============= Loaded Environment Metadata =============")
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path, ds_format=ds_format)
        if "MG_" in env_meta["env_name"]:
            env_meta["env_name"] = env_meta["env_name"].replace("MG_", "")

        env_meta.pop("obj_groups", None)
        env_meta.pop("exclude_obj_groups", None)

        # populate language instruction for env in env_meta
        env_meta["env_lang"] = dataset_cfg.get("lang", None)

        # update env meta if applicable
        from robomimic.utils.script_utils import deep_update
        
        env_meta_list.append(env_meta)

        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=dataset_path,
            action_keys=config_data.train.action_keys,
            all_obs_keys=config_data.all_obs_keys,
            ds_format=ds_format,
            verbose=True
        )
        shape_meta_list.append(shape_meta)

    if config_data.experiment.env is not None:
        env_meta["env_name"] = config_data.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    eval_env_meta_list = []
    eval_shape_meta_list = []
    eval_env_name_list = []
    eval_env_horizon_list = []
    for (dataset_i, dataset_cfg) in enumerate(config_data.train.data):
        
        dataset_path = os.path.expanduser(dataset_cfg["path"])
        if args.environment not in dataset_path:
            continue
        
        eval_env_meta_list.append(env_meta_list[dataset_i])
        eval_shape_meta_list.append(shape_meta_list[dataset_i])
        eval_env_name_list.append(env_meta_list[dataset_i]["env_name"])
        horizon = dataset_cfg.get("horizon", config_data.experiment.rollout.horizon)
        eval_env_horizon_list.append(horizon)

    env_lst = [] # should only have one env

    print(eval_env_meta_list,)
    for (env_meta, shape_meta, env_name) in zip(eval_env_meta_list, eval_shape_meta_list, eval_env_name_list):

        # TODO: needs change 

        env_kwargs = dict(
            env_meta=env_meta,
            env_name=env_name,
            render=False,
            render_offscreen=config_data.experiment.render_video,
            use_image_obs=shape_meta["use_images"],
            seed=config_data.train.seed * 1000 + random.randint(0, 1000),
        )
        env = EnvUtils.create_env_from_metadata(**env_kwargs)
        # handle environment wrappers
        env = EnvUtils.wrap_env_from_config(env, config=config_data)  # apply environment warpper, if applicable

        env_lst.append(env)
    
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
    print(env_lst) 
    return rollout_model, env_lst[0], config, eval_env_horizon_list, data_logger, env_meta["env_lang"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(os.path.abspath(os.path.join(os.path.dirname(robomimic.__file__), os.pardir)), "datasets", "raw"),
    )

    parser.add_argument("--environment", type=str, default="NutAssemblySquare")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--config", type=str, default="single-arm-opposed",
                        help="Specified environment configuration if necessary")
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="agentview", help="Which camera to use for collecting demos")
    parser.add_argument("--device", type=str, default="spacemouse")
    parser.add_argument("--pos-sensitivity", type=float, default=1.8, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.8, help="How much to scale rotation user inputs")


    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_rollouts", type=int, default=50, help="Number of trajectories to collect / evaluate")
    parser.add_argument("--training-iter", type=int, default=-1)
    parser.add_argument("--human-sample-ratio", type=float, default=None)
    parser.add_argument("--human-sample-number", type=float, default=None)
    parser.add_argument("--all-demos", action="store_true")
    parser.add_argument("--term-condition", default="fixed_length", type=str)
    parser.add_argument("--render-onscreen", action="store_true")
    parser.add_argument("--sleep-time", default=0.08, type=float)
    parser.add_argument("--demos-embedding-path", default=None, type=str)
    parser.add_argument("--percentile", default=95, type=float)
    parser.add_argument("--human_always_override", action="store_true")


    parser.add_argument(
        "--detector_type",
        type=str,
        choices=[
                 "siriusfleet_failure", "siriusfleet_ood", "siriusfleet_combined"
                 ],
        default="siriusfleet_failure",
    )

    parser.add_argument(
        "--detector_checkpoints",
        type=str,
        nargs='+',
    )

    args = parser.parse_args()

    policy, env, config, eval_env_horizon_list, data_logger, task_lang = obtain_model_and_env(args)
    
    detector = detector_from_config(args.detector_type, args.detector_checkpoints, args)

    if is_v1:
        from robosuite.wrappers import VisualizationWrapper
        # Wrap this environment in a visualization wrapper
        env = VisualizationWrapper(env)#, disable_vis=True)

    GOOD_EPISODE_LENGTH = 100
    MAX_EPISODE_LENGTH = eval_env_horizon_list[0]
    SUCCESS_HOLD = 3

    script_start_time = time.time()

    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    print(current_directory)
    file_path = os.path.join(current_directory, "configs", "env_config.json")
    with open(file_path, 'r') as file:
        config_other = json.load(file)
        config_other["env_name"] = env.name

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config_other)

    tmp_directory = "/tmp/{}".format(str(script_start_time).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    # Setup printing options for numbers
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        env.viewer.add_keypress_callback("any", device.on_press)
        env.viewer.add_keyup_callback("any", device.on_release)
        env.viewer.add_keyrepeat_callback("any", device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard' or 'spacemouse'."
        )

    start = time.perf_counter()

    total_human_samples = 0
    total_samples = 0
    agent_success = 0
    success_at_time_lst = []

    excluded_eps = []

    t1, t2 = str(script_start_time).split(".")
    hdf5_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(hdf5_dir)

    num_traj_saved = 0
    num_traj_discarded = 0

    num_episodes = config.experiment.rollout.n
    
    while True:
        
        print("Collecting traj # {}".format(num_traj_saved + 1))
        
        ep_directory, timestep_count, num_human_samples, success_at_time, discard_traj = collect_trajectory(env, device, args)
        print("\tkeep this traj? ", not discard_traj)
        print("\tsuccess at time: ", success_at_time)
        print()
        if discard_traj:
            excluded_eps.append(ep_directory.split('/')[-1])
            num_traj_discarded += 1
        else:
            total_human_samples += num_human_samples
            total_samples += timestep_count
            
            # Print agent success condition changes
            # if num_human_samples == 0:
            #     agent_success += 1
            if success_at_time > -1: 
                agent_success += 1
            
            success_at_time_lst.append(success_at_time)
            num_traj_saved += 1

        meta_stats = dict(
            training_iter = args.training_iter,
            checkpoint = 'None' if args.checkpoint is None else args.checkpoint,
            total_time = time.perf_counter() - start,

            num_traj_saved = num_traj_saved,
            num_traj_discarded = num_traj_discarded,
            num_agent_success = agent_success,
            total_human_samples = total_human_samples,
            total_samples = total_samples,
            success_rate = agent_success / num_traj_saved if num_traj_saved > 0 else 0,
        )

        print("AGENT SUCCESS: ", agent_success)
        print("AGENT FAILURE: ", num_traj_saved - agent_success)
        print("AGENT SUCCESS RATE: ", agent_success / num_traj_saved if num_traj_saved > 0 else 0)

        gather_demonstrations_as_hdf5(tmp_directory, hdf5_dir, env_info, excluded_eps, meta_stats)

        if args.human_sample_ratio is not None:
            threshold = int(GOOD_EPISODE_LENGTH * args.num_rollouts * args.human_sample_ratio)
            print("# human samples: ", total_human_samples)
            print("progress: {}%".format(int(total_human_samples / threshold * 100)))
            print()
            if total_human_samples >= threshold:
                break
        elif args.human_sample_number is not None:
            threshold = args.human_sample_number
            print("progress: {}%".format(int(total_human_samples / threshold * 100)))
            if total_human_samples >= threshold:
                break
        else:
            if num_traj_saved == args.num_rollouts:
                break

    device.thread._delete()

    def count_traj(dataset_path):
        with h5py.File(dataset_path, 'r') as f:
            return len(f['data'])


print("\n### meta stats ###")
for (k, v) in meta_stats.items():
    print("{k}: {v}".format(k=k, v=v))
