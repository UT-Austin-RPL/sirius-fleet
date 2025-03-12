import argparse
import os
import time
import datetime

import robomimic
import robomimic.utils.hyperparam_utils as HyperparamUtils

from robomimic.scripts.config_gen.dataset_registry import get_ds_cfg

base_path = os.path.abspath(os.path.join(os.path.dirname(robomimic.__file__), os.pardir))

def scan_datasets(folder, postfix=".h5"):
    dataset_paths = []
    for root, dirs, files in os.walk(os.path.expanduser(folder)):
        for f in files:
            if f.endswith(postfix):
                dataset_paths.append(os.path.join(root, f))
    return dataset_paths


def get_generator(algo_name, config_file, args, algo_name_short=None, pt=False):
    if args.wandb_proj_name is None:
        strings = [
            algo_name_short if (algo_name_short is not None) else algo_name,
            args.name,
            args.env,
            args.mod,
        ]
        args.wandb_proj_name = '_'.join([str(s) for s in strings if s is not None])

    if args.script is not None:
        generated_config_dir = os.path.join(os.path.dirname(args.script), "json")
    else:
        curr_time = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d-%y-%H-%M-%S')
        generated_config_dir=os.path.join(
            '~/', 'tmp/autogen_configs/ril', algo_name, args.env, args.mod, args.name, curr_time, "json",
        )

    generator = HyperparamUtils.ConfigGenerator(
        base_config_file=config_file,
        generated_config_dir=generated_config_dir,
        wandb_proj_name=args.wandb_proj_name,
        script_file=args.script,
    )

    args.algo_name = algo_name
    args.pt = pt

    return generator


def set_env_settings(generator, args):
    if args.env in ["r2d2"]:
        assert args.mod == "im"
        generator.add_param(
            key="experiment.rollout.enabled",
            name="",
            group=-1,
            values=[
                False
            ],
        )
        generator.add_param(
            key="experiment.save.every_n_epochs",
            name="",
            group=-1,
            values=[50],
        )
        generator.add_param(
            key="experiment.mse.enabled",
            name="",
            group=-1,
            values=[False],
        ),
        generator.add_param(
            key="experiment.mse.every_n_epochs",
            name="",
            group=-1,
            values=[50],
        ),
        generator.add_param(
            key="experiment.mse.on_save_ckpt",
            name="",
            group=-1,
            values=[True],
        ),
        generator.add_param(
            key="experiment.mse.num_samples",
            name="",
            group=-1,
            values=[20],
        ),
        generator.add_param(
            key="experiment.mse.visualize",
            name="",
            group=-1,
            values=[True],
        ),
        if "observation.modalities.obs.low_dim" not in generator.parameters:
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot_state/eef_pos", "robot_state/eef_quat", "robot_state/gripper_position"]
                ],
            )
        if "observation.modalities.obs.rgb" not in generator.parameters:
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[
                    [
                        "camera/image/varied_camera_2_left_image", # in our setup: like a "left image"
                        "camera/image/varied_camera_1_left_image", # in our setup: like a "right image"
                        "camera/image/hand_camera_left_image",
                    ]
                ],
            )
        generator.add_param(
            key="observation.encoder.rgb.obs_randomizer_class",
            name="obsrand",
            group=-1,
            values=[
                ["ColorRandomizer", "CropRandomizer"], # jitter, followed by crop
            ],
            hidename=True,
        )
        generator.add_param(
            key="observation.encoder.rgb.obs_randomizer_kwargs",
            name="obsrandargs",
            group=-1,
            values=[
                [{}, {"crop_height": 116, "crop_width": 116, "num_crops": 1, "pos_enc": False}], # jitter, followed by crop
            ],
            hidename=True,
        )
        if ("observation.encoder.rgb.obs_randomizer_kwargs" not in generator.parameters) and \
            ("observation.encoder.rgb.obs_randomizer_kwargs.crop_height" not in generator.parameters):
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_height",
                name="",
                group=-1,
                values=[
                    116
                ],
            )
            generator.add_param(
                key="observation.encoder.rgb.obs_randomizer_kwargs.crop_width",
                name="",
                group=-1,
                values=[
                    116
                ],
            )
        # remove spatial softmax by default for r2d2 dataset
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.pool_class",
            name="",
            group=-1,
            values=[
                None
            ],
        )
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.pool_kwargs",
            name="",
            group=-1,
            values=[
                None
            ],
        )

        # language conditioned architecture
        generator.add_param(
            key="observation.encoder.rgb.core_class",
            name="",
            group=-1,
            values=[
                "VisualCoreLanguageConditioned"
            ],
        )
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.backbone_class",
            name="",
            group=-1,
            values=[
                "ResNet18ConvFiLM"
            ],
        )

        # specify dataset type is r2d2 rather than default robomimic
        generator.add_param(
            key="train.data_format",
            name="",
            group=-1,
            values=[
                "r2d2"
            ],
        )
        
        # here, we list how each action key should be treated (normalized etc)
        generator.add_param(
            key="train.action_config",
            name="",
            group=-1,
            values=[
                {
                    "action/cartesian_position":{
                        "normalization": "min_max",
                    },
                    "action/abs_pos":{
                        "normalization": "min_max",
                    },
                    "action/abs_rot_6d":{
                        "normalization": "min_max",
                        "format": "rot_6d",
                        "convert_at_runtime": "rot_euler",
                    },
                    "action/abs_rot_euler":{
                        "normalization": "min_max",
                        "format": "rot_euler",
                    },
                    "action/gripper_position":{
                        "normalization": "min_max",
                    },
                    "action/cartesian_velocity":{
                        "normalization": None,
                    },
                    "action/rel_pos":{
                        "normalization": None,
                    },
                    "action/rel_rot_6d":{
                        "format": "rot_6d",
                        "normalization": None,
                        "convert_at_runtime": "rot_euler",
                    },
                    "action/rel_rot_euler":{
                        "format": "rot_euler",
                        "normalization": None,
                    },
                    "action/gripper_velocity":{
                        "normalization": None,
                    },
                }
            ],
        )
        generator.add_param(
            key="train.dataset_keys",
            name="",
            group=-1,
            values=[[]],
        )
        if "train.action_keys" not in generator.parameters:
            generator.add_param(
                key="train.action_keys",
                name="ac_keys",
                group=-1,
                values=[
                    [
                        "action/rel_pos",
                        "action/rel_rot_euler",
                        "action/gripper_velocity",
                    ],
                ],
                value_names=[
                    "rel",
                ],
                hidename=True,
            )
    elif args.env == "robocasa":
        generator.add_param(
            key="train.action_config",
            name="",
            group=-1,
            values=[
                {
                    "actions":{
                        "normalization": None,
                    },
                    "action_dict/abs_pos": {
                        "normalization": "min_max"
                    },
                    "action_dict/abs_rot_axis_angle": {
                        "normalization": "min_max",
                        "format": "rot_axis_angle"
                    },
                    "action_dict/abs_rot_6d": {
                        "normalization": None,
                        "format": "rot_6d"
                    },
                    "action_dict/rel_pos": {
                        "normalization": None,
                    },
                    "action_dict/rel_rot_axis_angle": {
                        "normalization": None,
                        "format": "rot_axis_angle"
                    },
                    "action_dict/rel_rot_6d": {
                        "normalization": None,
                        "format": "rot_6d"
                    },
                    "action_dict/gripper": {
                        "normalization": None,
                    },
                    "action_dict/base_mode": {
                        "normalization": None,
                    }
                }
            ],
        )

        # language conditioned architecture
        generator.add_param(
            key="observation.encoder.rgb.core_class",
            name="",
            group=-1,
            values=[
                "VisualCoreLanguageConditioned"
            ],
        )
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.backbone_class",
            name="",
            group=-1,
            values=[
                "ResNet18ConvFiLM"
            ],
        )

        env_kwargs = {
            "generative_textures": None,
            "scene_split": None,
            "style_ids": None,
            "layout_ids": None,
            "layout_and_style_ids": [[1, 1], [2, 2], [4, 4], [6, 9], [7, 10]],
        }
        if args.abs_actions:
            env_kwargs["controller_configs"] = {"control_delta": False}

        # don't use generative textures for evaluation
        generator.add_param(
            key="experiment.env_meta_update_dict",
            name="",
            group=-1,
            values=[{"env_kwargs": env_kwargs}],
        )

        generator.add_param(
            key="observation.encoder.rgb.obs_randomizer_kwargs",
            name="obsrandargs",
            group=-1,
            values=[
                {"crop_height": 116, "crop_width": 116, "num_crops": 1, "pos_enc": False},
            ],
            hidename=True,
        )
        generator.add_param(
            key="experiment.rollout.n",
            name="",
            group=-1,
            values=[25],
            value_names=[""],
        )
        generator.add_param(
            key="experiment.rollout.horizon",
            name="",
            group=-1,
            values=[500],
            value_names=[""],
        )
        if args.mod == 'im':
            
            generator.add_param(
                key="observation.modalities.obs.rgb",
                name="",
                group=-1,
                values=[
                    ["robot0_agentview_left_image",
                    "robot0_eye_in_hand_image"],
                
                ],
            ) 

        else:
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ["robot0_eef_pos",
                     "robot0_eef_quat",
                     "robot0_gripper_qpos",
                     "robot0_base_pos",
                     "object",
                    ]
                ],
            )
    
    elif args.env == "mutex":
        generator.add_param(
            key="train.action_config",
            name="",
            group=-1,
            values=[
                {
                    "actions":{
                        "normalization": None,
                    },
                    "action_dict/abs_pos": {
                        "normalization": "min_max"
                    },
                    "action_dict/abs_rot_axis_angle": {
                        "normalization": "min_max",
                        "format": "rot_axis_angle"
                    },
                    "action_dict/abs_rot_6d": {
                        "normalization": None,
                        "format": "rot_6d"
                    },
                    "action_dict/rel_pos": {
                        "normalization": None,
                    },
                    "action_dict/rel_rot_axis_angle": {
                        "normalization": None,
                        "format": "rot_axis_angle"
                    },
                    "action_dict/rel_rot_6d": {
                        "normalization": None,
                        "format": "rot_6d"
                    },
                    "action_dict/gripper": {
                        "normalization": None,
                    },
                    "action_dict/base_mode": {
                        "normalization": None,
                    }
                }
            ],
        )

        # language conditioned architecture
        generator.add_param(
            key="observation.encoder.rgb.core_class",
            name="",
            group=-1,
            values=[
                "VisualCoreLanguageConditioned"
            ],
        )
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.backbone_class",
            name="",
            group=-1,
            values=[
                "ResNet18ConvFiLM"
            ],
        )

        generator.add_param(
            key="observation.encoder.rgb.obs_randomizer_kwargs",
            name="obsrandargs",
            group=-1,
            values=[
                {"crop_height": 116, "crop_width": 116, "num_crops": 1, "pos_enc": False},
            ],
            hidename=True,
        )
        
        generator.add_param(
            key="experiment.rollout.enabled",
            name="",
            group=-1,
            values=[
                False
            ],
        )

        if not args.train_reward:
            generator.add_param(
                key="experiment.save.every_n_epochs",
                name="",
                group=-1,
                values=[50],
            )        
            
        assert args.mod == "im"

        if not args.no_ld:
            generator.add_param(
                key="observation.modalities.obs.low_dim",
                name="",
                group=-1,
                values=[
                    ['gripper_states', 
                     'joint_states']
                ],
            )

        generator.add_param(
            key="observation.modalities.obs.rgb",
            name="",
            group=-1,
            values=[
                ['agentview_rgb', 
                 'eye_in_hand_rgb'],
            ],
        ) 
    else:
        raise ValueError


def set_mod_settings(generator, args):
    if args.mod == 'ld':
        if "experiment.save.epochs" not in generator.parameters:
            generator.add_param(
                key="experiment.save.epochs",
                name="",
                group=-1,
                values=[
                    [2000]
                ],
            )
    elif args.mod == 'im':
        if "experiment.save.every_n_epochs" not in generator.parameters:
            generator.add_param(
                key="experiment.save.every_n_epochs",
                name="",
                group=-1,
                values=[200],
            )

        generator.add_param(
            key="experiment.epoch_every_n_steps",
            name="",
            group=-1,
            values=[500],
        )
        if "train.num_data_workers" not in generator.parameters:
            generator.add_param(
                key="train.num_data_workers",
                name="",
                group=-1,
                values=[5],
            )
        generator.add_param(
            key="train.hdf5_cache_mode",
            name="",
            group=-1,
            # values=["low_dim"],
            values=[None],
        )
        if args.batch_size > 0:
            generator.add_param(
                key="train.batch_size",
                name="",
                group=-1,
                values=[args.batch_size],
            )
        else:
            generator.add_param(
                key="train.batch_size",
                name="",
                group=-1,
                values=[16],
            )
        if "train.num_epochs" not in generator.parameters:
            if args.num_epochs > 0:
                generator.add_param(
                    key="train.num_epochs",
                    name="",
                    group=-1,
                    values=[args.num_epochs],
                )
            else:
                generator.add_param(
                    key="train.num_epochs",
                    name="",
                    group=-1,
                    values=[1000],
                )
        if "experiment.rollout.rate" not in generator.parameters:
            generator.add_param(
                key="experiment.rollout.rate",
                name="",
                group=-1,
                values=[200],
            )


def set_debug_mode(generator, args):
    if not args.debug:
        return

    generator.add_param(
        key="experiment.mse.every_n_epochs",
        name="",
        group=-1,
        values=[2],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.mse.visualize",
        name="",
        group=-1,
        values=[True],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.rollout.n",
        name="",
        group=-1,
        values=[2],
        value_names=[""],
    )
    
    # set horizon to 30
    ds_cfg_list = generator.parameters["train.data"].values
    for ds_cfg in ds_cfg_list:
        for d in ds_cfg:
            d["horizon"] = 30
    
    generator.add_param(
        key="experiment.rollout.rate",
        name="",
        group=-1,
        values=[2],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.epoch_every_n_steps",
        name="",
        group=-1,
        values=[2], #50
        value_names=[""],
    )
    generator.add_param(
        key="experiment.save.every_n_epochs",
        name="",
        group=-1,
        values=[2],
        value_names=[""],
    )
    generator.add_param(
        key="experiment.validation_epoch_every_n_steps",
        name="",
        group=-1,
        values=[2],
        value_names=[""],
    )
    generator.add_param(
        key="train.num_epochs",
        name="",
        group=-1,
        values=[2],
        value_names=[""],
    )
    if args.name is None:
        generator.add_param(
            key="experiment.name",
            name="",
            group=-1,
            values=["debug"],
            value_names=[""],
        )
    generator.add_param(
        key="experiment.save.enabled",
        name="",
        group=-1,
        values=[False],
        value_names=[""],
    )
    generator.add_param(
        key="train.hdf5_cache_mode",
        name="",
        group=-1,
        # values=["low_dim"],
        values=[None],
        value_names=[""],
    )
    generator.add_param(
        key="train.num_data_workers",
        name="",
        group=-1,
        values=[3],
    )


def set_output_dir(generator, args):
    assert args.name is not None

    vals = generator.parameters["train.output_dir"].values

    for i in range(len(vals)):
        vals[i] = os.path.join(vals[i], args.name)


def set_wandb_mode(generator, args):
    generator.add_param(
        key="experiment.logging.log_wandb",
        name="",
        group=-1,
        values=[not args.no_wandb],
    )

def set_rollout_mode(generator, args):
    if args.no_rollout:
        generator.add_param(
            key="experiment.rollout.enabled",
            name="",
            group=-1,
            values=[False],
        )


def set_num_seeds(generator, args):
    if "train.seed" not in generator.parameters:
        if args.n_seeds is not None:
            generator.add_param(
                key="train.seed",
                name="seed",
                group=-10,
                values=[i + 1 for i in range(args.n_seeds)],
                prepend=True,
            )
        else:
            generator.add_param(
                key="train.seed",
                name="seed",
                group=-10,
                values=[123],
                prepend=True,
            )


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        type=str,
    )

    parser.add_argument(
        "--env",
        type=str,
        default='robocasa',
    )

    parser.add_argument(
        '--mod',
        type=str,
        choices=['ld', 'im'],
        default='im',
    )

    parser.add_argument(
        "--ckpt_mode",
        type=str,
        choices=["off", "all", "best_only"],
        default=None,
    )

    parser.add_argument(
        "--script",
        type=str,
        default=None
    )

    parser.add_argument(
        "--wandb_proj_name",
        type=str,
        default=None
    )

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    parser.add_argument(
        '--no_video',
        action='store_true'
    )

    parser.add_argument(
        '--no_rollout',
        action='store_true'
    )

    parser.add_argument(
        "--tmplog",
        action="store_true",
    )

    parser.add_argument(
        "--nr",
        type=int,
        default=-1
    )

    parser.add_argument(
        "--no_wandb",
        action="store_true",
    )

    parser.add_argument(
        "--abs_actions",
        action="store_true",
    )

    parser.add_argument(
        "--n_seeds",
        type=int,
        default=None
    )

    parser.add_argument(
        "--num_cmd_groups",
        type=int,
        default=None
    )

    parser.add_argument(
        "--no_ld",
        action="store_true",
    )

    parser.add_argument(
        "--two_images",
        action="store_true",
    )         

    parser.add_argument(
        "--single_image",
        action="store_true",
    )     

    parser.add_argument(
        "--batch_size",
        type=int,
        default=-1,
    )         

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=-1,
    )       

    parser.add_argument(
        "--validate",
        action="store_true",
    )

    parser.add_argument(
        "--train_reward",
        action="store_true",
    )

    return parser


def make_generator(args, make_generator_helper):
    if args.tmplog or args.debug and args.name is None:
        args.name = "debug"
    else:
        time_str = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d-')
        args.name = time_str + str(args.name)

    if args.debug or args.tmplog:
        args.no_wandb = True

    if args.wandb_proj_name is not None:
        pass

    if (args.debug or args.tmplog) and (args.wandb_proj_name is None):
        args.wandb_proj_name = 'debug'

    if not args.debug:
        assert args.name is not None

    # make config generator
    generator = make_generator_helper(args)

    if args.ckpt_mode is None:
        if args.pt:
            args.ckpt_mode = "all"
        else:
            args.ckpt_mode = "best_only"

    set_env_settings(generator, args)
    set_mod_settings(generator, args)
    set_output_dir(generator, args)
    set_num_seeds(generator, args)
    set_wandb_mode(generator, args)

    # set the debug settings last, to override previous setting changes
    set_debug_mode(generator, args)

    set_rollout_mode(generator, args)

    """ validate settings """
    if "experiment.validate" not in generator.parameters:
        if args.validate:
            generator.add_param(
                key="experiment.validate",
                name="",
                group=-1,
                values=[
                    True,
                ],
            )
        else:
            generator.add_param(
                key="experiment.validate",
                name="",
                group=-1,
                values=[
                    False,
                ],
            )

    if "experiment.save.on_best_rollout_success_rate" not in generator.parameters:
        generator.add_param(
            key="experiment.save.on_best_rollout_success_rate",
            name="",
            group=-1,
            values=[
                False,
            ],
        )

    # generate jsons and script
    generator.generate(override_base_name=True)
