"""
The main entry point for training policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    debug (bool): set this flag to run a quick training run for debugging purposes    
"""

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

VALIDATE = True

def train(config, device):
    """
    Train a model using the algorithm.
    """

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    # set num workers
    torch.set_num_threads(1)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir, vis_dir = TrainUtils.get_exp_dir(config)

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

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

    # setup for a new training run
    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta_list[0]["all_shapes"],
        ac_dim=shape_meta_list[0]["ac_dim"],
        device=device,
    )
    
    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    # load training data
    lang_encoder = LangUtils.LangEncoder(
        device=device,
    )
    
    # the dataset loaded is valid dataset by default
    validset, _ = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"], lang_encoder=lang_encoder)

    if VALIDATE:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True
        )
    else:
        valid_loader = None

    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    # if checkpoint is specified, load in model weights
    ckpt_path = config.experiment.ckpt_path

    epoch = 800
    
    while True:
        
        epoch_ckpt_name = "model_epoch_{}.pth".format(epoch)
        
        full_model_path = os.path.join(ckpt_path, epoch_ckpt_name)
        
        print("LOADING MODEL WEIGHTS FROM " + full_model_path)
        from robomimic.utils.file_utils import maybe_dict_from_checkpoint
        
        try:
            ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=full_model_path)
            model.deserialize(ckpt_dict["model"])
        except:
            print(f"Model {epoch_ckpt_name} does not exist."); break
        
        # Evaluate the model on validation set
        if VALIDATE:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps)
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Valid_{}".format(k[5:]), v, epoch)
                else:
                    data_logger.record("Valid/{}".format(k), v, epoch)

                if k == "reconstructions":
                    for v_key in v:
                        if "image" in v_key:
                            assert (step_log[k][v_key] >= 0).all()
                            assert (step_log[k][v_key] <= 1).all()
                            data_logger.record("{}/{}".format(k, v_key), v[v_key], epoch, data_type='image')

            print("Validation Epoch {}".format(epoch))
            step_log.pop("reconstructions", None)
            step_log.pop("targets", None)
            print(json.dumps(step_log, sort_keys=True, indent=4))
        
        epoch += 100

    # terminate logging
    data_logger.close()


def main(args):

    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if args.dataset_config is not None:
        ext_dataset_cfg = json.load(open(args.dataset_config, 'r'))
        dataset_config = config_factory(ext_dataset_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with dataset_config.values_unlocked():
            dataset_config.update(ext_dataset_cfg)
           
        with config.values_unlocked(): 
            config.train.data = dataset_config.train.data

    if args.ckpt_path is not None:
        config.experiment.ckpt_path = args.ckpt_path

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        train(config, device=device)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset_config",
        type=str,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    args = parser.parse_args()
    main(args)
