"""
Implementation of Behavioral Cloning (BC).
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.models.base_nets as BaseNets
import robomimic.models.obs_nets as ObsNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.models.vae_nets as VAENets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.models.base_nets import MLP, ResidualMLP

import torch.distributions as D

import robomimic.models.dyn_nets as DynNets
import numpy as np

from robomimic.algo import algo_factory
from robomimic.config import config_factory
import json

from robomimic.algo.bc import BC_Transformer_GMM
from robomimic.utils.dyn_utils import activation_map, log_stoch_inputs, log_data_attributes, kl_loss, zdistr, diag_normal, loss_name_mapping, select_batch, confusion_matrix

import time

from robomimic.models.obs_nets import MIMO_MLP, MIMO_Transformer_Dyn, ObservationGroupEncoder

from diffusers.models.autoencoders.vae import Encoder, Decoder, DecoderOutput 
from diffusers import AutoencoderKL

import os

import torchvision.transforms as transforms

def move_to_cpu(object):
    for k in object:
        if "image" in k or "rgb" in k:
            object[k] = object[k].detach().cpu().numpy()

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f'{method.__name__} took: {te-ts:.4f} sec')
        return result
    return timed

@register_algo_factory_func("world_model")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.
    Args:
        algo_config (Config instance): algo config
    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """ 
    return DynamicsTrainer, {}

def dynamics_class(configs):
    
    try:
        if configs.train_reward:

            assert len(configs.load_ckpt) >= 0 # must load ckpt from dyn model
            assert configs.use_unet
            
            return DynamicsModel_DeterMLP_Unet_VAE_EmbedOnly_Reward
        
        else:
            
            if configs.use_unet:

                if configs.deterministic:
                    return DynamicsModel_DeterMLP_Unet
                else:
                    if configs.dyn_train_embed_only:
                        return DynamicsModel_DeterMLP_Unet_VAE_EmbedOnly
                    else:
                        return DynamicsModel_DeterMLP_Unet_VAE
                        
            return DynamicsModel_DeterMLP

    except:
            # hardcoded version
            return DynamicsModel_DeterMLP_Unet

class DynamicsTrainer(PolicyAlgo):
    
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """

        self.dyn_class = dynamics_class(self.algo_config.dyn)

        self.nets = nn.ModuleDict()
    
        self.nets["policy"] = self.dyn_class(self.obs_shapes,
                                        self.ac_dim,
                                        self.algo_config.dyn,
                                        self.algo_config,
                                        self.obs_config,
                                        self.goal_shapes,
                                        self.device,
                                        global_config=self.global_config
                                        )
        
        self._set_params_from_config()
        self.nets = self.nets.float().to(self.device)

    def train_on_batch(self, batch, epoch, validate=False, step=-1, max_step=-2):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DynamicsTrainer, self).train_on_batch(batch, epoch, validate=validate)
            
            discard_image = step != max_step and (not validate)
            
            predictions = self._forward_training(batch, discard_image=discard_image)
            
            losses = self._compute_losses(predictions, batch)
            
            if discard_image:
                predictions.pop("reconstructions", None)
                predictions.pop("targets", None)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)

        return info

    def _loss_name(self):
        return "total_loss"

    def _train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()        
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses[self._loss_name()],
            max_grad_norm=self.global_config.train.max_grad_norm,
        )
        info["policy_grad_norms"] = policy_grad_norms

        # step through optimizers
        for k in self.lr_schedulers:
            if self.lr_schedulers[k] is not None:
                self.lr_schedulers[k].step()
        return info

    def _downscale_img(self, obs):
        
        transformed_batch = {}
        
        for k, images in obs.items():
            if len(images.shape) != 5:
                continue
            B, T = images.shape[:2]

            data_permuted = images.reshape(B * T, 128, 128, 3).permute(0, 3, 1, 2)  # Now shape (B*T, 3, 128, 128)
            img_size = self._scaled_img_size
            resized_data = F.interpolate(data_permuted, size=(img_size, img_size), mode='bilinear', align_corners=False)
            resized_data_back = resized_data.permute(0, 2, 3, 1).reshape(B, T, img_size, img_size, 3)
            
            transformed_batch[k] = resized_data_back
        
        return transformed_batch

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        h = self.context_length
        
        input_batch["obs"] = {k: batch["obs"][k][:, :h, :] for k in batch["obs"]}
        
        if self.algo_config.dyn.downscale_img:
            
            self._scaled_img_size = self.algo_config.dyn.scaled_img_size
            input_batch["obs"] = self._downscale_img(input_batch["obs"])
        
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present

        if self.supervise_all_steps:
            # supervision on entire sequence (instead of just current timestep)
            if self.pred_future_acs:
                ac_start = h - 1
            else:
                ac_start = 0
            input_batch["actions"] = batch["actions"][:, ac_start:ac_start+h, :]
            
            # Assume to predict future first 
            if "intv_labels" in batch.keys():
                input_batch["intv_labels"] = batch["intv_labels"][:, ac_start:ac_start+h]

        else:
            # just use current timestep
            input_batch["actions"] = batch["actions"][:, h-1, :]

        if self.pred_future_acs:
            assert input_batch["actions"].shape[1] == h

        input_batch = TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        return input_batch
        
    def _set_params_from_config(self):
        """
        Read specific config variables we need for training / eval.
        Called by @_create_networks method
        """
        self.context_length = self.algo_config.transformer.context_length
        self.supervise_all_steps = self.algo_config.transformer.supervise_all_steps
        self.pred_future_acs = self.algo_config.transformer.pred_future_acs
        if self.pred_future_acs:
            assert self.supervise_all_steps is True
         
    def _forward_training(self, batch, epoch=None, discard_image=True): 
        return self.nets["policy"]._forward_training(batch, discard_image=discard_image)
    
    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC_Transformer_GMM algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        loss_dict = OrderedDict()
        for key in predictions:
            if "loss" in key:
                loss_dict[key] = predictions[key]
                
        return loss_dict

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of info
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(DynamicsTrainer, self).log_info(info)
        
        # Automatically log losses
        for loss_key, loss_value in info["losses"].items():
            mapped_name = loss_name_mapping.get(loss_key, None)
            if mapped_name:
                log[mapped_name] = loss_value.item()

        # Log policy grad norms if present
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        
        # process image reconstruction results
        if "reconstructions" in info["predictions"]:
            
            if type(info["predictions"]["reconstructions"]) != dict:  
                log["reconstructions"] = {"image": info["predictions"]["reconstructions"]}
            else:
                log["reconstructions"] = info["predictions"]["reconstructions"]
                
            for k in log["reconstructions"]:
                if "image" in k or "rgb" in k:
                    shape = list(range(len(log["reconstructions"][k].shape)))
                    log["reconstructions"][k] = torch.permute(log["reconstructions"][k], [1, 2, 0] + shape[3:])
                    curr_shape = list(log["reconstructions"][k].shape)
                    log["reconstructions"][k] = log["reconstructions"][k].reshape(curr_shape[:2] + [curr_shape[2] * curr_shape[3]] + [curr_shape[4]])
                        
        if "targets" in info["predictions"]:
            log["targets"] = info["predictions"]["targets"]
            for k in log["targets"]:
                if "image" in k or "rgb" in k:
                    shape = list(range(len(log["targets"][k].shape)))
                    log["targets"][k] = torch.permute(log["targets"][k], [1, 2, 0] + shape[3:])
                    curr_shape = list(log["targets"][k].shape)
                    log["targets"][k] = log["targets"][k].reshape(curr_shape[:2] + [curr_shape[2] * curr_shape[3]] + [curr_shape[4]])

        if self.algo_config.dyn.train_reward:
            log["loss/Reward Loss"] = info["losses"]["total_loss"].item()
            log["loss/reward/acc all"] = info["predictions"]["reward_overall_acc"].item()
            log["loss/reward/rollout"] = info["predictions"]["reward_rollout_acc"].item()
            log["loss/reward/intv"] = info["predictions"]["reward_intv_acc"].item()
            log["loss/reward/preintv"] = info["predictions"]["reward_preintv_acc"].item()
            
            if "confusion_matrix" in info["predictions"]:
                log["confusion_matrix"] = info["predictions"]["confusion_matrix"]

            if "class_samples" in info["predictions"]:
                for key in info["predictions"]["class_samples"]:
                    log["class_samples/" + key] = info["predictions"]["class_samples"][key].item()

        return log    
      
class DynamicsModel(nn.Module):
    
    def __init__(self,
                 obs_shapes,
                 ac_dim,
                 wm_configs,
                 algo_config,
                 obs_config,
                 goal_shapes,
                 device,
                 global_config=None,
                 ):

        super(DynamicsModel, self).__init__()

        self.action_dim = ac_dim
        self.ac_dim = ac_dim

        self.wm_configs = wm_configs
        self.algo_config = algo_config
        self.obs_config = obs_config
        self.obs_shapes = obs_shapes
        self.goal_shapes = goal_shapes
        self.device = device

        self.global_config = global_config

        # configs
        self.encoder_layer_dims = [1024, 1024]
        self.decoder_layer_dims = [1024, 1024]
        
        self.obs_embedding_dim = 1024
        
        self.dyn_embed_dim = None # defined later
        self._encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        
        self.output_shapes = self.obs_shapes

        self.nets = nn.ModuleDict()

        self.image_size = int(self.wm_configs.scaled_img_size if self.wm_configs.downscale_img else 128)

        self.create_encoder()
        self.create_dynamics()
        self.create_decoder()

        self.nets = self.nets.float().to(self.device)
        
    # create encoder
    def create_encoder(self):
         
        encoder_obs_group_shapes = OrderedDict()
        encoder_obs_group_shapes["input"] = OrderedDict(self.obs_shapes)

        encoder_output_shapes = OrderedDict(
            embed=(self.obs_embedding_dim,)
        )

        self.nets["encoder"] = MIMO_MLP(
            input_obs_group_shapes=encoder_obs_group_shapes,
            output_shapes=encoder_output_shapes,
            layer_dims=self.encoder_layer_dims,
            encoder_kwargs=self._encoder_kwargs,
        )
        
        self.dyn_embed_dim_dict = self.nets["encoder"].output_shape()

        
    # create dynamics model
    def create_dynamics(self):
        assert NotImplementedError
        
    # create decoder 
    def create_decoder(self):

        decoder_obs_group_shapes = OrderedDict()
        decoder_obs_group_shapes["input"] = OrderedDict(
            #embed=(self.dyn_embed_dim,)
            self.dyn_embed_dim_dict
        )

        if self.wm_configs.image_output_activation is None:
            image_output_activation = None
        elif self.wm_configs.image_output_activation == "sigmoid":
            image_output_activation = nn.Sigmoid
        elif self.wm_configs.image_output_activation == "softmax":
            image_output_activation = nn.Softmax
        else:
            assert NotImplementedError

        self.nets["decoder"] = MIMO_MLP(
            input_obs_group_shapes=decoder_obs_group_shapes,
            output_shapes=self.output_shapes,
            layer_dims=self.decoder_layer_dims,
            encoder_kwargs=self._encoder_kwargs,
            image_output_activation=image_output_activation,
        )
        
    # encode function
    def encode(self, inputs):
        return self.nets["encoder"](
            input=inputs,
        )
    # dynamics model function
    def dynamics_predict(self, embed, actions):
        return self.nets["dynamics"](
            embed, actions
        )
    
    # decode function
    def decode(self, inputs):

        reconstruction = self.nets["decoder"](input=inputs)
        return reconstruction        

    def _forward_training(self, batch, discard_image=True):
        
        obs = batch["obs"]
        actions = batch["actions"]
        
        # encode image
        obs_to_embed = {"inputs": obs}
        embed = TensorUtils.time_distributed(obs_to_embed, self.encode, inputs_as_kwargs=True)

        # debug: see image reconstruction without dynamics
        if not self.wm_configs.no_dyn_debug:
            # dynamics predict
            latent_features = self.dynamics_predict(embed=embed["embed"], actions=actions)
        else:
            embed["embed"] = embed["embed"][:,-10:,:]
            latent_features = embed

        # decode image
        decode_input = {"inputs": latent_features}
        reconstructions = TensorUtils.time_distributed(decode_input, self.decode, inputs_as_kwargs=True)
        
        if not self.wm_configs.no_dyn_debug:
            batch_segment = select_batch(batch, 11, 20) # predict 11 images
        else:
            batch_segment = select_batch(batch, 10, 20) # predict 10 images
        obs_segment = batch_segment["obs"]
        
        reconstruction_loss = self.reconstruction_loss(
            reconstructions=reconstructions,
            targets=obs_segment,
        )

        targets = {}
        for k in self.output_shapes:
            targets[k] = obs[k]
        
        if not discard_image:
            returned = {
                "latent_features": latent_features,
                "reconstruction_loss": reconstruction_loss,
                "reconstructions": reconstructions,
                "targets": targets,
            }
        else:
            returned = {
                "latent_features": latent_features,
                "reconstruction_loss": reconstruction_loss,
            }
        return returned
    
    def reconstruction_loss(self, reconstructions, targets):
        random_key = list(reconstructions.keys())[0]
        batch_size = reconstructions[random_key].shape[0]
        seq_length = reconstructions[random_key].shape[1]
        num_mods = len(reconstructions.keys())

        # collect errors per modality, while preserving shapes in @reconstructions
        recons_errors = []
        for k in reconstructions:
            L2_loss = (reconstructions[k] - targets[k]).pow(2)
            recons_errors.append(L2_loss)
        # reduce errors across modalities and dimensions
        if self.wm_configs.decoder_reconstruction_sum_across_elements:
            # average across batch but sum across modalities and dimensions
            loss = sum([x.sum() for x in recons_errors])
            loss /= batch_size 
            loss /= seq_length
        else:
            # compute mse loss in each modality and average across modalities
            loss = sum([x.mean() for x in recons_errors])
            loss /= num_mods
            
        return loss
    
    
class DynamicsModel_DeterMLP(DynamicsModel):
    
    def __init__(self,
                 obs_shapes,
                 ac_dim,
                 wm_configs,
                 algo_config,
                 obs_config,
                 goal_shapes,
                 device,
                 global_config=None,
                 ):
        
        super(DynamicsModel_DeterMLP, self).__init__(
                 obs_shapes,
                 ac_dim,
                 wm_configs,
                 algo_config,
                 obs_config,
                 goal_shapes,
                 device,
                 global_config,
                 )
        
    def create_dynamics(self):
        
        # deterministic version
        self.nets["dynamics"] = DynNets.Dynamics(
            embed_dim=self.dyn_embed_dim_dict["embed"][0],
            action_dim=self.ac_dim,
            wm_configs=self.wm_configs,
            algo_configs=self.algo_config
        )
        

class DynamicsModel_DeterMLP_Unet(DynamicsModel_DeterMLP):
    """
    Using encoder decoder implementation from diffuser.
    """
    
    def __init__(self,
                 obs_shapes,
                 ac_dim,
                 wm_configs,
                 algo_config,
                 obs_config,
                 goal_shapes,
                 device,
                 global_config=None,
                 ):
        
        self.version = "new"
        
        # how to combine the image embeddings
        try:
            self.obs_fusion_method = wm_configs.obs_fusion_method
            self.deterministic = wm_configs.deterministic
        except:
            self.obs_fusion_method = "concat"
            self.deterministic = False
            self.version = "old" # previous dev version
            
        assert self.obs_fusion_method  == "concat"
            
        super(DynamicsModel_DeterMLP_Unet, self).__init__(
                 obs_shapes,
                 ac_dim,
                 wm_configs,
                 algo_config,
                 obs_config,
                 goal_shapes,
                 device,
                 global_config,
                 )
    
    def create_encoder(self):

        # Hardcoded versions of encoders list
        
        if self.version == "old":
            self.encoder_index = ["encoder", "encoder_1", "encoder_2"]
            self.quant_conv_index = ["quant_conv", "quant_conv_1", "quant_conv_2"]
            self.AutoencoderKL_index = ["AutoencoderKL", "AutoencoderKL_1", "AutoencoderKL_2"]    
            self.obs_sequence = self.wm_configs.obs_sequence

        else:
            self.encoder_index = ["encoder_0", "encoder_1", "encoder_2"]
            self.quant_conv_index = ["quant_conv_0", "quant_conv_1", "quant_conv_2"]    
            self.AutoencoderKL_index = ["AutoencoderKL_0", "AutoencoderKL_1", "AutoencoderKL_2"]    
            self.obs_sequence = self.wm_configs.obs_sequence

        for idx in range(len(self.obs_sequence)):
            
            self.nets[self.encoder_index[idx]] = Encoder(
                in_channels=self.wm_configs.unet.in_channels,
                out_channels=self.wm_configs.unet.latent_channels,
                down_block_types=self.wm_configs.unet.down_block_types,
                block_out_channels=self.wm_configs.unet.block_out_channels,
                layers_per_block=self.wm_configs.unet.layers_per_block,
                act_fn=self.wm_configs.unet.act_fn,
                norm_num_groups=self.wm_configs.unet.norm_num_groups,
                double_z=False,
            )

            self.nets[self.quant_conv_index[idx]] = nn.Conv2d(self.wm_configs.unet.latent_channels, 
                                                self.wm_configs.unet.latent_channels, 1)
        
            self.nets[self.AutoencoderKL_index[idx]] = AutoencoderKL(
                    block_out_channels=[32, 64],
                    in_channels=3,
                    out_channels=3,
                    down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
                    up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
                    latent_channels=4,
                )
            
            self.nets[self.AutoencoderKL_index[idx]] = AutoencoderKL(
                    block_out_channels=[32, 64],
                    in_channels=3,
                    out_channels=3,
                    down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
                    up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
                    latent_channels=4,
                )
            
        self._set_embedding_dim()
    
    def _set_embedding_dim(self):
        
        self.image_size = 84
        
        assert self.obs_fusion_method == "concat"
        
        if self.obs_fusion_method == "concat": # simple concat
            
            output_dim = int(self.image_size / 2)
            IMG_EMBEDDING_SIZE = 4 * output_dim * output_dim 
            self.embed_dim = IMG_EMBEDDING_SIZE * len(self.obs_sequence)
    
    def create_decoder(self):
    
        if self.version == "old":
            self.decoder_index = ["decoder", "decoder_1", "decoder_2"]
            self.post_quant_conv_index = ["post_quant_conv", "post_quant_conv_1", "post_quant_conv_2"]   
    
        else:
            self.decoder_index = ["decoder_0", "decoder_1", "decoder_2"]
            self.post_quant_conv_index = ["post_quant_conv_0", "post_quant_conv_1", "post_quant_conv_2"]  
    
        for idx in range(len(self.obs_sequence)):
    
            self.nets[self.decoder_index[idx]] = Decoder(
                in_channels=self.wm_configs.unet.latent_channels,
                out_channels=self.wm_configs.unet.out_channels,
                up_block_types=self.wm_configs.unet.up_block_types,
                block_out_channels=self.wm_configs.unet.block_out_channels,
                layers_per_block=self.wm_configs.unet.layers_per_block,
                norm_num_groups=self.wm_configs.unet.norm_num_groups,
                act_fn=self.wm_configs.unet.act_fn,
            )
            
            self.nets[self.post_quant_conv_index[idx]] = nn.Conv2d(self.wm_configs.unet.latent_channels, 
                                                    self.wm_configs.unet.latent_channels, 1)
        
    def create_dynamics(self):
        
        # deterministic version
        self.nets["dynamics"] = DynNets.Dynamics(
            embed_dim=self.embed_dim,
            action_dim=self.ac_dim,
            wm_configs=self.wm_configs,
            algo_configs=self.algo_config
        )
    
    # encode function
    def encode(self, inputs):
        
        img_idx = self._img_idx

        encoded = self.nets[self.encoder_index[img_idx]](inputs)
        encoded_quant_conv = self.nets[self.quant_conv_index[img_idx]](encoded)
        
        return encoded_quant_conv

    # decode function
    def decode(self, inputs):      
        
        img_idx = self._dec_img_idx  
            
        inputs = self.nets[self.post_quant_conv_index[img_idx]](
                inputs
            )
        reconstruction = self.nets[self.decoder_index[img_idx]](
            inputs
        )
        reconstruction = nn.Sigmoid()(reconstruction)
        
        return reconstruction
       
    def _generate_latent_futures(self, batch):
        
        obs = batch["obs"]
        actions = batch["actions"]
        
        # encode image

        feats = []
        for idx in range(len(self.obs_sequence)):
            img_input = obs[self.obs_sequence[idx]]
            
            self._img_idx = idx
            
            obs_to_embed = {"inputs": img_input}
            
            embed = TensorUtils.time_distributed(obs_to_embed, self.encode, inputs_as_kwargs=True)
            
            embed = torch.flatten(embed, start_dim=2)
            feats.append(embed)
            
        embed = torch.cat(feats, dim=-1)
            
        # dynamics predict
        latent_features = self.dynamics_predict(embed=embed, actions=actions)
        
        return embed, latent_features
       
    def _forward_training(self, batch, discard_image=True):
        
        embed, latent_features = self._generate_latent_futures(batch)

        if self.obs_fusion_method == "concat":
            latent_features = latent_features["embed"] 
            img_embeddings = torch.chunk(latent_features, chunks=len(self.obs_sequence), dim=2)
            
            reconstruction_lst = {}
        
            for i in range(len(img_embeddings)):
                embed = img_embeddings[i]
                
                embed = torch.reshape(embed, (-1, 9, 4, int(self.image_size / 2), int(self.image_size / 2)))

                self._dec_img_idx = i

                # decode image
                decode_input = {"inputs": embed}
                
                reconstructions = TensorUtils.time_distributed(decode_input, self.decode, inputs_as_kwargs=True)
                
                reconstruction_lst[self.obs_sequence[i]] = reconstructions

        batch_segment = select_batch(batch, 11, 20) # predict 11 images
        
        obs_segment = batch_segment["obs"]

        reconstruction_loss = self.reconstruction_loss(
            reconstructions=reconstruction_lst,
            targets=obs_segment,
        )

        targets = {}
        for k in self.output_shapes:
            targets[k] = obs_segment[k]
        
        if not discard_image:
            returned = {
                #"latent_features": latent_features,
                "total_loss": reconstruction_loss,
                "reconstruction_loss": reconstruction_loss,
                "reconstructions": reconstruction_lst,
                "targets": targets,
            }
        else:
            returned = {
                #"latent_features": latent_features,
                "total_loss": reconstruction_loss,
                "reconstruction_loss": reconstruction_loss
            }
        return returned

    def reconstruction_loss(self, reconstructions, targets):

        num_mods = len(reconstructions.keys())

        recons_errors = []
        for key in reconstructions:
            L2_loss = (reconstructions[key] - targets[key]).pow(2)
            loss = L2_loss
            recons_errors.append(L2_loss)

        # compute mse loss in each modality and average across modalities
        loss = sum([x.mean() for x in recons_errors])
        loss /= num_mods
            
        return loss
    
    def get_curr_latent_features(self, batch):
        
        obs = batch['obs']
                
        feats = []
        for idx in range(len(self.obs_sequence)):
            img_input = obs[self.obs_sequence[idx]]
            
            if len(img_input.shape) == 4:
                img_input = img_input.unsqueeze(0)
                
            assert len(img_input.shape) == 5, "Input shape should be (B, T, C, H, W) or (B, C, H, W), but got {}".format(obs.shape)

            
            self._img_idx = idx
            
            obs_to_embed = {"inputs": img_input}
            
            embed = TensorUtils.time_distributed(obs_to_embed, self.encode, inputs_as_kwargs=True)
            
            embed = torch.flatten(embed, start_dim=2)
            feats.append(embed)
            
        embed = torch.cat(feats, dim=-1)

        return embed
    
    def forward_latent(self, batch):
        
        obs = batch["obs"]
        if "actions" in batch.keys():
            actions = batch["actions"]
        else:
            # breakpoint()
            actions = torch.zeros(list(obs[self.obs_sequence[0]].shape[:-2]) + [self.ac_dim])
            actions = TensorUtils.to_device(TensorUtils.to_float(actions), self.device)
        
        embed = self.get_curr_latent_features(batch)
        
        if self.obs_fusion_method == "concat":
            pass
        elif self.obs_fusion_method == "network_layer":
            
            embed_input = {"inputs": embed}
            
            embed = TensorUtils.time_distributed(embed_input, 
                                         self.nets["fusion_layer"], 
                                         inputs_as_kwargs=True)["embed"]
            
        # dynamics predict
        latent_features = self.nets["dynamics"].predict_future(embed, actions)
        
        return latent_features["embed"]
    
    def imagine(self, batch, _=None):
        return self.forward_latent(batch)


class DynamicsModel_DeterMLP_Unet_VAE(DynamicsModel_DeterMLP):
    """
    Using encoder decoder implementation from diffuser.
    """
    
    def __init__(self,
                 obs_shapes,
                 ac_dim,
                 wm_configs,
                 algo_config,
                 obs_config,
                 goal_shapes,
                 device,
                 global_config=None,
                 ):
        
        # how to combine the image embeddings
        self.obs_fusion_method = wm_configs.obs_fusion_method
        self.deterministic = wm_configs.deterministic
        
        assert self.deterministic is False, "Use VAE version of the model."
        
        super(DynamicsModel_DeterMLP_Unet_VAE, self).__init__(
                 obs_shapes,
                 ac_dim,
                 wm_configs,
                 algo_config,
                 obs_config,
                 goal_shapes,
                 device,
                 global_config,
                 )
    
    
    def create_encoder(self):

        # Hardcoded versions of encoders list

        self.encoder_index = ["encoder_0", "encoder_1", "encoder_2"]
        self.quant_conv_index = ["quant_conv_0", "quant_conv_1", "quant_conv_2"]
        
        self.obs_sequence = self.wm_configs.obs_sequence

        for idx in range(len(self.obs_sequence)):
            
            self.nets[self.encoder_index[idx]] = Encoder(
                in_channels=self.wm_configs.unet.in_channels,
                out_channels=self.wm_configs.unet.latent_channels,
                down_block_types=self.wm_configs.unet.down_block_types,
                block_out_channels=self.wm_configs.unet.block_out_channels,
                layers_per_block=self.wm_configs.unet.layers_per_block,
                act_fn=self.wm_configs.unet.act_fn,
                norm_num_groups=self.wm_configs.unet.norm_num_groups,
                double_z=False,
            )

            self.nets[self.quant_conv_index[idx]] = nn.Conv2d(self.wm_configs.unet.latent_channels, 
                                                              self.wm_configs.unet.latent_channels, 1)
            
        self._set_embedding_dim()

    
    def create_decoder(self):
    
        self.decoder_index = ["decoder_0", "decoder_1", "decoder_2"]
        self.post_quant_conv_index = ["post_quant_conv_0", "post_quant_conv_1", "post_quant_conv_2"]  
    
        for idx in range(len(self.obs_sequence)):
    
            self.nets[self.decoder_index[idx]] = Decoder(
                in_channels=self.wm_configs.unet.latent_channels,
                out_channels=self.wm_configs.unet.out_channels,
                up_block_types=self.wm_configs.unet.up_block_types,
                block_out_channels=self.wm_configs.unet.block_out_channels,
                layers_per_block=self.wm_configs.unet.layers_per_block,
                norm_num_groups=self.wm_configs.unet.norm_num_groups,
                act_fn=self.wm_configs.unet.act_fn,
            )
            
            self.nets[self.post_quant_conv_index[idx]] = nn.Conv2d(self.wm_configs.unet.latent_channels, 
                                                                   self.wm_configs.unet.latent_channels, 1)
        
    
    def _set_embedding_dim(self):

        self.embed_dim = 4 * int(self.image_size / 2) * int(self.image_size / 2)

    def create_dynamics(self):
        
        if len(self.wm_configs.obs_sequence) == 1:
        
            self.nets["dynamics"] = DynNets.DynamicsVAE_MultiTask(
                embed_dim=self.embed_dim,
                device=self.device,
                algo_config=self.algo_config
            )
        
        else:
            
            self.nets["dynamics"] = DynNets.DynamicsVAE_MultiTask_MultiCamera(
                embed_dim=self.embed_dim,
                device=self.device,
                algo_config=self.algo_config
            )
    
    def dynamics_predict(self, embed_dict):
        return self.nets["dynamics"](
            embed_dict
        )
    
    # encode function
    def _encode_image_single(self, inputs, return_dict=True):
        
        img_idx = self._img_idx

        encoded = self.nets[self.encoder_index[img_idx]](inputs)
        encoded_quant_conv = self.nets[self.quant_conv_index[img_idx]](encoded)
        
        return encoded_quant_conv

    # decode function
    def _decode_image_single(self, inputs):      
        
        img_idx = self._dec_img_idx  
            
        inputs = self.nets[self.post_quant_conv_index[img_idx]](
                inputs
            )
        reconstruction = self.nets[self.decoder_index[img_idx]](
            inputs
        )
        reconstruction = nn.Sigmoid()(reconstruction)
        
        return reconstruction
       
    def _forward_training_single_camera(self, batch, discard_image=True):

        obs = batch["obs"]
        actions = batch["actions"]
        
        # encode image

        embed_dict = {}
        for idx in range(len(self.obs_sequence)):
            img_input = obs[self.obs_sequence[idx]]
            
            self._img_idx = idx
            
            obs_to_embed = {"inputs": img_input}
            
            embed = TensorUtils.time_distributed(obs_to_embed, self._encode_image_single, inputs_as_kwargs=True)
            
            embed = torch.flatten(embed, start_dim=2)
            embed_dict[self.obs_sequence[idx]] = embed

        # dynamics predict: process VAE sampling; predict T steps iteratively
        dynamics_predictions = self.dynamics_predict(embed_dict=embed_dict)
        
        latent_features = dynamics_predictions["latent_features"]
        
        B_l, T_l = latent_features.shape[:2]
        latent_features = torch.reshape(latent_features, 
                                        (B_l, T_l, 4, int(self.image_size / 2), int(self.image_size / 2))
                                        )
        
        """ Reconstruction Loss """
        reconstructions = {}
        
        for i in range(len(self.obs_sequence)):
            self._dec_img_idx = i
            name = self.obs_sequence[i]
            reconstruction = TensorUtils.time_distributed({"inputs": latent_features}, 
                                                          self._decode_image_single, 
                                                          inputs_as_kwargs=True)
            reconstructions[name] = reconstruction

        returned_result = self.calculate_reconstruction(batch, reconstructions)
        
        """ Reconstruction Embedding Loss """
        recon_embedding_loss = dynamics_predictions["recon_embedding_loss"].mean()
        returned_result["recon_embedding_loss"] = recon_embedding_loss
        
        """ KL Loss """        
        returned_result["kl_loss"] = dynamics_predictions["kl_loss"].mean()
        returned_result["total_loss"] = returned_result["kl_loss"] * self.algo_config.vae.kl_weight + \
                                        returned_result["reconstruction_loss"]
        
        if discard_image:
            returned_result.pop("reconstructions", None)
            returned_result.pop("targets", None)

        return returned_result
       
    def _forward_training(self, batch, discard_image=True):
        
        if len(self.obs_sequence) == 1:
            # single camera version
            return self._forward_training_single_camera(batch, discard_image=discard_image)
            
        obs = batch["obs"]
        actions = batch["actions"]
        
        # encode image

        embed_dict = OrderedDict()
        for idx in range(len(self.obs_sequence)):
            img_input = obs[self.obs_sequence[idx]]
            
            self._img_idx = idx
            
            obs_to_embed = {"inputs": img_input}
            
            embed = TensorUtils.time_distributed(obs_to_embed, self._encode_image_single, inputs_as_kwargs=True)
            
            embed = torch.flatten(embed, start_dim=2)
            embed_dict[self.obs_sequence[idx]] = embed

        # dynamics predict: process VAE sampling; predict T steps iteratively
        dynamics_predictions = self.dynamics_predict(embed_dict=embed_dict)
        
        latent_features = dynamics_predictions["latent_features"]
        
        random_key = list(batch["obs"].keys())[0]
        T_total = batch["obs"][random_key].shape[1]
        
        if self.wm_configs.recons_full_batch:
            for key in latent_features:
                # concat the last history_length + 1 embed_dict frames with the latent features
                latent_features[key] = torch.cat([embed_dict[key][:,:self.wm_configs.history_length + 1,:], latent_features[key]], dim=1)
                assert latent_features[key].shape[1] == T_total 
        
        random_img_key = self.obs_sequence[0]
        B_l, T_l = latent_features[random_img_key].shape[:2]
        
        for key in latent_features:
            latent_features[key] = torch.reshape(latent_features[key], 
                                            (B_l, T_l, 4, int(self.image_size / 2), int(self.image_size / 2))
                                            )
        
        """ Reconstruction Loss """
        reconstructions = {}
        
        for i in range(len(self.obs_sequence)):
            self._dec_img_idx = i
            name = self.obs_sequence[i]
            reconstruction = TensorUtils.time_distributed({"inputs": latent_features[name]}, 
                                                          self._decode_image_single, 
                                                          inputs_as_kwargs=True)
            reconstructions[name] = reconstruction

        returned_result = self.calculate_reconstruction(batch, reconstructions, full_batch=self.wm_configs.recons_full_batch)
        
        """ KL Loss """        
        returned_result["kl_loss"] = dynamics_predictions["kl_loss"].mean()
        
        """ Reconstruction Embedding Loss """
        recon_embedding_loss = dynamics_predictions["recon_embedding_loss"].mean()
        returned_result["recon_embedding_loss"] = recon_embedding_loss
        
        returned_result["total_loss"] = returned_result["kl_loss"] * self.algo_config.vae.kl_weight + \
                                        returned_result["reconstruction_loss"]
        
        if self.wm_configs.use_embedding_loss:
            returned_result["total_loss"] += returned_result["recon_embedding_loss"] * self.wm_configs.embedding_loss_weight 
        
        if discard_image:
            returned_result.pop("reconstructions", None)
            returned_result.pop("targets", None)

        return returned_result

    def calculate_reconstruction(self, target_batch, reconstructions, full_batch=False):
    
        random_key = list(target_batch["obs"].keys())[0]
        T_total = target_batch["obs"][random_key].shape[1]
        
        if full_batch:
            batch_segment = target_batch
        else:
            batch_segment = select_batch(target_batch, 
                                        self.wm_configs.history_length + 1, 
                                        T_total) 
        
        obs_segment = batch_segment["obs"]

        reconstruction_loss = self.reconstruction_loss(
            reconstructions=reconstructions,
            targets=obs_segment,
        )

        targets = {}
        for k in self.obs_sequence:
            targets[k] = obs_segment[k]
            
        returned_result = {
            "reconstruction_loss": reconstruction_loss,
            "targets": targets,
            "reconstructions": reconstructions
        }
        
        return returned_result
    
    def get_curr_latent_features(self, batch, return_dict=False):
        
        obs = batch['obs']
                
        feats = []
        embed_dict = {}
        for idx in range(len(self.obs_sequence)):
            img_input = obs[self.obs_sequence[idx]]
            
            if len(img_input.shape) == 4:
                img_input = img_input.unsqueeze(0)
                
            assert len(img_input.shape) == 5, "Input shape should be (B, T, C, H, W) or (B, C, H, W), but got {}".format(obs.shape)

            
            self._img_idx = idx
            
            obs_to_embed = {"inputs": img_input}
            
            embed = TensorUtils.time_distributed(obs_to_embed, self._encode_image_single, inputs_as_kwargs=True)
            
            embed = torch.flatten(embed, start_dim=2)
            feats.append(embed)
            embed_dict[self.obs_sequence[idx]] = embed
            
        if return_dict:
            return embed_dict
        else:
            embed = torch.cat(feats, dim=-1)
            return embed
    
    def forward_latent(self, batch, n_future=20, return_flatten=True, return_dict=False):
        
        obs = batch["obs"]
        if "actions" in batch.keys():
            actions = batch["actions"]
        else:
            # breakpoint()
            actions = torch.zeros(list(obs[self.obs_sequence[0]].shape[:-2]) + [self.ac_dim])
            actions = TensorUtils.to_device(TensorUtils.to_float(actions), self.device)
        
        embed_dict = self.get_curr_latent_features(batch, return_dict=True)
        
        dynamics_predictions = self.nets['dynamics'].sample_future(embeds_dict=embed_dict, n_future=n_future)
        
        latent_features = dynamics_predictions["latent_features"]
        
        B_l, T_l = latent_features[self.obs_sequence[0]].shape[:2]
        if return_flatten:
            for key in latent_features:
                latent_features[key] = torch.reshape(latent_features[key], 
                                                (B_l, T_l, 4 * int(self.image_size / 2) * int(self.image_size / 2))
                                                )
        else:
            for key in latent_features:
                latent_features[key] = torch.reshape(latent_features[key],
                                                (B_l, T_l, 4, int(self.image_size / 2), int(self.image_size / 2)))
        if not return_dict:
            feat_list = [feat for feat in latent_features.values()]
            latent_features = torch.cat(feat_list, dim=-1)
                                          
        return latent_features
    
    def imagine(self, batch, n_future=20):
        return self.forward_latent(batch, n_future=n_future)
    
    
class DynamicsModel_DeterMLP_Unet_VAE_EmbedOnly(DynamicsModel_DeterMLP_Unet_VAE):
    """
    DynamicsModel_DeterMLP_Unet_VAE_EmbedOnly: for future prediction, 
    only use the embedding l2 loss, not the image reconstruction loss 
    """
    
    def __init__(self,
                 obs_shapes,
                 ac_dim,
                 wm_configs,
                 algo_config,
                 obs_config,
                 goal_shapes,
                 device,
                 global_config=None,
                 ):
        
        super(DynamicsModel_DeterMLP_Unet_VAE_EmbedOnly, self).__init__(
                 obs_shapes,
                 ac_dim,
                 wm_configs,
                 algo_config,
                 obs_config,
                 goal_shapes,
                 device,
                 global_config,
                 )

    def _generate_latent_futures(self, batch):
        
        obs = batch["obs"]
        actions = batch["actions"]
        
        # encode image
        embed_dict = {}
        for idx in range(len(self.obs_sequence)):
            img_input = obs[self.obs_sequence[idx]]
            self._img_idx = idx
            obs_to_embed = {"inputs": img_input}
            embed = TensorUtils.time_distributed(obs_to_embed, self._encode_image_single, inputs_as_kwargs=True)
            embed_dict[self.obs_sequence[idx]] = embed
        
        """ 2. Dynamics: KL and Reconstruction Embedding """
        embed_shape = embed_dict[self.obs_sequence[0]].shape
        for key in embed_dict:
            embed_dict[key] = torch.flatten(embed_dict[key], start_dim=2)
        # dynamics predict: process VAE sampling; predict T steps iteratively
        dynamics_predictions = self.dynamics_predict(embed_dict=embed_dict)
        
        for key in embed_dict:
            embed_dict[key] = embed_dict[key].reshape(embed_shape)
        return embed_dict, dynamics_predictions


    def _forward_training(self, batch, discard_image=True):
        
        if len(self.obs_sequence) == 1:
            # single camera version
            return self._forward_training_single_camera(batch, discard_image=discard_image)
        
        embed_dict, dynamics_predictions = self._generate_latent_futures(batch)
        
        """ 1. Image Reconstruction Loss """
        reconstructions = {}
        
        for i in range(len(self.obs_sequence)):
            self._dec_img_idx = i
            name = self.obs_sequence[i]
            reconstruction = TensorUtils.time_distributed({"inputs": embed_dict[name]}, 
                                                          self._decode_image_single, 
                                                          inputs_as_kwargs=True)
            reconstructions[name] = reconstruction

        returned_result = self.calculate_reconstruction(batch, reconstructions, full_batch=True) # by default full batch
        
        """    2.1 KL Loss """        
        returned_result["kl_loss"] = dynamics_predictions["kl_loss"].mean()
        
        """    2.2 Reconstruction Embedding Loss """
        recon_embedding_loss = dynamics_predictions["recon_embedding_loss"].mean()
        returned_result["recon_embedding_loss"] = recon_embedding_loss
        
        
        """ Calculate Total Loss """
        returned_result["total_loss"] = returned_result["kl_loss"] * self.algo_config.vae.kl_weight + \
                                        returned_result["reconstruction_loss"]
        if self.wm_configs.use_embedding_loss:
            returned_result["total_loss"] += returned_result["recon_embedding_loss"] * self.wm_configs.embedding_loss_weight 
        
        if discard_image:
            returned_result.pop("reconstructions", None)
            returned_result.pop("targets", None)

        return returned_result
    


def create_dynamics_model_reward_new(parent_class):

    class DynamicsModel_Reward_New(parent_class):
        
        def __init__(self,
                    obs_shapes,
                    ac_dim,
                    wm_configs,
                    algo_config,
                    obs_config,
                    goal_shapes,
                    device,
                    global_config=None,
                    ):
            
            self.version = "new"
            
            super(DynamicsModel_Reward_New, self).__init__(
                    obs_shapes,
                    ac_dim,
                    wm_configs,
                    algo_config,
                    obs_config,
                    goal_shapes,
                    device,
                    global_config,
                    )
            
            self.load_ckpt = self.wm_configs.load_ckpt
            
            model = self._load_model_ckpt()

            self._load_encoder(model)
            self._load_dynamics(model)
            self._load_decoder(model)
            self._load_reward(model)           
            
        def _load_encoder(self, model):
            
            self.encoder_index = ["encoder_0", "encoder_1", "encoder_2"]
            self.quant_conv_index = ["quant_conv_0", "quant_conv_1", "quant_conv_2"] 

            self.obs_sequence = self.wm_configs.obs_sequence
            
            for idx in range(len(self.obs_sequence)):
        
                self.nets[self.encoder_index[idx]] = model.nets["policy"].nets[self.encoder_index[idx]]
                self.nets[self.quant_conv_index[idx]] = model.nets["policy"].nets[self.quant_conv_index[idx]]
        
            self._set_embedding_dim() 
    
        def _load_dynamics(self, model):    
            
            self.nets["dynamics"] = model.nets["policy"].nets["dynamics"]
    
        def _load_decoder(self, model):    
            
            self.decoder_index = ["decoder_0", "decoder_1", "decoder_2"]
            self.post_quant_conv_index = ["post_quant_conv_0", "post_quant_conv_1", "post_quant_conv_2"]  
        
            for idx in range(len(self.obs_sequence)):
        
                self.nets[self.decoder_index[idx]] = model.nets["policy"].nets[self.decoder_index[idx]]
                self.nets[self.post_quant_conv_index[idx]] = model.nets["policy"].nets[self.post_quant_conv_index[idx]]
        
        def _load_reward(self, model):
            
            self.nets["reward"] = DynNets.RewardTransformer(
                    self.embed_dim, 
                    self.ac_dim, 
                    algo_configs=self.algo_config,
                    wm_configs=self.wm_configs
                    )

        def _load_model_ckpt(self):
            dir_name = os.path.dirname(os.path.dirname(self.load_ckpt)) 
            load_config = os.path.join(dir_name, "config.json")
            ext_cfg = json.load(open(load_config, 'r'))
            
            config = config_factory(ext_cfg["algo_name"])
            with config.values_unlocked():
                config.update(ext_cfg)
            self.load_config = config
            
            model = algo_factory(
                algo_name=self.global_config.algo_name,
                config=self.load_config,
                obs_key_shapes=self.obs_shapes,
                ac_dim=self.ac_dim,
                device=self.device,
            )

            print('Loading model weights from:', self.load_ckpt)

            from robomimic.utils.file_utils import maybe_dict_from_checkpoint
            ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=self.load_ckpt)
            model.deserialize(ckpt_dict["model"])
            
            return model
        
        def reward_predict(self, embed, actions):
            return self.nets["reward"](embed, actions)
        
        def _forward_training(self, batch, discard_image=True):
            
            curr_embed, latent_features = self._generate_latent_futures(batch)
            
            if self.wm_configs.rew.latent_stop_grad:
                curr_embed = TensorUtils.detach(curr_embed)
                latent_features = TensorUtils.detach(latent_features)
            
            for key in latent_features:
                assert latent_features[key].shape[1] == 9 

            pred_reward = self.reward_predict(embed=latent_features["embed"],
                                              actions=batch["actions"][:,11:,:]
                                              ) 
            
            reward_labels = self._create_reward_label(batch)
            
            pred_reward_for_loss = torch.permute(pred_reward, (0,2,1))
            reward_labels_for_loss = reward_labels
            
            reward_loss_func = nn.CrossEntropyLoss(reduction="none")
            reward_loss = reward_loss_func(pred_reward_for_loss, reward_labels_for_loss).mean()

            pred_classes = torch.argmax(pred_reward, dim=-1)
            overal_acc = (pred_classes == reward_labels).float().mean()
            
            with np.errstate(divide='ignore', invalid='ignore'):
                class_acc, matrix = self._confusion_matrix(y_true=reward_labels, y_pred=pred_classes, num_classes=3)
            
            predictions = OrderedDict()
            predictions["total_loss"] = reward_loss
            predictions["reward_loss"] = reward_loss        
            predictions["reward_overall_acc"] = overal_acc
            predictions["reward_rollout_acc"] = class_acc[0] 
            predictions["reward_intv_acc"] = class_acc[1] 
            predictions["reward_preintv_acc"] = class_acc[2]
            predictions["confusion_matrix"] = matrix
            
            self._calculate_class_samples(predictions=predictions, 
                                          y_true=reward_labels, 
                                          num_classes=3)

            return predictions
        
        def _calculate_class_samples(self, predictions, y_true, num_classes):
            
            results = {}
            for i in range(num_classes):
                results[f"class_{i}_samples"] = (y_true == i).sum() / y_true.numel()
                
            predictions["class_samples"] = results

        def _create_reward_label(self, batch):
            
            batch_segment = select_batch(batch, 11, 20) # predict 9 images
            intv_labels = batch_segment["intv_labels"]
            
            x = intv_labels
            x = torch.where((x == -1) | (x == 0), torch.tensor(0), x)
            x = torch.where(x == -10, torch.tensor(2), x)
            x = x.long()
            return x
    
        def _confusion_matrix(self, y_true, y_pred, num_classes):
            conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
            for t, p in zip(y_true.view(-1), y_pred.view(-1)):
                conf_matrix[t.long(), p.long()] += 1
            conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
            conf_matrix_no_nan = torch.where(torch.isnan(conf_matrix_norm), torch.zeros_like(conf_matrix_norm), conf_matrix_norm)
            return torch.diag(conf_matrix_no_nan), conf_matrix_no_nan
        
    return DynamicsModel_Reward_New

DynamicsModelReward_DeterMLP_Unet_New = create_dynamics_model_reward_new(DynamicsModel_DeterMLP_Unet)


class DynamicsModel_DeterMLP_Unet_Reward(DynamicsModel_DeterMLP_Unet):
    
    def __init__(self,
                obs_shapes,
                ac_dim,
                wm_configs,
                algo_config,
                obs_config,
                goal_shapes,
                device,
                global_config=None,
                ):
        
        self.version = "new"
        
        super(DynamicsModel_DeterMLP_Unet_Reward, self).__init__(
                obs_shapes,
                ac_dim,
                wm_configs,
                algo_config,
                obs_config,
                goal_shapes,
                device,
                global_config,
                )
        
        self.load_ckpt = self.wm_configs.load_ckpt
        
        model = self._load_model_ckpt()

        self._load_encoder(model)
        self._load_dynamics(model)
        self._load_decoder(model)
        self._load_reward(model)           
        
    def _load_encoder(self, model):
        
        self.encoder_index = ["encoder_0", "encoder_1", "encoder_2"]
        self.quant_conv_index = ["quant_conv_0", "quant_conv_1", "quant_conv_2"] 

        self.obs_sequence = self.wm_configs.obs_sequence
        
        for idx in range(len(self.obs_sequence)):
    
            self.nets[self.encoder_index[idx]] = model.nets["policy"].nets[self.encoder_index[idx]]
            self.nets[self.quant_conv_index[idx]] = model.nets["policy"].nets[self.quant_conv_index[idx]]
    
        self._set_embedding_dim() 

    def _load_dynamics(self, model):    
        
        self.nets["dynamics"] = model.nets["policy"].nets["dynamics"]

    def _load_decoder(self, model):    
        
        self.decoder_index = ["decoder_0", "decoder_1", "decoder_2"]
        self.post_quant_conv_index = ["post_quant_conv_0", "post_quant_conv_1", "post_quant_conv_2"]  
    
        for idx in range(len(self.obs_sequence)):
    
            self.nets[self.decoder_index[idx]] = model.nets["policy"].nets[self.decoder_index[idx]]
            self.nets[self.post_quant_conv_index[idx]] = model.nets["policy"].nets[self.post_quant_conv_index[idx]]
    
    def _load_reward(self, model):
        
        self.nets["reward"] = DynNets.RewardTransformer(
                self.embed_dim, 
                self.ac_dim, 
                algo_configs=self.algo_config,
                wm_configs=self.wm_configs
                )

    def _load_model_ckpt(self):
        dir_name = os.path.dirname(os.path.dirname(self.load_ckpt)) 
        load_config = os.path.join(dir_name, "config.json")
        ext_cfg = json.load(open(load_config, 'r'))
        
        config = config_factory(ext_cfg["algo_name"])
        with config.values_unlocked():
            config.update(ext_cfg)
        self.load_config = config
        
        model = algo_factory(
            algo_name=self.global_config.algo_name,
            config=self.load_config,
            obs_key_shapes=self.obs_shapes,
            ac_dim=self.ac_dim,
            device=self.device,
        )

        print('Loading model weights from:', self.load_ckpt)

        from robomimic.utils.file_utils import maybe_dict_from_checkpoint
        ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=self.load_ckpt)
        model.deserialize(ckpt_dict["model"])
        
        return model
    
    def reward_predict(self, embed, actions):
        return self.nets["reward"](embed, actions)
    
    def _forward_training(self, batch, discard_image=True):
        
        curr_embed, latent_features = self._generate_latent_futures(batch)
        
        if self.wm_configs.rew.latent_stop_grad:
            curr_embed = TensorUtils.detach(curr_embed)
            latent_features = TensorUtils.detach(latent_features)
        
        for key in latent_features:
            assert latent_features[key].shape[1] == 9

        pred_reward = self.reward_predict(embed=latent_features["embed"],
                                            actions=batch["actions"][:,11:,:]
                                            ) 
        
        reward_labels = self._create_reward_label(batch)
        
        pred_reward_for_loss = torch.permute(pred_reward, (0,2,1))
        reward_labels_for_loss = reward_labels
        
        reward_loss_func = nn.CrossEntropyLoss(reduction="none")
        reward_loss = reward_loss_func(pred_reward_for_loss, reward_labels_for_loss).mean()

        pred_classes = torch.argmax(pred_reward, dim=-1)
        overal_acc = (pred_classes == reward_labels).float().mean()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            class_acc, matrix = self._confusion_matrix(y_true=reward_labels, y_pred=pred_classes, num_classes=3)
        
        predictions = OrderedDict()
        predictions["total_loss"] = reward_loss
        predictions["reward_loss"] = reward_loss        
        predictions["reward_overall_acc"] = overal_acc
        predictions["reward_rollout_acc"] = class_acc[0] 
        predictions["reward_intv_acc"] = class_acc[1] 
        predictions["reward_preintv_acc"] = class_acc[2]
        predictions["confusion_matrix"] = matrix
        
        self._calculate_class_samples(predictions=predictions, 
                                        y_true=reward_labels, 
                                        num_classes=3)

        return predictions
    
    def _calculate_class_samples(self, predictions, y_true, num_classes):
        
        results = {}
        for i in range(num_classes):
            results[f"class_{i}_samples"] = (y_true == i).sum() / y_true.numel()
            
        predictions["class_samples"] = results

    def _create_reward_label(self, batch):
        
        batch_segment = select_batch(batch, 11, 20) # predict 9 images
        intv_labels = batch_segment["intv_labels"]
        
        x = intv_labels
        x = torch.where((x == -1) | (x == 0), torch.tensor(0), x)
        x = torch.where(x == -10, torch.tensor(2), x)
        x = x.long()
        return x

    def _confusion_matrix(self, y_true, y_pred, num_classes):
        conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        for t, p in zip(y_true.view(-1), y_pred.view(-1)):
            conf_matrix[t.long(), p.long()] += 1
        conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
        conf_matrix_no_nan = torch.where(torch.isnan(conf_matrix_norm), torch.zeros_like(conf_matrix_norm), conf_matrix_norm)
        return torch.diag(conf_matrix_no_nan), conf_matrix_no_nan
    
    
class DynamicsModel_DeterMLP_Unet_VAE_EmbedOnly_Reward(DynamicsModel_DeterMLP_Unet_VAE_EmbedOnly):
    
    def __init__(self,
                obs_shapes,
                ac_dim,
                wm_configs,
                algo_config,
                obs_config,
                goal_shapes,
                device,
                global_config=None,
                ):
        
        self.version = "new"
        
        super(DynamicsModel_DeterMLP_Unet_VAE_EmbedOnly_Reward, self).__init__(
                obs_shapes,
                ac_dim,
                wm_configs,
                algo_config,
                obs_config,
                goal_shapes,
                device,
                global_config,
                )
        
        self.load_ckpt = self.wm_configs.load_ckpt
        
        model = self._load_model_ckpt()

        self._load_encoder(model)
        self._load_dynamics(model)
        self._load_decoder(model)
        self._load_reward(model)           
        
    def _load_encoder(self, model):
        
        self.encoder_index = ["encoder_0", "encoder_1", "encoder_2"]
        self.quant_conv_index = ["quant_conv_0", "quant_conv_1", "quant_conv_2"] 

        self.obs_sequence = self.wm_configs.obs_sequence
        
        for idx in range(len(self.obs_sequence)):
    
            self.nets[self.encoder_index[idx]] = model.nets["policy"].nets[self.encoder_index[idx]]
            self.nets[self.quant_conv_index[idx]] = model.nets["policy"].nets[self.quant_conv_index[idx]]
    
        self._set_embedding_dim() 

    def _load_dynamics(self, model):    
        
        self.nets["dynamics"] = model.nets["policy"].nets["dynamics"]

    def _load_decoder(self, model):    
        
        self.decoder_index = ["decoder_0", "decoder_1", "decoder_2"]
        self.post_quant_conv_index = ["post_quant_conv_0", "post_quant_conv_1", "post_quant_conv_2"]  
    
        for idx in range(len(self.obs_sequence)):
    
            self.nets[self.decoder_index[idx]] = model.nets["policy"].nets[self.decoder_index[idx]]
            self.nets[self.post_quant_conv_index[idx]] = model.nets["policy"].nets[self.post_quant_conv_index[idx]]
    
    def _load_reward(self, model):
        
        self.nets["reward"] = DynNets.RewardTransformer(
                self.embed_dim, 
                self.ac_dim, 
                algo_configs=self.algo_config,
                wm_configs=self.wm_configs
                )

    def _set_embedding_dim(self):

        self.embed_dim = 4 * int(self.image_size / 2) * int(self.image_size / 2) * len(self.obs_sequence)

    def _load_model_ckpt(self):
        dir_name = os.path.dirname(os.path.dirname(self.load_ckpt)) 
        load_config = os.path.join(dir_name, "config.json")
        ext_cfg = json.load(open(load_config, 'r'))
        
        config = config_factory(ext_cfg["algo_name"])
        with config.values_unlocked():
            config.update(ext_cfg)
        self.load_config = config
        
        model = algo_factory(
            algo_name=self.global_config.algo_name,
            config=self.load_config,
            obs_key_shapes=self.obs_shapes,
            ac_dim=self.ac_dim,
            device=self.device,
        )

        print('Loading model weights from:', self.load_ckpt)

        from robomimic.utils.file_utils import maybe_dict_from_checkpoint
        ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=self.load_ckpt)
        model.deserialize(ckpt_dict["model"])
        
        return model
    
    def reward_predict(self, embed, actions):
        return self.nets["reward"](embed, actions)
    
    def _forward_training(self, batch, discard_image=True):
        
        curr_embed, latent_features = self._generate_latent_futures(batch)
        
        if self.wm_configs.rew.latent_stop_grad:
            curr_embed = TensorUtils.detach(curr_embed)
            latent_features = TensorUtils.detach(latent_features)
        
        latent_features = latent_features["latent_features"]
        
        for key in latent_features:
            assert latent_features[key].shape[1] == 9 # hack for now

        latent_feature_embed_concat = torch.cat([latent_features[key] for key in self.obs_sequence], dim=-1)

        pred_reward = self.reward_predict(embed=latent_feature_embed_concat,
                                            actions=batch["actions"][:,11:,:]
                                            ) # predict 9 rewards
        
        reward_labels = self._create_reward_label(batch)
        
        pred_reward_for_loss = torch.permute(pred_reward, (0,2,1))
        reward_labels_for_loss = reward_labels
        
        reward_loss_func = nn.CrossEntropyLoss(reduction="none")
        reward_loss = reward_loss_func(pred_reward_for_loss, reward_labels_for_loss).mean()

        pred_classes = torch.argmax(pred_reward, dim=-1)
        overal_acc = (pred_classes == reward_labels).float().mean()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            class_acc, matrix = self._confusion_matrix(y_true=reward_labels, y_pred=pred_classes, num_classes=3)
        
        predictions = OrderedDict()
        predictions["total_loss"] = reward_loss
        predictions["reward_loss"] = reward_loss        
        predictions["reward_overall_acc"] = overal_acc
        predictions["reward_rollout_acc"] = class_acc[0] 
        predictions["reward_intv_acc"] = class_acc[1] 
        predictions["reward_preintv_acc"] = class_acc[2]
        predictions["confusion_matrix"] = matrix
        
        self._calculate_class_samples(predictions=predictions, 
                                        y_true=reward_labels, 
                                        num_classes=3)

        return predictions
    
    def _calculate_class_samples(self, predictions, y_true, num_classes):
        
        results = {}
        for i in range(num_classes):
            results[f"class_{i}_samples"] = (y_true == i).sum() / y_true.numel()
            
        predictions["class_samples"] = results

    def _create_reward_label(self, batch):
        
        batch_segment = select_batch(batch, 11, 20) # predict 9 images
        intv_labels = batch_segment["intv_labels"]
        
        x = intv_labels
        x = torch.where((x == -1) | (x == 0), torch.tensor(0), x)
        x = torch.where(x == -10, torch.tensor(2), x)
        x = x.long()
        return x

    def _confusion_matrix(self, y_true, y_pred, num_classes):
        conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        for t, p in zip(y_true.view(-1), y_pred.view(-1)):
            conf_matrix[t.long(), p.long()] += 1
        conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
        conf_matrix_no_nan = torch.where(torch.isnan(conf_matrix_norm), torch.zeros_like(conf_matrix_norm), conf_matrix_norm)
        return torch.diag(conf_matrix_no_nan), conf_matrix_no_nan