import textwrap
import numpy as np
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.models.base_nets import Module, MLP, ResidualMLP
from robomimic.models.obs_nets import MIMO_MLP, MIMO_Transformer_Dyn
from robomimic.models.vae_nets import *
import robomimic.models.vae_nets as VAENets
import robomimic.models.base_nets as BaseNets
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import res_mlp_args_from_config_vae


class Dynamics(nn.Module):

    def __init__(self, embed_dim, action_dim, wm_configs, algo_configs):
        super().__init__()
        
        # self.use_history = wm_configs.use_history
        # self.use_real = wm_configs.use_real
        try:
            no_action = wm_configs.no_action
        except:
            no_action = False
        self.cell = DynamicsCell(embed_dim, 
                                 action_dim,
                                 no_action=no_action,
                                 algo_configs=algo_configs
                                 )

    def forward(self,
                embeds,              # tensor(B, T, E)
                actions,             # tensor(B, T, A)
                ):
        
        B, T = embeds.shape[:2]

        pred_embeds_lst = []
        
        context = T//2

        for i in range(T//2 - 1):

            if i == 0:
                curr_embed = embeds[:,:context]
            else:
                curr_embed = torch.cat([embeds[:,i:context], pred_embeds], dim=1)

            assert curr_embed.shape[1] == T // 2 # (B, T, E)
            assert curr_embed.shape[0] == B

            action = actions[:,i:i+context]

            next_embed_pred = self.cell.forward(
                embed=curr_embed, 
                action=action,
            )

            pred_embeds_lst.append(next_embed_pred)
            pred_embeds = torch.cat(pred_embeds_lst, dim=1) 

            assert pred_embeds.shape[1] == i + 1
        
        return OrderedDict(
            embed=pred_embeds,
        )
    
    @torch.no_grad()
    def predict_future(self, 
                       embeds, 
                       actions, 
                       n_future=20,
                       ):
        
        B, T = embeds.shape[:2]

        pred_embeds_lst = []
        context = T # In evaluation, we have no future states

        for i in range(T - 1):

            if i == 0:
                curr_embed = embeds[:,-context:]
            else:
                curr_embed = torch.cat([embeds, pred_embeds], dim=1)[:,-context:]

            assert curr_embed.shape[1] == T # (B, T, E)
            assert curr_embed.shape[0] == B

            action = actions[:,i:i+context]

            next_embed_pred = self.cell.forward(
                embed=curr_embed, 
                action=action,
            )

            pred_embeds_lst.append(next_embed_pred)
            pred_embeds = torch.cat(pred_embeds_lst, dim=1) 

            assert pred_embeds.shape[1] == i + 1
        
        return OrderedDict(
            embed=pred_embeds,
        )

class DynamicsCell(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 action_dim, 
                 no_action=False, # do not use action in prediction
                 algo_configs=None,
                 hidden_dim=1024, 
                 action_network_dim=0, 
                 num_layers=2, 
                ):
        super().__init__()
        
        self.action_network_dim = action_network_dim
        self.algo_configs = algo_configs
        self.no_action = no_action
        
        # Do not use action, just use past state embedding to do autoregressive prediction.
        if no_action:
            
            input_dim = embed_dim
            
        else:
            
            if action_network_dim > 0:
                self._action_network = MLP(
                    input_dim=action_dim,
                    output_dim=action_network_dim,
                    layer_dims=[hidden_dim],
                    activation=nn.ELU,
                    output_activation=None,
                    normalization=True,
                )
                input_dim = embed_dim + action_network_dim
            else:
                input_dim = embed_dim + action_dim
        
        embedding_output_shapes = OrderedDict(
            embed=(embed_dim,)
        )
        
        algo_configs_transformer = deepcopy(self.algo_configs.transformer)
        algo_configs_transformer.unlock()
        algo_configs_transformer.context_length = 10
    
        self._mlp = MIMO_Transformer_Dyn(
            input_dim=input_dim,
            output_shapes=embedding_output_shapes,
            **BaseNets.transformer_args_from_config(algo_configs_transformer),
            )

    def forward(self, embed, action):
        
        if self.no_action:

            pred = embed    
        
        else:
                       
            if self.action_network_dim > 0:
                action = self._action_network(action)
            pred = torch.cat([embed, action], dim=-1)
        
        pred = self._mlp(pred)["embed"][:, -1:, :]

        return pred 


########################################################################################

class DynamicsVAE_MultiTask(nn.Module):

    def __init__(self, embed_dim, device, algo_config):
        super().__init__()

        self.algo_config = algo_config

        self.input_shapes = OrderedDict()
        self.input_shapes["next_embed"] = (embed_dim,)

        algo_configs_transformer = deepcopy(self.algo_config.transformer)
        algo_configs_transformer.unlock()
        algo_configs_transformer.context_length = 10

        self.cell = DynamicsVAE_MultiTask_Cell(
                        embed_dim=embed_dim,
                        history_length=algo_config.dyn.history_length,
                        output_shapes=self.input_shapes,
                        device=device,
                        **VAENets.vae_args_from_config_dyn(algo_config.vae),
                        encoder_kwargs=None,
                        algo_configs_transformer=algo_configs_transformer,
                    )

    def forward(self,
                embeds_dict,              # tensor(T, B, E)
                ):
        
        if len(list(embeds_dict.keys())) == 1:
            embeds = embeds_dict[list(embeds_dict.keys())[0]]
        else:
            assert NotImplementedError
            
        B, T = embeds.shape[:2]

        pred_embeds_lst = []
        kl_loss = []
        reconstruction_loss = []
        
        context = T // 2 - 1
        
        for i in range(T//2 - 1):

            if i == 0:
                curr_embed = embeds[:,:context]
            else:
                curr_embed = torch.cat([embeds[:,i:context], pred_embeds], dim=1)

            assert curr_embed.shape[1] == T // 2 - 1 # (B, T, E)
            assert curr_embed.shape[0] == B

            next_embed = embeds[:,T // 2 + i]

            vae_outputs = self.cell.forward(inputs=next_embed, 
                                            outputs={"next_embed": next_embed}, 
                                            conditions=curr_embed)

            next_embed_pred = vae_outputs["decoder_outputs"]['next_embed']
            kl_loss.append(vae_outputs["kl_loss"])
            reconstruction_loss.append(vae_outputs["reconstruction_loss"])

            next_embed_pred = next_embed_pred.unsqueeze(1)
            pred_embeds_lst.append(next_embed_pred)
            pred_embeds = torch.cat(pred_embeds_lst, dim=1) 

            assert pred_embeds.shape[1] == i + 1
        
        kl_loss = torch.stack(kl_loss) 
        reconstruction_loss = torch.stack(reconstruction_loss)

        return OrderedDict(
            recon_embedding_loss=reconstruction_loss,
            kl_loss=kl_loss,
            latent_features=pred_embeds,
        )
    
    @torch.no_grad()
    def sample_future(self, 
                      embeds_dict,
                      n_future=20,
                      ):
        if len(list(embeds_dict.keys())) == 1:
            embeds = embeds_dict[list(embeds_dict.keys())[0]]
        else:
            assert NotImplementedError
            
        B, T = embeds.shape[:2]

        if B != 1:
            assert NotImplementedError

        pred_embeds_lst = []

        context = T - 1 # In evaluation, we have no future states

        embeds = torch.tile(embeds, (n_future, 1, 1))
        
        for i in range(T - 1):

            if i == 0:
                curr_embed = embeds[:,-context:]
            else:
                curr_embed = torch.cat([embeds[:, :], pred_embeds], dim=1)[:,-context:]

            assert curr_embed.shape[1] == T - 1 # (B, T, E)
            assert curr_embed.shape[0] == B * n_future

            next_embed_pred = self.cell.decode(conditions=curr_embed, z=None, n=n_future)['next_embed'][..., -1, :]

            next_embed_pred = next_embed_pred.unsqueeze(1)
            pred_embeds_lst.append(next_embed_pred)
            pred_embeds = torch.cat(pred_embeds_lst, dim=1) 

            assert pred_embeds.shape[1] == i + 1

        return OrderedDict(
            latent_features=pred_embeds,
        )
        


class DynamicsVAE_MultiTask_Cell(torch.nn.Module):
    """ 
    Input: image embeddings (not images)
    Output: next image embedding
    """
    def __init__(
        self,
        embed_dim,
        history_length,
        output_shapes,
        latent_dim,
        device,
        condition_shapes=None,
        decoder_is_conditioned=True,
        decoder_reconstruction_sum_across_elements=False,
        latent_clip=None,
        output_squash=(),
        output_scales=None,
        output_ranges=None,
        prior_learn=False,
        prior_is_conditioned=False,
        prior_layer_dims=(),
        prior_use_gmm=False,
        prior_gmm_num_modes=10,
        prior_gmm_learn_weights=False,
        encoder_kwargs=None,
        algo_configs_transformer=None,
    ):
        
        super(DynamicsVAE_MultiTask_Cell, self).__init__()

        self.embed_dim = embed_dim
        self.history_length = history_length

        self.latent_dim = latent_dim
        self.latent_clip = latent_clip
        self.device = device

        self.algo_configs_transformer = algo_configs_transformer

        # encoder and decoder input dicts and output shapes dict for reconstruction
        assert isinstance(output_shapes, OrderedDict)
        self.output_shapes = deepcopy(output_shapes)

        # cVAE configs
        self._is_cvae = True
        self.decoder_is_conditioned = decoder_is_conditioned
        self.prior_is_conditioned = prior_is_conditioned
        assert self.decoder_is_conditioned or self.prior_is_conditioned, \
            "cVAE must be conditioned in decoder and/or prior"
        if self.prior_is_conditioned:
            assert prior_learn, "to pass conditioning inputs to prior, prior must be learned"

        self.condition_shapes = deepcopy(condition_shapes) if condition_shapes is not None else OrderedDict()

        # determines whether outputs are squashed with tanh and if so, to what scaling
        assert not (output_scales is not None and output_ranges is not None)
        self.output_squash = output_squash
        self.output_scales = output_scales if output_scales is not None else OrderedDict()
        self.output_ranges = output_ranges if output_ranges is not None else OrderedDict()

        assert set(self.output_squash) == set(self.output_scales.keys())
        assert set(self.output_squash).issubset(set(self.output_shapes))

        # decoder settings
        self.decoder_reconstruction_sum_across_elements = decoder_reconstruction_sum_across_elements

        self._encoder_kwargs = encoder_kwargs

        # prior parameters
        self.prior_learn = prior_learn
        self.prior_layer_dims = prior_layer_dims
        self.prior_use_gmm = prior_use_gmm
        self.prior_gmm_num_modes = prior_gmm_num_modes
        self.prior_gmm_learn_weights = prior_gmm_learn_weights

        if self.prior_use_gmm:
            assert self.prior_learn, "GMM must be learned"

        # create encoder, decoder, prior
        self._create_layers()

    def _create_layers(self):
        """
        Creates the encoder, decoder, and prior networks.
        """
        self.nets = nn.ModuleDict()

        # VAE Encoder
        self._create_encoder()

        # VAE Decoder
        self._create_decoder()

        # VAE Prior.
        self._create_prior()

    def _create_encoder(self):
        """
        Helper function to create encoder.
        """
        # input_dim: concat history and current embedding
        input_dim = self.embed_dim #* (self.history_length + 1)
        
        # encoder outputs posterior distribution parameters
        encoder_output_shapes = OrderedDict(
            mean=(self.latent_dim,), 
            logvar=(self.latent_dim,),
        )

        self.nets["encoder"] = MIMO_Transformer_Dyn(
            input_dim=input_dim,
            output_shapes=encoder_output_shapes,
            **BaseNets.transformer_args_from_config(self.algo_configs_transformer),
            )

    def _create_decoder(self):
        """
        Helper function to create decoder.
        """

        # inputs: latent + history condition
        input_dim = self.latent_dim + self.embed_dim #* (self.history_length)

        algo_configs_transformer = deepcopy(self.algo_configs_transformer)
        algo_configs_transformer.unlock()
        algo_configs_transformer.context_length = algo_configs_transformer.context_length - 1 

        self.nets["decoder"] = MIMO_Transformer_Dyn(
            input_dim=input_dim,
            output_shapes=self.output_shapes,
            **BaseNets.transformer_args_from_config(algo_configs_transformer),
            )

    def _create_prior(self):

        # prior possibly takes "condition" (if cVAE) and "goal" (if goal-conditioned)
        prior_obs_group_shapes = OrderedDict(condition=None, goal=None)
        prior_obs_group_shapes["condition"] = OrderedDict(self.condition_shapes)

        self.nets["prior"] = GaussianPrior(
            latent_dim=self.latent_dim,
            device=self.device,
            latent_clip=self.latent_clip,
            learnable=self.prior_learn,
            use_gmm=self.prior_use_gmm,
            gmm_num_modes=self.prior_gmm_num_modes,
            gmm_learn_weights=self.prior_gmm_learn_weights,
            obs_shapes=prior_obs_group_shapes["condition"],
            mlp_layer_dims=self.prior_layer_dims,
            goal_shapes=prior_obs_group_shapes["goal"],
            encoder_kwargs=self._encoder_kwargs,
        )

    def encode(self, inputs, conditions=None, goals=None):

        assert conditions is not None
    
        inputs = inputs[:,None,:]
        inputs = torch.cat((inputs, conditions), dim=-2)

        return self.nets["encoder"](inputs)

    def reparameterize(self, posterior_params):

        return TorchUtils.reparameterize(
            mu=posterior_params["mean"], 
            logvar=posterior_params["logvar"],
        )

    def decode(self, conditions=None, goals=None, z=None, n=None):

        if z is None:
            # sample latents from prior distribution
            assert n is not None
            z = self.sample_prior(n=n, conditions=conditions, goals=goals)
            
        assert conditions is not None
        
        B, T, _ = conditions.shape
        B_z, E_z = z.shape
        assert B == B_z
        
        z = z.unsqueeze(1).expand(B, T, E_z)
        
        inputs = torch.cat((z, conditions), dim=-1)

        # pass through decoder to reconstruct variables in @self.output_shapes
        recons = self.nets["decoder"](inputs)

        # apply tanh squashing to output modalities
        for k in self.output_squash:
            recons[k] = self.output_scales[k] * torch.tanh(recons[k])

        for k, v_range in self.output_ranges.items():
            assert v_range[1] > v_range[0]
            recons[k] = torch.sigmoid(recons[k]) * (v_range[1] - v_range[0]) + v_range[0]
        return recons

    def sample_prior(self, n, conditions=None, goals=None):

        return self.nets["prior"].sample(n=n, obs_dict=conditions, goal_dict=goals)

    def kl_loss(self, posterior_params, encoder_z=None, conditions=None, goals=None):
        
        return self.nets["prior"].kl_loss(
            posterior_params=posterior_params,
            z=encoder_z,
            obs_dict=conditions, 
            goal_dict=goals,
        )

    def reconstruction_loss(self, reconstructions, targets):
        """
        Reconstruction loss. Note that we compute the average per-dimension error
        in each modality and then average across all the modalities.

        The beta term for weighting between reconstruction and kl losses will
        need to be tuned in practice for each situation (see
        https://twitter.com/memotv/status/973323454350090240 for more 
        discussion).

        Args:
            reconstructions (dict): reconstructed inputs, consistent with
                @self.output_shapes
            targets (dict): reconstruction targets, consistent with
                @self.output_shapes

        Returns:
            reconstruction_loss (torch.Tensor): VAE reconstruction loss
        """
        random_key = list(reconstructions.keys())[0]
        batch_size = reconstructions[random_key].shape[0]
        num_mods = len(reconstructions.keys())

        # collect errors per modality, while preserving shapes in @reconstructions
        recons_errors = []
        for k in reconstructions:
            L2_loss = (reconstructions[k] - targets[k]).pow(2)
            recons_errors.append(L2_loss)

        # reduce errors across modalities and dimensions
        if self.decoder_reconstruction_sum_across_elements:
            # average across batch but sum across modalities and dimensions
            loss = sum([x.sum() for x in recons_errors])
            loss /= batch_size
        else:
            # compute mse loss in each modality and average across modalities
            loss = sum([x.mean() for x in recons_errors])
            loss /= num_mods
        return loss

    def forward(self, inputs, outputs, conditions=None, goals=None, freeze_encoder=False):

        # In the comments below, X = inputs, Y = conditions, and we seek to learn P(X | Y).
        # The decoder and prior only have knowledge about Y and try to reconstruct X.
        # Notice that when Y is the empty set, this reduces to a normal VAE.

        # mu, logvar <- Enc(X, Y)
        posterior_params = self.encode(
            inputs=inputs, 
            conditions=conditions,
            goals=goals,
        )
        
        for key in posterior_params:
            posterior_params[key] = posterior_params[key][:, -1, :]

        if freeze_encoder:
            posterior_params = TensorUtils.detach(posterior_params)

        # z ~ Enc(z | X, Y)
        encoder_z = self.reparameterize(posterior_params)

        # hat(X) = Dec(z, Y)
        reconstructions = self.decode(
            conditions=conditions, 
            goals=goals,
            z=encoder_z,
        )   
        
        reconstructions.pop("transformer_encoder_outputs", None)
             
        for key in reconstructions:
            # print(key)
            reconstructions[key] = reconstructions[key][:, -1, :]
        
        
        # this will also train prior network z ~ Prior(z | Y)
        kl_loss = self.kl_loss(
            posterior_params=posterior_params,
            encoder_z=encoder_z,
            conditions=conditions,
            goals=goals,
        )

        # just calculate first; might not use later.
        reconstruction_loss = self.reconstruction_loss(
            reconstructions=reconstructions, 
            targets=outputs,
        )

        return {
            "encoder_params" : posterior_params,
            "encoder_z" : encoder_z,
            "decoder_outputs" : reconstructions,
            "kl_loss" : kl_loss,
            "reconstruction_loss" : reconstruction_loss,
            "reconstructions": reconstructions,
            "targets": outputs,
        }


###################################################



class DynamicsVAE_MultiTask_MultiCamera(torch.nn.Module):

    def __init__(self, embed_dim, device, algo_config):
        super().__init__()

        self.algo_config = algo_config

        self.input_shapes = OrderedDict()
        
        obs_keys = self.algo_config.dyn.obs_sequence
        for key in obs_keys:
            self.input_shapes[key] = (embed_dim,)

        algo_configs_transformer = deepcopy(self.algo_config.transformer)
        algo_configs_transformer.unlock()
        algo_configs_transformer.context_length = 10

        condition_shapes = OrderedDict()
        for key in self.algo_config.dyn.obs_sequence:
            condition_shapes[key] = (embed_dim,)
        try:
            cell_type = self.algo_config.dyn.dyn_cell_type
        except:
            cell_type = "default"
        
        assert cell_type in ["no_fusion", "default"]
        
        if cell_type == "no_fusion":
            cell_class = DynamicsVAE_MultiTask_Cell_MultiCamera_NoFusion
        else:
            cell_class = DynamicsVAE_MultiTask_Cell_MultiCamera
            
        self.cell_type = cell_type

        self.cell = cell_class(
                        embed_dim=embed_dim,
                        history_length=algo_config.dyn.history_length,
                        obs_keys=algo_config.dyn.obs_sequence,
                        output_shapes=self.input_shapes,
                        device=device,
                        **VAENets.vae_args_from_config_dyn(algo_config.vae),
                        encoder_kwargs=None,
                        algo_configs_transformer=algo_configs_transformer,
                    )


    def forward(self, embeds_dict):
        
        B, T, _ = next(iter(embeds_dict.values())).shape
        
        context = T // 2 - 1
        
        pred_embeds_dict = {key: [] for key in embeds_dict}
        kl_loss = []
        reconstruction_loss = []

        for i in range(T // 2 - 1):
            if i == 0:
                curr_embed = {key: embeds[:, :context, :] for key, embeds in embeds_dict.items()}
            else:
                curr_embed = {key: torch.cat([embeds_dict[key][:, i:context, :], torch.cat(pred_embeds_dict[key], dim=1)], dim=1)
                            for key in embeds_dict}
            
            next_embed = {key: embeds[:, T // 2 + i, :] for key, embeds in embeds_dict.items()}

            vae_outputs = self.cell.forward(inputs=next_embed, outputs=next_embed, conditions=curr_embed)

            # Collect outputs for each key
            for key in embeds_dict:
                next_embed_pred = vae_outputs["decoder_outputs"][key]
                next_embed_pred = next_embed_pred.unsqueeze(1)
                pred_embeds_dict[key].append(next_embed_pred)
                
            # only one kl loss
            kl_loss.append(vae_outputs["kl_loss"])
            reconstruction_loss.append(vae_outputs["reconstruction_loss"])
            
        # Aggregate losses and construct final predictions dictionary
        kl_loss = torch.stack(kl_loss)
        reconstruction_loss = torch.stack(reconstruction_loss)
        final_pred_embeds = {key: torch.cat(preds, dim=1) for key, preds in pred_embeds_dict.items()}

        return OrderedDict(
            recon_embedding_loss=reconstruction_loss, # should be unused
            kl_loss=kl_loss,
            latent_features=final_pred_embeds,
        )
        
    def sample_future(self, 
                      embeds_dict,
                      n_future=20,
                      ):
        B, T, _ = next(iter(embeds_dict.values())).shape

        if B != 1:
            assert NotImplementedError

        pred_embeds_dict = {key: [] for key in embeds_dict}
        
        test_key = list(embeds_dict.keys())[0]

        context = T - 1 # In evaluation, we have no future states
        
        for key, embeds in embeds_dict.items():
            embeds_dict[key] = torch.tile(embeds, (n_future, 1, 1))
        
        for i in range(T - 1):
            if i == 0:
                curr_embed = {key: embeds[:, -context:] for key, embeds in embeds_dict.items()}
            else:
                curr_embed = {key: torch.cat([embeds_dict[key][:, :, :], torch.cat(pred_embeds_dict[key], dim=1)], dim=1)[:, -context:]
                            for key in embeds_dict}

            assert curr_embed[test_key].shape[1] == T - 1 # (B, T, E)
            assert curr_embed[test_key].shape[0] == B * n_future
            
            conditions = curr_embed
            reconstructions = {}
            conditions_lst = list(conditions.values())  # Extracting all tensor values
            concat_conditions = torch.cat(conditions_lst, dim=-1)  # Concatenate along the feature dimension
            
            for key in self.algo_config.dyn.obs_sequence:
            
                conditions_passed_in = conditions[key] 
                if self.cell_type == "no_fusion":
                    reconstructions_single = self.cell.decode(
                        conditions=concat_conditions,
                        z=None,
                        n=n_future,
                    )   
                else:
                    reconstructions_single = self.cell.decode(
                        conditions=concat_conditions if self.cell.decoder_concat_condition else conditions_passed_in,
                        z=None,
                        obs_key=key,
                        n=n_future,
                        prior_conditions=concat_conditions if self.cell.prior_concat_condition else conditions_passed_in,
                    )   
                
                reconstructions_single.pop("transformer_encoder_outputs", None)
                
                for key in reconstructions_single:
                    # print(key)
                    reconstructions[key] = reconstructions_single[key][:, -1, :]
            
            for key in embeds_dict:
                next_embed_pred = reconstructions[key]
                next_embed_pred = next_embed_pred.unsqueeze(1)
                pred_embeds_dict[key].append(next_embed_pred)
                
            final_pred_embeds = {key: torch.cat(preds, dim=1) for key, preds in pred_embeds_dict.items()}
            if len(list(embeds_dict.keys())) == 1:
                final_pred_embeds = final_pred_embeds[list(embeds_dict.keys())[0]]

        return OrderedDict(
            latent_features=final_pred_embeds,
        )


class DynamicsVAE_MultiTask_Cell_MultiCamera(torch.nn.Module):
    """ 
    Input: image embeddings (not images)
    Output: next image embedding
    """
    def __init__(
        self,
        embed_dim,
        history_length,
        obs_keys,
        output_shapes,
        latent_dim,
        device,
        
        decoder_concat_condition=False,
        prior_concat_condition=True,
        
        condition_shapes=None,
        decoder_is_conditioned=True,
        decoder_reconstruction_sum_across_elements=False,
        latent_clip=None,
        output_squash=(),
        output_scales=None,
        output_ranges=None,
        prior_learn=False,
        prior_is_conditioned=False,
        prior_layer_dims=(),
        prior_use_gmm=False,
        prior_gmm_num_modes=10,
        prior_gmm_learn_weights=False,
        encoder_kwargs=None,
        algo_configs_transformer=None,
    ):
        
        super(DynamicsVAE_MultiTask_Cell_MultiCamera, self).__init__()

        self.embed_dim = embed_dim
        self.history_length = history_length

        self.latent_dim = latent_dim
        self.latent_clip = latent_clip
        self.device = device
        
        # if decoder concat condition: separate decoders, each with condition[key]
        # else: combine into one condition, put into separate decoders 
        self.decoder_concat_condition = decoder_concat_condition
        self.prior_concat_condition = prior_concat_condition
        
        self.obs_keys = obs_keys
        assert isinstance(self.obs_keys, list) 

        self.algo_configs_transformer = algo_configs_transformer

        # encoder and decoder input dicts and output shapes dict for reconstruction
        assert isinstance(output_shapes, OrderedDict)
        self.output_shapes = deepcopy(output_shapes)

        # cVAE configs
        self._is_cvae = True
        self.decoder_is_conditioned = decoder_is_conditioned
        self.prior_is_conditioned = prior_is_conditioned
        assert self.decoder_is_conditioned or self.prior_is_conditioned, \
            "cVAE must be conditioned in decoder and/or prior"
        if self.prior_is_conditioned:
            assert prior_learn, "to pass conditioning inputs to prior, prior must be learned"

        self.condition_shapes = deepcopy(condition_shapes) if condition_shapes is not None else OrderedDict()

        # determines whether outputs are squashed with tanh and if so, to what scaling
        assert not (output_scales is not None and output_ranges is not None)
        self.output_squash = output_squash
        self.output_scales = output_scales if output_scales is not None else OrderedDict()
        self.output_ranges = output_ranges if output_ranges is not None else OrderedDict()

        assert set(self.output_squash) == set(self.output_scales.keys())
        assert set(self.output_squash).issubset(set(self.output_shapes))

        # decoder settings
        self.decoder_reconstruction_sum_across_elements = decoder_reconstruction_sum_across_elements

        self._encoder_kwargs = encoder_kwargs

        # prior parameters
        self.prior_learn = prior_learn
        self.prior_layer_dims = prior_layer_dims
        self.prior_use_gmm = prior_use_gmm
        self.prior_gmm_num_modes = prior_gmm_num_modes
        self.prior_gmm_learn_weights = prior_gmm_learn_weights

        if self.prior_use_gmm:
            assert self.prior_learn, "GMM must be learned"

        # create encoder, decoder, prior
        self._create_layers()

    def _create_layers(self):
        """
        Creates the encoder, decoder, and prior networks.
        """
        self.nets = nn.ModuleDict(
            {
                "encoders": nn.ModuleDict(),
                "decoders": nn.ModuleDict(),
            }
        )

        # VAE Encoder
        self._create_encoder()

        # VAE Decoder
        self._create_decoder()

        # VAE Prior.
        self._create_prior()

    def _create_encoder(self):
        """
        Helper function to create encoder.
        """
        # input_dim: concat history and current embedding
        input_dim = self.embed_dim #* (self.history_length + 1)
        
        # encoder outputs posterior distribution parameters
        encoder_output_shapes = OrderedDict(
            mean=(self.latent_dim,), 
            logvar=(self.latent_dim,),
        )

        for key in self.obs_keys:
            
            self.nets["encoders"][key] = MIMO_Transformer_Dyn(
                input_dim=input_dim,
                output_shapes=encoder_output_shapes,
                **BaseNets.transformer_args_from_config(self.algo_configs_transformer),
                )

    def _create_decoder(self):
        """
        Helper function to create decoder.
        """
        
        if self.decoder_concat_condition:
            E_dim = self.embed_dim * len(list(self.obs_keys))
        else:
            E_dim = self.embed_dim

        # inputs: latent + history condition
        input_dim = self.latent_dim + E_dim #* (self.history_length)

        algo_configs_transformer = deepcopy(self.algo_configs_transformer)
        algo_configs_transformer.unlock()
        algo_configs_transformer.context_length = algo_configs_transformer.context_length - 1 

        for key in self.obs_keys:
            self.nets["decoders"][key] = MIMO_Transformer_Dyn(
                input_dim=input_dim,
                output_shapes=OrderedDict({key: self.output_shapes[key]}), # output only one single key 
                **BaseNets.transformer_args_from_config(algo_configs_transformer),
                )

    def _create_prior(self):

        assert self.prior_concat_condition is True # by default, concat conditions and pass into one prior 

        # # prior possibly takes "condition" (if cVAE) and "goal" (if goal-conditioned)
        # prior_obs_group_shapes = OrderedDict(condition=None, goal=None)
        # prior_obs_group_shapes["condition"] = OrderedDict(self.condition_shapes)

        E_dim = self.embed_dim * len(list(self.obs_keys)) # by default, concat conditions and pass into one prior 

        algo_configs_transformer = deepcopy(self.algo_configs_transformer)
        algo_configs_transformer.unlock()
        algo_configs_transformer.context_length = algo_configs_transformer.context_length - 1 

        self.nets["prior"] = GaussianPrior_Transformer(
            latent_dim=self.latent_dim,
            device=self.device,
            latent_clip=self.latent_clip,
            learnable=self.prior_learn,
            use_gmm=self.prior_use_gmm,
            gmm_num_modes=self.prior_gmm_num_modes,
            gmm_learn_weights=self.prior_gmm_learn_weights,
            input_dim=E_dim,
            mlp_layer_dims=self.prior_layer_dims,
            goal_shapes=None,
            encoder_kwargs=self._encoder_kwargs,
            algo_configs_transformer=algo_configs_transformer,
        )

    def encode(self, inputs, conditions=None, goals=None, obs_key=None):

        assert conditions is not None
    
        inputs = inputs[:,None,:]
        inputs = torch.cat((inputs, conditions), dim=-2)

        return self.nets["encoders"][obs_key](inputs)

    def reparameterize(self, posterior_params):

        return TorchUtils.reparameterize(
            mu=posterior_params["mean"], 
            logvar=posterior_params["logvar"],
        )

    def decode(self, conditions=None, goals=None, z=None, n=None, obs_key=None, prior_conditions=None):

        if z is None:
            # sample latents from prior distribution
            assert n is not None
            z = self.sample_prior(n=n, conditions=prior_conditions, goals=goals)
            
        assert conditions is not None
        
        B, T, _ = conditions.shape
        B_z, E_z = z.shape
        
        # print("conditions.shape: ", conditions.shape)
        # print("z.shape: ", z.shape)
        
        # print("B, T: ", B, T)
        # print("B_z, E_z: ", B_z, E_z)
        
        assert B == B_z
        
        z = z.unsqueeze(1).expand(B, T, E_z)
        
        inputs = torch.cat((z, conditions), dim=-1)

        # pass through decoder to reconstruct variables in @self.output_shapes
        recons = self.nets["decoders"][obs_key](inputs)

        # apply tanh squashing to output modalities
        for k in self.output_squash:
            recons[k] = self.output_scales[k] * torch.tanh(recons[k])

        for k, v_range in self.output_ranges.items():
            assert v_range[1] > v_range[0]
            recons[k] = torch.sigmoid(recons[k]) * (v_range[1] - v_range[0]) + v_range[0]
        return recons

    def sample_prior(self, n, conditions=None, goals=None):

        return self.nets["prior"].sample(n=n, obs_dict=conditions, goal_dict=goals)

    def kl_loss(self, posterior_params, encoder_z=None, conditions=None, goals=None):
        
        return self.nets["prior"].kl_loss(
            posterior_params=posterior_params,
            z=encoder_z,
            obs_dict=conditions, 
            goal_dict=goals,
        )

    def reconstruction_loss(self, reconstructions, targets):
        """
        Reconstruction loss. Note that we compute the average per-dimension error
        in each modality and then average across all the modalities.

        The beta term for weighting between reconstruction and kl losses will
        need to be tuned in practice for each situation (see
        https://twitter.com/memotv/status/973323454350090240 for more 
        discussion).

        Args:
            reconstructions (dict): reconstructed inputs, consistent with
                @self.output_shapes
            targets (dict): reconstruction targets, consistent with
                @self.output_shapes

        Returns:
            reconstruction_loss (torch.Tensor): VAE reconstruction loss
        """
        random_key = list(reconstructions.keys())[0]
        batch_size = reconstructions[random_key].shape[0]
        num_mods = len(reconstructions.keys())

        # collect errors per modality, while preserving shapes in @reconstructions
        recons_errors = []
        for k in reconstructions:
            L2_loss = (reconstructions[k] - targets[k]).pow(2)
            recons_errors.append(L2_loss)

        # reduce errors across modalities and dimensions
        if self.decoder_reconstruction_sum_across_elements:
            # average across batch but sum across modalities and dimensions
            loss = sum([x.sum() for x in recons_errors])
            loss /= batch_size
        else:
            # compute mse loss in each modality and average across modalities
            loss = sum([x.mean() for x in recons_errors])
            loss /= num_mods
        return loss

    def product_of_experts(self, m_vect, v_vect):
        T_vect = 1.0 / v_vect

        mu = (m_vect * T_vect).sum(2) * (1 / T_vect.sum(2))
        var = 1 / T_vect.sum(2)

        return mu, var

    def forward(self, inputs, outputs, conditions=None, goals=None, freeze_encoder=False):

        # In the comments below, X = inputs, Y = conditions, and we seek to learn P(X | Y).
        # The decoder and prior only have knowledge about Y and try to reconstruct X.
        # Notice that when Y is the empty set, this reduces to a normal VAE.

        # mu, logvar <- Enc(X, Y)
        
        posterior_params = {}
        
        for key in inputs:
            posterior_params[key] = self.encode(
                inputs=inputs[key], 
                conditions=conditions[key],
                goals=goals,
                obs_key=key,
            )
        
            # Take only the last time step
            posterior_params[key] = {param: value[:, -1, :] for param, value in posterior_params[key].items()}

        if freeze_encoder:
            for key in posterior_params:
                posterior_params[key] = {param: TensorUtils.detach(value) for param, value in posterior_params[key].items()}

        # Step 2: Use Product of Experts to combine posterior parameters
        mean_vect = torch.stack([posterior_params[key]['mean'] for key in posterior_params], dim=-1)
        logvar_vect = torch.stack([posterior_params[key]['logvar'] for key in posterior_params], dim=-1)

        mean_combined, var_combined = self.product_of_experts(mean_vect, torch.exp(logvar_vect))
        
        posterior_params_combined = {
            "mean": mean_combined,
            "logvar": torch.log(var_combined)  
        }
        
        conditions_lst = list(conditions.values())  # Extracting all tensor values
        concat_conditions = torch.cat(conditions_lst, dim=-1)  # Concatenate along the feature dimension

        # z ~ Enc(z | X, Y)
        encoder_z = self.reparameterize(posterior_params_combined)

        reconstructions = {}
        
        for key in self.obs_keys:
            
            conditions_passed_in = concat_conditions if self.decoder_concat_condition else conditions[key] 
            
            # hat(X) = Dec(z, Y)
            reconstructions_single = self.decode(
                conditions=conditions_passed_in, 
                goals=goals,
                z=encoder_z,
                obs_key=key,
            )   
            
            reconstructions_single.pop("transformer_encoder_outputs", None)
                
            for key in reconstructions_single:
                # print(key)
                reconstructions[key] = reconstructions_single[key][:, -1, :]

        # just calculate first; might not use later.
        reconstruction_loss = self.reconstruction_loss(
            reconstructions=reconstructions, 
            targets=outputs,
        )
        
        if self.prior_concat_condition: # assert this is true first 
            
            # this will also train prior network z ~ Prior(z | Y)
            kl_loss = self.kl_loss(
                posterior_params=posterior_params_combined,
                encoder_z=encoder_z,
                conditions=concat_conditions,
                goals=goals,
            )

        else:
            assert NotImplementedError

        return {
            "encoder_params" : posterior_params_combined,
            "encoder_z" : encoder_z,
            "decoder_outputs" : reconstructions,
            "kl_loss" : kl_loss,
            "reconstruction_loss" : reconstruction_loss,
            "reconstructions": reconstructions,
            "targets": outputs,
        }



class DynamicsVAE_MultiTask_Cell_MultiCamera_NoFusion(DynamicsVAE_MultiTask_Cell_MultiCamera):
    """ 
    Input: image embeddings (not images)
    Output: next image embedding
    """

    def _create_layers(self):
        """
        Creates the encoder, decoder, and prior networks.
        """
        # single encoder / decoder; no sensory fusion; prior not conditioned
        self.nets = nn.ModuleDict()
        
        # VAE Encoder
        self._create_encoder()

        # VAE Decoder
        self._create_decoder()

        # VAE Prior.
        self._create_prior()

    def _create_encoder(self):
        """
        Helper function to create encoder.
        """
        
        # single encoder / decoder; no sensory fusion; prior not conditioned
        input_dim = self.embed_dim * len(self.obs_keys)
        
        encoder_output_shapes = OrderedDict(
            mean=(self.latent_dim,), 
            logvar=(self.latent_dim,),
        )
        
        self.nets["encoders"] = MIMO_Transformer_Dyn(
            input_dim=input_dim,
            output_shapes=encoder_output_shapes,
            **BaseNets.transformer_args_from_config(self.algo_configs_transformer),
            )

        
    def _create_decoder(self):
        """
        Helper function to create decoder.
        """
            
        E_dim = self.embed_dim * len(list(self.obs_keys))
            
        # inputs: latent + history condition
        input_dim = self.latent_dim + E_dim #* (self.history_length)

        algo_configs_transformer = deepcopy(self.algo_configs_transformer)
        algo_configs_transformer.unlock()
        algo_configs_transformer.context_length = algo_configs_transformer.context_length - 1 

        for key in self.obs_keys:
            self.nets["decoders"] = MIMO_Transformer_Dyn(
                input_dim=input_dim,
                output_shapes=OrderedDict(self.output_shapes), 
                **BaseNets.transformer_args_from_config(algo_configs_transformer),
                )

    def _create_prior(self):

        algo_configs_transformer = deepcopy(self.algo_configs_transformer)
        algo_configs_transformer.unlock()
        algo_configs_transformer.context_length = algo_configs_transformer.context_length - 1 
        
        self.nets["prior"] = GaussianPrior(
            latent_dim=self.latent_dim,
            device=self.device,
            latent_clip=self.latent_clip,
            learnable=self.prior_learn,
            use_gmm=self.prior_use_gmm,
            gmm_num_modes=self.prior_gmm_num_modes,
            gmm_learn_weights=self.prior_gmm_learn_weights,
            obs_shapes=None,
            mlp_layer_dims=self.prior_layer_dims,
            goal_shapes=None,
            encoder_kwargs=self._encoder_kwargs,
        )

    def encode(self, inputs, conditions=None, goals=None):

        assert conditions is not None
    
        inputs = inputs[:,None,:]
        inputs = torch.cat((inputs, conditions), dim=-2)

        return self.nets["encoders"](inputs)

    def decode(self, conditions=None, goals=None, z=None, n=None):

        if z is None:
            # sample latents from prior distribution
            assert n is not None
            z = self.sample_prior(n=n, conditions=None, goals=goals) # conditions = None for prior
            
        assert conditions is not None # conditions is not None for decoder 
        
        B, T, _ = conditions.shape
        B_z, E_z = z.shape
        
        assert B == B_z
        
        z = z.unsqueeze(1).expand(B, T, E_z)
        
        inputs = torch.cat((z, conditions), dim=-1)

        # pass through decoder to reconstruct variables in @self.output_shapes
        recons = self.nets["decoders"](inputs)

        # apply tanh squashing to output modalities
        for k in self.output_squash:
            recons[k] = self.output_scales[k] * torch.tanh(recons[k])

        for k, v_range in self.output_ranges.items():
            assert v_range[1] > v_range[0]
            recons[k] = torch.sigmoid(recons[k]) * (v_range[1] - v_range[0]) + v_range[0]
        return recons

    def sample_prior(self, n, conditions=None, goals=None):
        
        assert conditions is None

        return self.nets["prior"].sample(n=n, obs_dict=conditions, goal_dict=goals)

    def kl_loss(self, posterior_params, encoder_z=None, conditions=None, goals=None):
        
        assert conditions is None # prior not conditioned
        
        return self.nets["prior"].kl_loss(
            posterior_params=posterior_params,
            z=encoder_z,
            obs_dict=conditions, 
            goal_dict=goals,
        )

    def forward(self, inputs, outputs, conditions=None, goals=None, freeze_encoder=False):

        # In the comments below, X = inputs, Y = conditions, and we seek to learn P(X | Y).
        # The decoder and prior only have knowledge about Y and try to reconstruct X.
        # Notice that when Y is the empty set, this reduces to a normal VAE.

        # mu, logvar <- Enc(X, Y)
        
        # concat all observations in inputs by keys
        inputs_concat = torch.cat([inputs[key] for key in self.obs_keys], dim=-1)
        conditions_concat = torch.cat([conditions[key] for key in self.obs_keys], dim=-1)
        
        posterior_params = self.encode(
            inputs=inputs_concat, 
            conditions=conditions_concat,
            goals=goals,
        )
    
        # Take only the last time step
        posterior_params = {param: value[:, -1, :] for param, value in posterior_params.items()}
        
        if freeze_encoder:
            posterior_params = {param: TensorUtils.detach(value) for param, value in posterior_params.items()}

        # z ~ Enc(z | X, Y)
        encoder_z = self.reparameterize(posterior_params)
        
        # hat(X) = Dec(z, Y)
        reconstructions = self.decode(
            conditions=conditions_concat, 
            goals=goals,
            z=encoder_z,
        )   
        
        reconstructions.pop("transformer_encoder_outputs", None)
            
        for key in reconstructions:
            reconstructions[key] = reconstructions[key][:, -1, :]

        # just calculate first; might not use later.
        reconstruction_loss = self.reconstruction_loss(
            reconstructions=reconstructions, 
            targets=outputs,
        )
        
        # this will also train prior network z ~ Prior(z | Y)
        # assume condition is None
        kl_loss = self.kl_loss(
            posterior_params=posterior_params,
            encoder_z=encoder_z,
            conditions=None,
            goals=goals,
        )

        return {
            "encoder_params" : posterior_params,
            "encoder_z" : encoder_z,
            "decoder_outputs" : reconstructions,
            "kl_loss" : kl_loss,
            "reconstruction_loss" : reconstruction_loss,
            "reconstructions": reconstructions,
            "targets": outputs,
        }


#######################################################################################

class DynamicsVAE(nn.Module):

    def __init__(self, embed_dim, action_dim, device, algo_config, obs_config, use_history=False, use_real=True):
        super().__init__()

        self.algo_config = algo_config
        self.obs_config = obs_config

        self.use_history = use_history
        self.use_real = use_real 

        self.cell = DynamicsVAE_Cell(
                        embed_dim=embed_dim,
                        action_dim=action_dim,
                        device=device,
                        use_history=use_history,
                        encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(obs_config.encoder),
                        **VAENets.vae_args_from_config(algo_config.vae),
                        enc_use_res_mlp=algo_config.vae.enc_use_res_mlp,
                        dec_use_res_mlp=algo_config.vae.dec_use_res_mlp,
                        **res_mlp_args_from_config_vae(algo_config.res_mlp),  
                    )

    def forward(self,
                embeds,              # tensor(T,      B, E)
                actions,             # tensor(T // 2, B, A)
                ):

        T, B = embeds.shape[:2]

        kl_loss = []
        reconstruction_loss = []
        pred_embeds = []

        embed_first_half = embeds[:T//2]
        embed_second_half = embeds[T//2:]

        curr_history = embeds[1:T//2+1]

        embed = embed_second_half[0]

        assert actions.shape[0] == embed_second_half.shape[0] == T // 2

        for i in range(T//2 - 1):

            assert len(pred_embeds) == i

            if self.use_history:
                if self.use_real:
                    curr_embed = torch.cat([embed_first_half[i+1:], embed_second_half[:i+1]], dim=0)
                else:
                    if i == 0:
                        curr_embed = curr_history[i:]
                    else:
                        curr_embed = torch.cat([curr_history[i:], torch.stack(pred_embeds[:i])], dim=0)
                
                assert curr_embed.shape[0] == T // 2 # (T, B, E)

                curr_embed = torch.permute(curr_embed, (1, 0, 2)) # (B, T, E)
                curr_embed = curr_embed.reshape(-1, curr_embed.shape[1] * curr_embed.shape[2]) # (B, T * E)

                assert curr_embed.shape[0] == B

            else:
                if self.use_real:
                    curr_embed = embed_second_half[i]
                else:
                    curr_embed = embed
            
            action = actions[i]
            next_embed=embed_second_half[i+1]

            vae_outputs = self.cell.forward(
                next_embed=next_embed, 
                curr_embed=curr_embed, 
                action=action,
            )
            
            embed = vae_outputs["decoder_outputs"]['next_embed']
            kl_loss.append(vae_outputs["kl_loss"])
            reconstruction_loss.append(vae_outputs["reconstruction_loss"])
            pred_embeds.append(embed)
        
        pred_embeds = torch.stack(pred_embeds)
        kl_loss = torch.stack(kl_loss) 
        reconstruction_loss = torch.stack(reconstruction_loss)
        total_loss = reconstruction_loss + self.algo_config.vae.kl_weight * kl_loss

        return OrderedDict(
            recons_loss=reconstruction_loss,
            kl_loss=kl_loss,
            dyn_loss=total_loss,
            pred_obs_embedding=pred_embeds,
        )


class DynamicsVAE_Cell(Module):
    """
    Modelling dynamics with a VAE.
    """
    def __init__(
        self,
        embed_dim,
        action_dim,
        encoder_layer_dims,
        decoder_layer_dims,
        latent_dim,
        device,
        use_history=True,
        conditioned_on_obs=True,
        decoder_is_conditioned=True,
        decoder_reconstruction_sum_across_elements=False,
        latent_clip=None,
        prior_learn=False,
        prior_is_conditioned=False,
        prior_layer_dims=(),
        prior_use_gmm=False,
        prior_gmm_num_modes=10,
        prior_gmm_learn_weights=False,
        prior_use_categorical=False,
        prior_categorical_dim=10,
        prior_categorical_gumbel_softmax_hard=False,
        goal_shapes=None,
        encoder_kwargs=None,
        enc_use_res_mlp=False,
        dec_use_res_mlp=False,
        res_mlp_kwargs=None,
        action_network=True,
    ):
        super(DynamicsVAE_Cell, self).__init__()

        self.action_network = action_network

        # input: next_obs embedding
        self.input_shapes = OrderedDict()
        self.input_shapes["next_embed"] = (embed_dim,)

        # condition: curr_obs embedding, action
        self.condition_shapes = OrderedDict()
        if use_history:
            self.condition_shapes["curr_embed"] = (embed_dim * 10,)
        else:
            self.condition_shapes["curr_embed"] = (embed_dim,)

        if self.action_network:
            self.condition_shapes["action"] = (embed_dim,)
        else:
            self.condition_shapes["action"] = (action_dim,)

        self._vae = VAE(
            input_shapes=self.input_shapes,
            output_shapes=self.input_shapes,
            encoder_layer_dims=encoder_layer_dims,
            decoder_layer_dims=decoder_layer_dims,
            latent_dim=latent_dim,
            device=device,
            condition_shapes=self.condition_shapes,
            decoder_is_conditioned=decoder_is_conditioned,
            decoder_reconstruction_sum_across_elements=decoder_reconstruction_sum_across_elements,
            latent_clip=latent_clip,
            prior_learn=prior_learn,
            prior_is_conditioned=prior_is_conditioned,
            prior_layer_dims=prior_layer_dims,
            prior_use_gmm=prior_use_gmm,
            prior_gmm_num_modes=prior_gmm_num_modes,
            prior_gmm_learn_weights=prior_gmm_learn_weights,
            prior_use_categorical=prior_use_categorical,
            prior_categorical_dim=prior_categorical_dim,
            prior_categorical_gumbel_softmax_hard=prior_categorical_gumbel_softmax_hard,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
            enc_use_res_mlp=enc_use_res_mlp,
            dec_use_res_mlp=dec_use_res_mlp,
            resmlp_kwargs=res_mlp_kwargs,
        )

        if self.action_network:
            self._action_embedding = nn.Linear(action_dim, embed_dim)

    def sample_prior(self, obs_dict=None, goal_dict=None, n=None):
        return self._vae.sample_prior(n=n, conditions=obs_dict, goals=goal_dict)

    def decode(self, obs_dict, goal_dict=None, z=None, n=None):
        conditions = {} # conditioned on the first image in a sequence
        for key in obs_dict:
            conditions[key] = obs_dict[key].expand(n, -1, -1)

        assert (n is not None) and (z is None)

        return self._vae.decode(conditions=conditions, goals=goal_dict, z=None, n=n)

    def decode_branched_future(self, obs_dict, goal_dict=None, z=None, n=None):
        conditions = {} # conditioned on the first image in a sequence
        for key in obs_dict:
            conditions[key] = obs_dict[key]
            
        assert (n is not None) and (z is None)

        return self._vae.decode(conditions=conditions, goals=goal_dict, z=None, n=n)

    def forward_train(self, inputs, conditions):

        return self._vae.forward(
            inputs=inputs,
            outputs=inputs,
            conditions=conditions,
            goals=None,
            freeze_encoder=False)

    def forward(self, next_embed, curr_embed, action):

        if self.action_network:
            action = self._action_embedding(action)

        return self.forward_train(
                inputs={"next_embed": next_embed},
                conditions={"curr_embed": curr_embed, "action": action},
            )


class RewardTransformer(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 action_dim, 
                 algo_configs=None,
                 wm_configs=None,
                ):
        super().__init__()
        
        self.algo_configs = algo_configs
        self.wm_configs = wm_configs
        
        if self.wm_configs.rew.use_action:
            hidden_dim = self.wm_configs.rew.action.hidden_dim
            action_network_dim = self.wm_configs.rew.action.output_dim
            num_layers = self.wm_configs.rew.action.num_layers
            
            self._action_network = MLP(
                input_dim=action_dim,
                output_dim=action_network_dim,
                layer_dims=[hidden_dim] * num_layers,
                activation=nn.ELU,
                output_activation=None,
                normalization=True,
            )
            input_dim = embed_dim + action_network_dim
        else:
            input_dim = embed_dim + action_dim

        # Three classses prediction
        output_shapes = OrderedDict(
            reward=(3,)
        )
        
        algo_configs_transformer = deepcopy(self.algo_configs.transformer)
        algo_configs_transformer.unlock()
        algo_configs_transformer.context_length = 9 # TODO: hack for now
    
        self._transformer = MIMO_Transformer_Dyn(
            input_dim=input_dim,
            output_shapes=output_shapes,
            **BaseNets.transformer_args_from_config(algo_configs_transformer),
            )

    def forward(self, embed, action):
        if self.wm_configs.rew.use_action:
            action = self._action_network(action)
            
        pred = torch.cat([embed, action], dim=-1)        
        pred = self._transformer(pred)["reward"]
        return pred 