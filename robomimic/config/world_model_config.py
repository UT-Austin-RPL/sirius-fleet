"""
Config for BC algorithm.
"""

from robomimic.config.base_config import BaseConfig


class WorldModelConfig(BaseConfig):
    ALGO_NAME = "world_model"

    def train_config(self):
        """
        BC algorithms don't need "next_obs" from hdf5 - so save on storage and compute by disabling it.
        """
        super(WorldModelConfig, self).train_config()
        self.train.hdf5_load_next_obs = False

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """

        # optimization parameters (dynamics)
        self.algo.optim_params.dynamics.optimizer_type = "adam"
        self.algo.optim_params.dynamics.learning_rate.initial = 0.0001     # world model rate
        self.algo.optim_params.dynamics.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.dynamics.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
        self.algo.optim_params.dynamics.learning_rate.scheduler_type = "constant" # learning rate scheduler ("multistep", "linear", etc) 
        self.algo.optim_params.dynamics.regularization.L2 = 0.00          # L2 regularization strength

        # optimization parameters (policy)
        self.algo.optim_params.policy.optimizer_type = "adam"
        self.algo.optim_params.policy.learning_rate.initial = 0.001      # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
        self.algo.optim_params.policy.learning_rate.scheduler_type = "constant_with_warmup" # learning rate scheduler ("multistep", "linear", etc) 
        self.algo.optim_params.policy.regularization.L2 = 0.00          # L2 regularization strength

        # loss weights
        self.algo.loss.l2_weight = 1.0      # L2 loss weight
        self.algo.loss.l1_weight = 0.0      # L1 loss weight
        self.algo.loss.cos_weight = 0.0     # cosine loss weight

        # MLP network architecture (layers after observation encoder and RNN, if present)
        self.algo.actor_layer_dims = (1024, 1024)

        # stochastic Gaussian policy settings
        self.algo.gaussian.enabled = False              # whether to train a Gaussian policy
        self.algo.gaussian.fixed_std = False            # whether to train std output or keep it constant
        self.algo.gaussian.init_std = 0.1               # initial standard deviation (or constant)
        self.algo.gaussian.min_std = 0.01               # minimum std output from network
        self.algo.gaussian.std_activation = "softplus"  # activation to use for std output from policy net
        self.algo.gaussian.low_noise_eval = True        # low-std at test-time 

        # stochastic GMM policy settings
        self.algo.gmm.enabled = False                   # whether to train a GMM policy
        self.algo.gmm.num_modes = 5                     # number of GMM modes
        self.algo.gmm.min_std = 0.0001                  # minimum std output from network
        self.algo.gmm.std_activation = "softplus"       # activation to use for std output from policy net
        self.algo.gmm.low_noise_eval = True             # low-std at test-time 

        # stochastic VAE policy settings
        self.algo.vae.enabled = False                   # whether to train a VAE policy
        self.algo.vae.latent_dim = 16                   # VAE latent dimnsion - set to twice the dimensionality of action space
        self.algo.vae.latent_clip = None                # clip latent space when decoding (set to None to disable)
        self.algo.vae.kl_weight = 0.00001                    # beta-VAE weight to scale KL loss relative to reconstruction loss in ELBO

        # VAE decoder settings
        self.algo.vae.decoder.is_conditioned = True                         # whether decoder should condition on observation
        self.algo.vae.decoder.reconstruction_sum_across_elements = False    # sum instead of mean for reconstruction loss

        # VAE prior settings
        self.algo.vae.prior.learn = True                                   # learn Gaussian / GMM prior instead of N(0, 1)
        self.algo.vae.prior.is_conditioned = True                          # whether to condition prior on observations
        self.algo.vae.prior.use_gmm = True                                 # whether to use GMM prior
        self.algo.vae.prior.gmm_num_modes = 10                              # number of GMM modes
        self.algo.vae.prior.gmm_learn_weights = True                       # whether to learn GMM weights 
        
        self.algo.vae.prior_layer_dims = (1024, 1024)                         # prior MLP layer dimensions (if learning conditioned prior)

        # RNN policy settings
        self.algo.rnn.enabled = False                               # whether to train RNN policy
        self.algo.rnn.horizon = 10                                  # unroll length for RNN - should usually match train.seq_length
        self.algo.rnn.hidden_dim = 400                              # hidden dimension size    
        self.algo.rnn.rnn_type = "LSTM"                             # rnn type - one of "LSTM" or "GRU"
        self.algo.rnn.num_layers = 2                                # number of RNN layers that are stacked
        self.algo.rnn.open_loop = False                             # if True, action predictions are only based on a single observation (not sequence)
        self.algo.rnn.kwargs.bidirectional = False                  # rnn kwargs
        self.algo.rnn.kwargs.do_not_lock_keys()

        # Transformer policy settings
        self.algo.transformer.enabled = False                       # whether to train transformer policy
        self.algo.transformer.context_length = 10                   # length of (s, a) seqeunces to feed to transformer - should usually match train.frame_stack
        self.algo.transformer.embed_dim = 512                       # dimension for embeddings used by transformer
        self.algo.transformer.num_layers = 6                        # number of transformer blocks to stack
        self.algo.transformer.num_heads = 8                         # number of attention heads for each transformer block (should divide embed_dim evenly)
        self.algo.transformer.emb_dropout = 0.1                     # dropout probability for embedding inputs in transformer
        self.algo.transformer.attn_dropout = 0.1                    # dropout probability for attention outputs for each transformer block
        self.algo.transformer.block_output_dropout = 0.1            # dropout probability for final outputs for each transformer block
        self.algo.transformer.sinusoidal_embedding = False          # if True, use standard positional encodings (sin/cos)
        self.algo.transformer.activation = "gelu"                   # activation function for MLP in Transformer Block
        self.algo.transformer.supervise_all_steps = False           # if true, supervise all intermediate actions, otherwise only final one
        self.algo.transformer.nn_parameter_for_timesteps = True     # if true, use nn.Parameter otherwise use nn.Embedding
        self.algo.transformer.pred_future_acs = False               # shift action prediction forward to predict future actions instead of past actions
        self.algo.transformer.causal = True                         # whether the transformer is causal

        self.algo.language_conditioned = False                      # whether policy is language conditioned

        # Residual MLP settings
        self.algo.res_mlp.enabled = False
        self.algo.res_mlp.num_blocks = 4
        self.algo.res_mlp.hidden_dim = 1024
        self.algo.res_mlp.use_layer_norm = True
        
        # Dynamics Model
        self.algo.dyn.hidden_dim = 1024
        self.algo.dyn.action_network_dim = 100
        self.algo.dyn.num_layers = 2
        self.algo.dyn.dyn_weight = 1
        self.algo.dyn.use_res_mlp = False # unused
        self.algo.dyn.combine_enabled = True
        self.algo.dyn.stochastic_inputs = False
        self.algo.dyn.kl_balance = 0.8
        self.algo.dyn.dyn_class = "vae" # ["deter", "vae"]
        self.algo.dyn.start_training_epoch = None
        self.algo.dyn.image_output_activation = "sigmoid"
        
        self.algo.dyn.decoder_reconstruction_sum_across_elements = False

        self.algo.dyn.use_sep_decoder = False
        self.algo.dyn.no_dyn_debug = False
        self.algo.dyn.use_unet = True
        self.algo.dyn.no_action = False # Do not use action in future prediction.
        
        self.algo.dyn.obs_sequence = ["robot0_agentview_left_image",
                                      "robot0_eye_in_hand_image",
                                      ] # what image obs to use for encoding
        
        self.algo.dyn.obs_fusion_method = "concat" # how to fuse the image embeddings
        self.algo.dyn.obs_fusion_archi = "mlp" 
        self.algo.dyn.deterministic = True
        self.algo.dyn.history_length = 10
        self.algo.dyn.use_embedding_loss = False
        self.algo.dyn.embedding_loss_weight = 0.0001
        self.algo.dyn.recons_full_batch = False
        self.algo.dyn.dyn_train_embed_only = False
        self.algo.dyn.dyn_cell_type = "default"
        
        """ Classifier """ 
        self.algo.dyn.train_reward = False
        self.algo.dyn.load_ckpt = ""
        self.algo.dyn.rew.use_action = True
        self.algo.dyn.rew.action.hidden_dim = 1024
        self.algo.dyn.rew.action.output_dim = 100
        self.algo.dyn.rew.action.num_layers = 2
        self.algo.dyn.rew.activation = "elu"
        self.algo.dyn.rew.latent_stop_grad = True
        
        """ For the UNet architecture. """
        self.algo.dyn.unet.in_channels = 3
        self.algo.dyn.unet.out_channels = 3
        self.algo.dyn.unet.latent_channels = 4
        self.algo.dyn.unet.down_block_types = ["DownEncoderBlock2D", "DownEncoderBlock2D"] 
        self.algo.dyn.unet.block_out_channels = [32, 64] #(64,)
        self.algo.dyn.unet.layers_per_block = 1
        self.algo.dyn.unet.act_fn = "silu"
        self.algo.dyn.unet.norm_num_groups = 32
        self.algo.dyn.unet.up_block_types = ["UpDecoderBlock2D", "UpDecoderBlock2D"]
        
        """ For the FutureAhead architecture. """
        self.algo.dyn.future_ahead.enabled = False
        self.algo.dyn.future_ahead.number_of_history = 5
        self.algo.dyn.future_ahead.interval = 10
        
        self.algo.dyn.downscale_img = True
        self.algo.dyn.scaled_img_size = 84
        
        self.algo.dyn.kwargs.do_not_lock_keys()
        
        self.algo.max_gradient_norm = None