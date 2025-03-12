import os
import h5py
import argparse
import numpy as np

from shutil import copyfile

import matplotlib.pyplot as plt

import torch
import copy
import time

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.utils.file_utils import policy_from_checkpoint

import faiss

from sklearn import svm

from robomimic.algo.algo import RolloutPolicy

import robomimic_kitchen.utils.file_utils as FileUtilsKitchen
from robomimic_kitchen.utils.file_utils import policy_from_checkpoint as policy_from_checkpoint_kitchen
from robomimic_kitchen.utils.obs_utils import process_frame

def process_image(img, transpose=True):
    
    image = torch.from_numpy(img)[None, :, :, :, :].cuda().float()
    if transpose:
        image = image.permute(0, 1, 4, 2, 3) / 255.
    return image

def process_action(action):
    return torch.from_numpy(action)[None, :].cuda().float()

def get_obs_at_idx(obs, i):
    d = dict()
    for key in obs:
        d[key] = obs[key][i]
    return d

def process_shadowing_mode(obs):
    for key in obs:
        if "image" in key:
            obs[key] = ObsUtils.process_obs(obs[key], obs_modality='rgb')
    return obs

class ErrorDetector:
    def __init__(self):
        # shadowing: if human shadowing using recorded obs, set to True
        self.shadowing_node = False
        pass

    def evaluate(self, obs):
        assert NotImplementedError

    def evaluate_trajectory(self, obs_np):
        assert NotImplementedError

    def above_threshold(self, value):
        assert NotImplementedError
    
    def reset(self):
        pass

class VAEMoMaRT(ErrorDetector):
    def __init__(self, checkpoint, threshold):
        super(VAEMoMaRT, self).__init__()
        self.rollout_policy = policy_from_checkpoint(ckpt_path=checkpoint)[0]
        self.policy = self.rollout_policy.policy.nets["policy"]
        self.threshold = threshold
        self.seq_length = 10
        #self.image_key = "agentview_image"

        ckpt_dict = policy_from_checkpoint(ckpt_path=checkpoint)[1]
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        self.rollout_horizon = config.experiment.rollout.horizon

        self._value_history = []

    def evaluate(self, obs):
        vae_outputs = self.policy.forward(obs)
        return vae_outputs

    def evaluate_trajectory(self, obs_np):
        
        # raise NotImplementedError # not use for now 
        
        key_choices = list(obs_np.keys())
        keys = ["robot0_eye_in_hand_image", "robot0_agentview_left_image"]
        recons_loss_lst = []
        kl_loss_lst = []
        reconstructions_lst = []
        self.seq_length = 9
        for i in range(len(obs_np[keys[0]]) - self.seq_length):
            obs_input = {}
            for key in keys:
                obs_input[key] = obs_np[key][i : i + self.seq_length]
                obs_input[key] = process_image(obs_input[key])
                
            vae_outputs = self.evaluate(obs_input)
            reconstructions = vae_outputs["reconstructions"]
            recons_loss = vae_outputs["reconstruction_loss"].item()
            kl_loss = vae_outputs["kl_loss"].item()

            reconstructions_lst.append(reconstructions)
            recons_loss_lst.append(recons_loss)
            kl_loss_lst.append(kl_loss)

        # clear anomaly data
        for i in range(1):
            recons_loss_lst[i] = min(recons_loss_lst[i], 0.01)
            kl_loss_lst[i] = min(kl_loss_lst[i], 0.01)

        # append 0 for initial data (no history)
        recons_loss_lst = [0] * 10 + recons_loss_lst
        kl_loss_lst = [0] * 10 + kl_loss_lst

        print(recons_loss)

        return {"vae_score": np.array(recons_loss_lst), "vae_intv":self.above_threshold(np.array(recons_loss_lst))}


    def above_threshold(self, value):
        return value >= self.threshold

    def metric(self):
        return "Reconstruction Loss"

    def horizon(self):
        return self.rollout_horizon

    def human_intervene(self, obs_buffer): # of the current last observation
        # key shape in obs_buffer: (10, 128, 128, 3)
    
        keys = ["robot0_eye_in_hand_image", "robot0_agentview_left_image"]
        obs_input = {}
        for key in keys:
            obs_input[key] = obs_buffer[key][0 : 0 + self.seq_length]
            obs_input[key] = process_image(obs_input[key])
            
        vae_outputs = self.evaluate(obs_input)
        recons_loss = vae_outputs["reconstruction_loss"].item()
        
        return self.above_threshold(recons_loss), {"vae_score": recons_loss}

    def reset(self):
        self._value_history = []

class VAEGoal(VAEMoMaRT):
    def __init__(self, checkpoint, threshold):
        super(VAEGoal, self).__init__(checkpoint, threshold)
        self.sampling_num = 1024 # hardcode for now
        self.seq_length = 1 # only use the current observation

        self._value_history = []

    def evaluate(self, obs):
        keys = ["robot0_eye_in_hand_image", "robot0_agentview_left_image"]
        for key in keys:
            obs[key] = torch.squeeze(obs[key], 0)
            obs[key] = obs[key].expand(self.sampling_num, -1, -1, -1)
        
        recons = self.policy.decode(obs_dict=obs, n=self.sampling_num)
        
        recons_all = []
        for key in keys:
            recons_flat = torch.flatten(recons[key], start_dim=1)
            recons_all.append(recons_flat)
        recons_flat = torch.cat(recons_all, dim=1)

        variance = torch.var(recons_flat, dim=0).mean()
        # print(variance.item())
        return variance.item()


    def evaluate_trajectory(self, obs_np):
        var_list = []
        keys = ["robot0_eye_in_hand_image", "robot0_agentview_left_image"]
        for i in range(len(obs_np[keys[0]])):
            obs_input = {}
            for key in keys:
                obs_input[key] = obs_np[key][i:i+1]
                obs_input[key] = process_image(obs_input[key])
                
            variance = self.evaluate(obs_input)
            var_list.append(variance)
            
        var_np = np.array(var_list)
            
        return self.above_threshold(var_np), {"vae_score": var_np}

    def above_threshold(self, value):
        return value >= self.threshold

    def metric(self):
        return "VAE Goal Variance: "

    def horizon(self):
        return self.rollout_horizon

    def human_intervene(self, obs_buffer): # of the current last observation
        
        raise NotImplementedError # not use for now
        if len(obs_buffer) < self.seq_length: # not enough history yet, always good
            return False
        else:
            obs = obs_buffer[-1].copy()

            if self.shadowing_node:
                obs = process_shadowing_mode(obs)

            obs_input = RolloutPolicy._prepare_observation(self.rollout_policy, obs)
            variance = self.evaluate(obs_input)

            self._value_history.append(variance)

            print(self.metric(), variance)
            return self.above_threshold(variance) # if greater, is error
        
    def reset(self):

        self._value_history = []

        pass
    
class Ensemble(ErrorDetector):
    def __init__(self, checkpoints, threshold):
        super(Ensemble, self).__init__()
        self.policies = [policy_from_checkpoint_kitchen(ckpt_path=checkpoint)[0]
                         for checkpoint in checkpoints]

        for policy in self.policies:
            policy.start_episode()

        self.threshold = threshold

        ckpt_dict = policy_from_checkpoint_kitchen(ckpt_path=checkpoints[0])[1]
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        self.rollout_horizon = config.experiment.rollout.horizon
        self.no_gripper_action = False
        self.seq_length = 10

        self._value_history = []

    def evaluate(self, obs, no_gripper_action=False):

        actions = []
        for policy in self.policies:
            action = policy(obs)
            actions.append(action)
        action_variance = np.square(np.std(np.array(actions), axis=0)).mean()
        if no_gripper_action:
            for i in range(len(actions)):
                actions[i] = actions[i][:6]
            action_variance = np.square(np.std(np.array(actions), axis=0)).mean()
        return action_variance
    
    def get_uncompressed_single_step_action(self, obs):
        actions = []
        with torch.no_grad():
            for i in range(len(self.policies)):
                act = self.policies[i].policy.nets['policy'].forward(obs_dict=obs)
                actions.append(act)
        
        return torch.stack(actions)
    
    def get_action(self, ob):

        ob = TensorUtils.to_tensor(ob)
        ob = TensorUtils.to_batch(ob)
        ob = TensorUtils.to_device(ob, self.policies[0].policy.device)
        ob = TensorUtils.to_float(ob)
        
        self.actions = self.get_uncompressed_single_step_action(ob).cpu().detach().numpy()
        self.action_already_generated = True
        
        return np.mean(self.actions, axis=0)[0]

    def evaluate_trajectory(self, obs_np):
        
        keys = ["lang_emb", "robot0_eye_in_hand_image", "robot0_agentview_left_image", "robot0_agentview_right_image", 'robot0_eef_pos', 'robot0_base_pos', 'robot0_gripper_qpos', 'robot0_base_quat', 'robot0_eef_quat']
        intv_results = []
        assert "lang_emb" in obs_np, "Need language embedding in observation dict" 
        obs_buffer = []
        for i in range(len(obs_np[keys[0]])-self.seq_length):
            obs_input = {}
                
            for key in keys:
                obs_input[key] = obs_np[key][i:i+self.seq_length]
                if "image" in key:
                    obs_input[key] = process_image(obs_input[key])
                    obs_input[key] = obs_input[key].squeeze(0)
            self.get_action(obs_input)
            obs_buffer.append(obs_input)
            intv = self.human_intervene(obs_buffer)
            intv_results.append(intv)
            
        return intv_results, {"ens_score": self._value_history}


    def above_threshold(self, value):
        return value >= self.threshold

    def metric(self):
        return "Action Variance: "

    def horizon(self):
        return self.rollout_horizon

    def human_intervene(self, obs_buffer): # of the current last observation
        
        if len(obs_buffer) < self.seq_length: # not enough history yet, always good
            return False
        else:
            obs_np = obs_buffer[-1].copy()
            
            if self.shadowing_node:
                obs_np = process_shadowing_mode(obs_np)

            # Deal with ToolHang potentially
            obs_np.pop('frame_is_assembled', None)
            obs_np.pop('tool_on_frame', None)

            var_output = self.evaluate(obs_np, self.no_gripper_action)

            self._value_history.append(var_output)

            # print(self.metric(), var_output)
            return self.above_threshold(var_output) # if greater, is error
        
    def reset(self):
        for policy in self.policies:
            policy.start_episode()

        self._value_history = []
        
class PATO(ErrorDetector):
    def __init__(self, checkpoints, vae_goal_th, ensemble_th):
        
        super(PATO, self).__init__()
        
        self.vae_goal_detector = VAEGoal(checkpoints[0], vae_goal_th)
        # self.vae_goal_detector.shadowing_node = True # hardcode now
        self.vae_goal_th = vae_goal_th
        self.ensemble_th = ensemble_th

        assert len(checkpoints[1:]) >=3, "Need 3 checkpoints for ensemble"
        # for checkpoint in checkpoints[1:]:
            # assert "bs_sampling_T_" in checkpoint, "Need bootstrap checkpoint"
        
        self.ens_detector = Ensemble(checkpoints[1:], ensemble_th) 
        # self.ens_detector.shadowing_node = True # hardcode now

        self._value_history = {"vae_intv": [], "ens_intv": []}
        
    def evaluate_trajectory(self, obs_buffer):
        
        vae_intv, vae_score = self.vae_goal_detector.evaluate_trajectory(obs_buffer)
        ens_intv,  ens_score = self.ens_detector.evaluate_trajectory(obs_buffer)

        self._value_history["vae_intv"] = self.vae_goal_detector._value_history
        self._value_history["ens_intv"] = self.ens_detector._value_history
        
        return {"vae_score": np.array(vae_score["vae_score"]), "ens_score": np.array(ens_score["ens_score"]), "vae_intv": np.array(vae_intv), "ens_intv": np.array(ens_intv)}
    
    def reset(self):
        self.vae_goal_detector.reset()
        self.ens_detector.reset()

        self._value_history = {"vae_intv": [], "ens_intv": []}


class ThriftyDAggerED(ErrorDetector):
    def __init__(self, checkpoints, q_th, ensemble_th):

        super(ThriftyDAggerED, self).__init__()

        policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=checkpoints[0])
        self.policy = policy.policy
        self.q_th = q_th
        self.ensemble_th = ensemble_th
        self.seq_len = 10

        assert len(checkpoints) == 4 # 1 + 5

        self.policy.set_policy(checkpoints=checkpoints[1:])
        
        self.actions = None
        self.action_already_generated = False
        
        self._value_history = {"ens_intv": [], "q_intv": [], "ens_val": [], "q_val": []}

    def human_intervene(self, obs_buffer):
        ob = obs_buffer[-1]
       
        if self.shadowing_node:
            ob = process_shadowing_mode(ob) 

        #self.get_action(copy.deepcopy(ob))
 
        ob = TensorUtils.to_tensor(ob)
        # ob = TensorUtils.to_batch(ob)
        ob = TensorUtils.to_device(ob, self.policy.device)
        ob = TensorUtils.to_float(ob)
        
        assert self.action_already_generated, "Call get_action before human_intervene"
        self.action_already_generated = False
        action = np.mean(self.actions, axis=0)[0]
        
        action = TensorUtils.to_tensor(action)
        action = TensorUtils.to_device(action, self.policy.device)
        action = TensorUtils.to_float(action)
        
        ensemble_variance = np.mean(np.square(np.std(self.actions, axis=0)))
        q_val = self.policy.get_q_safety(ob, action) 
        q_val = q_val[-1] # pick the last one

        self._value_history["ens_val"].append(ensemble_variance)
        self._value_history["q_val"].append(q_val)
        self._value_history["ens_intv"].append(ensemble_variance > self.ensemble_th)
        self._value_history["q_intv"].append(q_val < self.q_th)

        print("\033[93m Q value: {} \033[0m".format(q_val))
        print("\033[93m Ensemble variance: {} \033[0m".format(ensemble_variance))

        if ensemble_variance > self.ensemble_th:
            # print("Ensemble intervene")
            return True
        if q_val < self.q_th:
            # print("Q val intervene")
            return True
        return False

    def get_action(self, ob):

        ob = TensorUtils.to_tensor(ob)
        ob = TensorUtils.to_batch(ob)
        ob = TensorUtils.to_device(ob, self.policy.device)
        ob = TensorUtils.to_float(ob)
        
        self.actions = self.policy.get_uncompressed_single_step_action(ob).cpu().detach().numpy()
        self.action_already_generated = True
        
        return np.mean(self.actions, axis=0)[0]
    
    def evaluate_trajectory(self, obs_np):
        
        key_choices = list(obs_np.keys())
        keys = ["lang_emb", "robot0_eye_in_hand_image", "robot0_agentview_left_image", "robot0_agentview_right_image", 'robot0_eef_pos', 'robot0_base_pos', 'robot0_gripper_qpos', 'robot0_base_quat', 'robot0_eef_quat']
        intv_results = []
        assert "lang_emb" in obs_np, "Need language embedding in observation dict" 
            
        for i in range(len(obs_np[keys[0]])-self.seq_len):
            obs_input = {}
                
            for key in keys:
                obs_input[key] = obs_np[key][i:i+self.seq_len]
                if "image" in key:
                    obs_input[key] = process_image(obs_input[key])
                    obs_input[key] = obs_input[key].squeeze(0)
            self.get_action(obs_input)
            intv = self.human_intervene([obs_input])
            intv_results.append(intv)
            
        return intv_results
           
    def reset(self):

        ens_policies = self.policy.policy.policies
        for policy in ens_policies:
            policy.start_episode()

        self._value_history = {"ens_intv": [], "q_intv": [], "ens_val": [], "q_val": []}

