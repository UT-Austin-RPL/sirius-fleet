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
from scipy.special import softmax

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence


# from sklearn import svm

from robomimic.algo.algo import RolloutPolicy

def get_obs_at_idx(obs, i):
    d = dict()
    for key in obs:
        d[key] = obs[key][i]
    return d

def process_shadowing_mode(obs):
    for key in obs:
        if "image" in key or "rgb" in key:
            obs[key] = ObsUtils.process_obs(obs[key], obs_modality='rgb')
    return obs

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
        if "image" in k or "rgb" in k:
            obs[k] = ObsUtils.process_obs(obs[k], obs_modality='rgb')
    return obs

class ErrorDetector:
    def __init__(self):
        # shadowing: if human shadowing using recorded obs, set to True
        self.shadowing_node = False
        self.frame_list = []
        self.pred_future = True
        self.n_future = 20
        pass

    def evaluate(self, obs):
        assert NotImplementedError
        
    def plot_demo(self, demo, history=None, name="test", title="Prediction"):
        print("save plot to ", name)
        fig, ax = plt.subplots(figsize=(10, 5))
        intv_labels = np.array(demo["intv_labels"]) 
        plt.bar(np.arange(len(intv_labels)), 10*(intv_labels==1), alpha=0.5, width=1.0, color="#94D2BD")
        ax.set_facecolor("#e5e5e5")
        ax.yaxis.grid(True)
        if history is not None:
            plt.plot(history, color="#03045E")
            
        if isinstance(self, SiriusFleet_OOD):
            ax.set_ylim(0.0, 1.5*self.threshold)
            plt.plot(self.threshold * np.ones(len(intv_labels)), color="#EE9B00", linestyle="--")
            
            plt.legend(["Predicted Distance to Centroid", "Distance Threshold",
                "Human Intervention"], loc="lower right", fontsize=12)
            ax.set_xlabel("Timestep", fontsize=12, )
            ax.set_ylabel("Distance", fontsize=12, )
            ax.set_title(title, fontsize=12)
            
            fig.savefig(name+".pdf", bbox_inches='tight')
            fig.clf()
            
        elif isinstance(self, SiriusFleet_Failure):
            ax.set_ylim(0.0, 1.0)
            
            plt.legend(["Predicted Failure Probability", "Human Intervention"], loc="lower right", fontsize=12)
            ax.set_xlabel("Timestep", fontsize=12, )
            ax.set_ylabel("Failure Probability", fontsize=12, )
            ax.set_title(title, fontsize=12)
            
            fig.savefig(name+"_fail.pdf", bbox_inches='tight')
            fig.clf()
        
        plt.savefig(name)
        plt.clf()
                
        plt.savefig(name)
        plt.clf()
        
    def get_current_embeddings(self, batch):
        with torch.no_grad():
            if self.pred_future:
                embeddings = self.policy.nets['policy'].imagine(batch, self.n_future)
            else:
                embeddings = self.policy.nets['policy'].get_curr_latent_features(batch)
                
        return embeddings
        
        
    def human_intervene_frame(self, frame):
        obs, action = frame['obs'], frame['actions']
        obs, action = copy.deepcopy(obs), copy.deepcopy(action)
        frame_copy = {'obs':obs, 'actions':action}
        self.frame_list.append(frame_copy)
        fail = False
        fail = self.human_intervene(self.frame_list)
        return fail

    def evaluate_trajectory(self, demo, verbose=False):
        try:
            traj_len = len(demo['obs']['robot0_agentview_left_image'])
        except:
            traj_len = len(demo['obs']['agentview_rgb'])
        start_time = time.time()
        for i in range(traj_len):
            obs = get_obs_at_idx(demo['obs'], i)
            result = self.human_intervene_frame({'obs':obs, 'actions':demo['actions'][i]})
            if verbose:
                detector_type = type(self).__name__
                if detector_type == "SiriusFleetCombined":
                    print(f"frame {i}, {detector_type}: OOD:{result[0]}, dist:{result[2]:.2f}, Failure:{result[1]}, prob:{result[3]:.2f}    Ground Truth: {demo['intv_labels'][i]}        ", end="\r")
                else:
                    print(f"frame {i}, {detector_type}:{result}             ", end="\r")
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.1f}s, fps: {traj_len / (end_time - start_time):.1f}" + " " * 20)
            
    def process_batch(self, batch):
        if type(batch) is list:
            batch = batch[-10:]
        batch = TensorUtils.list_of_flat_dict_to_dict_of_list(batch)
        batch['obs'] = TensorUtils.list_of_flat_dict_to_dict_of_list(batch['obs'])
        
        key_pop_lst = []
        for key in batch['obs'].keys():
            # pop key if key not image
            if "image" not in key and "rgb" not in key:
                key_pop_lst.append(key)
        for key in key_pop_lst: 
            batch['obs'].pop(key)
        
        for key in batch['obs']:
            batch['obs'][key] = np.array(batch['obs'][key])
            
        batch['actions'] = np.array(batch['actions'])
        batch = process_batch(batch, self.policy)
        batch['obs'] = downscale_img(batch['obs'], self.policy)
        
        return batch

    def above_threshold(self, value):
        assert NotImplementedError
    
    def reset(self):
        self.frame_list = []

def train_pca_kmeans(embeddings, n_clusters=500, n_components=100, save_path=None):
    from sklearn.decomposition import PCA
    from sklearn.cluster import MiniBatchKMeans
    pca = PCA(n_components=n_components)
    print("Fitting PCA")
    pca.fit(embeddings)
    pca_embeddings = pca.transform(embeddings)
    print("Fitting KMeans")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=2048, max_iter=100, n_init=1, verbose=1, random_state=2, max_no_improvement=20, reassignment_ratio=0.01)
    kmeans.fit(pca_embeddings)
    print("PCA and KMeans trained, Kmeans inertia/datasize: ", kmeans.inertia_/len(embeddings))
    if save_path is not None:
        np.save(save_path.replace(".npy", "_kmeans.npy"), kmeans)
        np.save(save_path.replace(".npy", "_pca.npy"), pca)
    return pca, kmeans

def load_pca_kmeans(save_path):
    pca = np.load(save_path.replace(".npy", "_pca.npy"), allow_pickle=True).item()
    kmeans = np.load(save_path.replace(".npy", "_kmeans.npy"), allow_pickle=True).item()
    return pca, kmeans
        
class SiriusFleet_OOD(ErrorDetector):
    def __init__(self, 
                 checkpoint, 
                 threshold,  # None for kmeans.inertia_
                 demos_embedding_path, 
                 eval_method, 
                 num_future=20, 
                 pred_future=True, 
                 dist_metric = "nearest_neighbor", # nearest_neighbor, kmeans
                 train_kmeans=False,
                 kmeans_embedding_path=None,
                 compile=False,
                 percentile=90,
                 ):
        
        super(SiriusFleet_OOD, self).__init__()
        
        self.rollout_policy = policy_from_checkpoint(ckpt_path=checkpoint)[0] #.nets["policy"]
        self.policy = self.rollout_policy.policy
        self.policy.nets['policy'].eval()
        if compile:
            print("compiling imagine for faster")
            self.policy.nets['policy'].imagine = torch.compile(self.policy.nets['policy'].imagine)
            print("imagin compiled")
        self.threshold = threshold
        self.seq_length = 10
        self.eval_method = eval_method
        assert eval_method in ["mean", "first", "last", "weighted"]
        self.num_future = num_future

        self.percentile = percentile

        kmeans_embedding_path = kmeans_embedding_path if kmeans_embedding_path is not None else demos_embedding_path

        ckpt_dict = policy_from_checkpoint(ckpt_path=checkpoint)[1]
        self.action_normalization_stats = None
        if "action_normalization_stats" in ckpt_dict:
            self.action_normalization_stats = ckpt_dict["action_normalization_stats"]
        self.config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)

        # Load embeddings and latent_lst
        self.demos_embedding = np.load(demos_embedding_path, 
                                       allow_pickle=True)
        for i in range(2):
            self.demos_embedding = np.squeeze(self.demos_embedding, 1)
            
        self.dist_metric = dist_metric
        train_embeddings, eval_embeddings = self.split_train_eval(self.demos_embedding)
        
        if dist_metric == "nearest_neighbor":
            import faiss
            train_embeddings /= np.linalg.norm(train_embeddings, axis=1)[:, np.newaxis]
            dimension = train_embeddings.shape[-1]
            index = faiss.IndexFlatL2(dimension)
            res = faiss.StandardGpuResources() # use a single GPU
            self.index = faiss.index_cpu_to_gpu(res, 0, index)
            self.index.add(train_embeddings)
        elif dist_metric == "kmeans":
            if os.path.exists(os.path.join(kmeans_embedding_path.replace(".npy", "_kmeans.npy"))) and not train_kmeans:
                self.pca, self.kmeans = load_pca_kmeans(kmeans_embedding_path)
                print("PCA and KMeans loaded, Kmeans inertia/datasize: ", self.kmeans.inertia_/len(train_embeddings))
            else:
                self.pca, self.kmeans = train_pca_kmeans(train_embeddings, save_path=kmeans_embedding_path)
            if self.threshold is None:
                eval_pca_embeddings = self.pca.transform(eval_embeddings)
                label = self.kmeans.predict(eval_pca_embeddings)
                dis = np.linalg.norm(eval_pca_embeddings - self.kmeans.cluster_centers_[label], axis=1)
                self.threshold = np.percentile(dis, self.percentile)
                
                for i in range(1, 20):
                    threshold = np.percentile(dis, 80 + i)
                    print("threshold for ", 80 + i, " percentile: ", threshold)
            
                print(f"Using threshold: {self.threshold:.5f}") 
        else:
            raise ValueError("dist_metric must be nearest_neighbor or kmeans")

        self._img_embedding_history = []
        self._value_history = []

        # for offline rollouts
        self.pred_future = pred_future
        
    def split_train_eval(self, embeddings):
        chunk_len = 200
        remainder = embeddings.shape[0] % chunk_len
        chunk_num = embeddings.shape[0] // chunk_len
        
        if remainder == 0:
            chunked_embeddings = embeddings.reshape(chunk_num, chunk_len, embeddings.shape[-1])
        else:
            chunked_embeddings = embeddings[:-remainder].reshape(chunk_num, chunk_len, embeddings.shape[-1])
        indices = np.arange(chunk_num)
        #use fix seed to shuffle
        
        rs = RandomState(0)
        rs.shuffle(indices)
        train_chunk_num = int(0.9 * chunk_num)
        train_indices, eval_indices = indices[:train_chunk_num], indices[train_chunk_num:]
        train_chunks = chunked_embeddings[train_indices]
        eval_chunks = chunked_embeddings[eval_indices]
        train_embeddings = train_chunks.reshape(-1, embeddings.shape[-1])
        
        if remainder == 0:
            eval_embeddings = eval_chunks.reshape(-1, embeddings.shape[-1])
        else:
            eval_embeddings =  np.concatenate([eval_chunks.reshape(-1, embeddings.shape[-1]), embeddings[-remainder:]])
        
        return train_embeddings, eval_embeddings
        

    # For online rollouts in HITL loop, obs should be in the form of a dictionary, and the rgb should be in the shape of [T, C, H, W]
    def evaluate(self, imagined_embeddings):
        
        imagined_embeddings = imagined_embeddings.cpu().numpy()
                
        if self.dist_metric == "nearest_neighbor":
            nearest_neighbor_distances = self._compute_nearest_neighbor_distance(imagined_embeddings, 
                                                                            self.demos_embedding)
            dist = nearest_neighbor_distances
            
        else:
            imagined_embeddings_shape = imagined_embeddings.shape
            pca_embeddings = self.pca.transform(imagined_embeddings.reshape(-1, imagined_embeddings_shape[-1]))
            label = self.kmeans.predict(pca_embeddings)
            dist = np.linalg.norm(pca_embeddings - self.kmeans.cluster_centers_[label], axis=1)
            dist = dist.reshape(imagined_embeddings_shape[:-1])

        if self.eval_method == "first":
            dist = dist[:,0].mean()
        elif self.eval_method == "last":
            dist = dist[:,-1].mean()
        elif self.eval_method == "mean":
            dist = np.mean(dist)
        elif self.eval_method == "weighted":
            dist = (dist*np.arange(dist.shape[1])/np.sum(np.arange(dist.shape[1]))).mean(axis=0).sum()
        # print(dist)
        return dist

    def human_intervene(self, frame_list=None, batch=None, embedding=None, return_dist=True): # of the current last observation    
        assert frame_list is not None or batch is not None, "Either frame_list or batch must be provided"
        
        # to hardcode for obs_buffer in hitl script
        if type(frame_list) is list:
            if len(frame_list) < 10:
                self._value_history.append(0)
                if return_dist:
                    return False, 0
                else:
                    return False
        if batch is None:
            batch = self.process_batch(frame_list)
            
        if self.shadowing_node:
            batch['obs'] = process_shadowing_mode(batch['obs'])
            
        if embedding is None:
            embedding = self.get_current_embeddings(batch)
        
        nearest_neighbor_distance = self.evaluate(embedding)
        
        self._value_history.append(nearest_neighbor_distance)
        
        
        if return_dist:
            return self.above_threshold(nearest_neighbor_distance), nearest_neighbor_distance
        
        return self.above_threshold(nearest_neighbor_distance)

    # For visualization of offline rollouts
    def evaluate_batch_trajectory(self, latent_lst, use_img_eval=False):
        # Takes in latent_lst for now, rather than raw obs
        # Compute nearest neighbor distances
        if self.dist_metric == "nearest_neighbor":
            nearest_neighbor_distances = self._compute_nearest_neighbor_distance(latent_lst, 
            self.demos_embedding)
            dist = nearest_neighbor_distances
        if self.pred_future or use_img_eval:
            # set first 13 to 0
            dist[:13] = 0
            
        return dist # error states are larger values

    def above_threshold(self, value):

        return value >= self.threshold  

    def _query_nn(self, query_vectors, dataset, k):
        query_vectors /= np.linalg.norm(query_vectors, axis=1)[:, np.newaxis]
        distances, indices = self.index.search(query_vectors, k)

        ave_dist_lst = []
        for i, (query_dist, query_idx) in enumerate(zip(distances, indices)):
            total_dist = 0
            for rank, (dist, idx) in enumerate(zip(query_dist, query_idx), start=1):
                total_dist += dist
            ave_dist_lst.append(total_dist / len(query_dist))

        return ave_dist_lst
    
    def _moving_average_past(self, data, window_size):
        assert window_size > 0, "Window size must be greater than 0"
        
        smoothed_data = np.zeros(len(data))
        for i in range(len(data)):
            start = max(0, i - window_size + 1)
            end = i + 1
            smoothed_data[i] = np.mean(data[start:end])
        
        return smoothed_data

    def _compute_nearest_neighbor_distance(self, latent_lst, demos_embedding):
        A,B,C = latent_lst.shape
        latent_lst = np.mean(latent_lst, axis=0)
        latent_lst = np.reshape(latent_lst, (B, C))
        ave_dist_lst = self._query_nn(latent_lst, demos_embedding, k=5)
        ave_dist_lst = self._moving_average_past(ave_dist_lst, window_size=3)

        ave_dist_lst = np.reshape(ave_dist_lst, (1, B))
        return ave_dist_lst
    
    def metric(self):
        # for plotting figure title
        if self.pred_future:
            return "OOD with Imagined Embedding"
        else:
            return "OOD with Current Embedding"
    
    def horizon(self):
        return 400
    
    def reset(self):
        
        super().reset()
        self.rollout_policy.start_episode()
        self._value_history = []

class SiriusFleet_Failure(ErrorDetector):
    def __init__(self, 
                 checkpoint, 
                 threshold, 
                 threhold_history, 
                 threshold_count,
                 eval_method, 
                 eval_idx=None,
                 use_prob=False,
                 pred_future=True,
                 compile=False,
                 use_intv=False,
                 ):
        
        super(SiriusFleet_Failure, self).__init__()
        
        self.rollout_policy = policy_from_checkpoint(ckpt_path=checkpoint)[0] #.nets["policy"]
        self.policy = self.rollout_policy.policy
        self.policy.nets['policy'].eval()
        self.threshold = threshold
        self.threhold_history = threhold_history
        self.threshold_count = threshold_count
        self.eval_method = eval_method
        self.eval_idx = eval_idx
        self.pred_future = pred_future
        self.frame_list = []
        if eval_method == "idx":
            assert eval_idx is not None
            
        self.use_prob = use_prob

        self._value_history = {"pred_prob": [], "pred_classes": []}
        
        self.error_idx = 1 if use_intv else 2

    def evaluate(self, actions, imagined_embeddings):
        
        with torch.no_grad():
            actions = actions[:,-imagined_embeddings.shape[1]:]
        
            pred_reward_original = self.policy.nets['policy'].reward_predict(imagined_embeddings, actions).cpu().numpy()
            # breakpoint()

        if self.eval_method == "idx":
            # take the given index from eval_method
            pred_reward = pred_reward_original[:,self.eval_idx] 
        else:
            # take the mean
            assert self.eval_method == "mean"
            
        # test: see the distribution of the reward
        pred_reward_test = np.mean(pred_reward, axis=1)
        pred_prob_test = softmax(pred_reward_test, axis=-1)
        pred_classes_test = np.argmax(pred_prob_test, axis=-1)
        
        pred_reward_test_time = np.mean(pred_reward_original, axis=0)
        pred_prob_test_time = softmax(pred_reward_test_time, axis=-1)
        pred_classes_test_time = np.argmax(pred_prob_test_time, axis=-1)
        
        # first average across futures
        pred_reward = np.mean(pred_reward, axis=0)
        
        # then average across different timesteps
        pred_prob = np.mean(pred_reward, axis=0)

        # lastly, find the classes
        pred_prob = softmax(pred_prob, axis=-1)
        pred_classes = np.argmax(pred_prob, axis=-1)
        
        pred_prob = np.expand_dims(pred_prob, 0)
        pred_classes = np.expand_dims(pred_classes, 0)

        self._value_history["pred_prob"].append(pred_prob)
        self._value_history["pred_classes"].append(pred_classes)

        return pred_prob, pred_classes
    

    def above_threshold(self, value, use_prob_eval=False):
        raise NotImplementedError("Not used here")
        
    def metric(self):
        # for plotting figure title
        return "Failure Probablity"
    
    def horizon(self):
        return 400

    def human_intervene(self, frame_list=None, batch=None, embedding=None, return_prob=True): # of the current last observation
        assert frame_list is not None or batch is not None, "Either frame_list or batch must be provided"
        
        if type(frame_list) is list and len(frame_list) < 10:
            self._value_history["pred_prob"].append(np.array([[1, 0, 0]]))
            self._value_history["pred_classes"].append(np.array([0]))
            if return_prob:
                return False, 0
            return False # no intervention
        
        if batch is None:
            batch = self.process_batch(frame_list)
                
        if self.shadowing_node:
            batch = process_shadowing_mode(batch)
            
        if embedding is None:
            embedding = self.get_current_embeddings(batch)
            
        if embedding.shape[0] > 1: # sampling multiple futures, expand action dimension
            
            # print(batch['actions'].shape)
            batch['actions'] = torch.tile(batch['actions'], (embedding.shape[0], 1, 1))

        pred_reward, pred_classes = self.evaluate(batch['actions'], embedding)
        # needs to be true for last 10 steps
        
        considered_segment_classes = np.array(self._value_history["pred_classes"][-self.threhold_history:])
        considered_segment_prob = np.array(self._value_history["pred_prob"][-self.threhold_history:])
        # considered_segment_prob = softmax(considered_segment_reward, axis=-1)
        
        if self.use_prob:
            considered_segment = considered_segment_prob[...,self.error_idx]
        else:
            considered_segment = (considered_segment_classes == self.error_idx)
            
        if sum(considered_segment >= self.threshold) >= self.threshold_count:
            if return_prob:
                return True, considered_segment_prob[...,self.error_idx].mean()
            return True
        else:
            if return_prob:
                return False, considered_segment_prob[...,self.error_idx].mean()
            return False
        
    def reset(self):
        
        super().reset()
        
        self.rollout_policy.start_episode()
        self._value_history = {"pred_prob": [], "pred_classes": []}
 
 
class SiriusFleetCombined(ErrorDetector):
    def __init__(
        self,
        ood_detector,
        failure_detector,
    ):
        super(SiriusFleetCombined, self).__init__()
        self.ood_detector = ood_detector
        self.failure_detector = failure_detector
        self._value_history = {"ood_intv": [], "failure_intv": [], "failure_prob": [], "ood_score": []}
        
    def evaluate(self, obs):
        raise NotImplementedError("Not used here")
    
    def human_intervene(self, frame_list, share_embedding=True, return_in_detail=True):
        if len(frame_list) < 10:
            if return_in_detail:
                return False, False, 0, 0
            return False
        batch = self.ood_detector.process_batch(frame_list)
        
        if share_embedding:
            embedding = self.ood_detector.get_current_embeddings(batch)
        else:
            embedding = None
        ood_result, dist = self.ood_detector.human_intervene(batch=batch, embedding=embedding)
        failure_result, prob = self.failure_detector.human_intervene(batch=batch, embedding=embedding)
        self._value_history["ood_intv"].append(ood_result)
        self._value_history["ood_score"].append(dist)
        self._value_history["failure_intv"].append(failure_result)
        self._value_history["failure_prob"].append(prob)
        

        if return_in_detail:
            return ood_result, failure_result, dist, prob
        else:
            return ood_result or failure_result
    
    def reset(self):
        
        self.ood_detector.reset()
        self.failure_detector.reset()
        self._value_history = {"ood_intv": [], "failure_intv": [], "failure_prob": [], "ood_score": []}
        
        
        
