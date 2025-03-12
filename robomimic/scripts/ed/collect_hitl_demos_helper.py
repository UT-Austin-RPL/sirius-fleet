
class RandomPolicy:
    def __init__(self, env):
        self.env = env
        self.low, self.high = env.action_spec

    def get_action(self, obs):
        return np.random.uniform(self.low, self.high) / 2

class TrainedPolicy:
    def __init__(self, checkpoint):
        from robomimic.utils.file_utils import policy_from_checkpoint
        self.policy = policy_from_checkpoint(ckpt_path=checkpoint)[0]
        #self.policy.policy.nets["policy"].low_noise_eval = False

    def get_action(self, obs):
        obs = copy.deepcopy(obs)
        di = obs
        postprocess_visual_obs = True

        ret = {}
        for k in di:
            if "image" in k:
                ret[k] = di[k][::-1]
                ret[k] = ObsUtils.process_obs(ret[k], obs_modality='rgb')
        obs.update(ret)
        pop_keys = []
        for k in obs:
            if isinstance(obs[k], np.float64):
                pop_keys.append(k)
        for k in pop_keys:
            obs.pop(k, None)

        obs.pop('frame_is_assembled', None)
        obs.pop('tool_on_frame', None)
        return self.policy(obs)

    def get_dist(self, obs):
        a, dist = self.policy.get_action_with_dist(obs)
        return a, dist

def is_empty_input_spacemouse(action):
    empty_input = np.array([0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000])
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
    # If the current grasp is active (1) and last grasp is not (-1) (i.e.: grasping input just pressed),
    # toggle arm control and / or camera viewing angle if requested
    if last_grasp < 0 < grasp:
        if args.switch_on_grasp:
            args.arm = "left" if args.arm == "right" else "right"
        if args.toggle_camera_on_grasp:
            cam_id = (cam_id + 1) % num_cam
            env.viewer.set_camera(camera_id=cam_id)
    # Update last grasp
    last_grasp = grasp

    if is_v1:
        env_action_dim = env.action_dim
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

class Renderer:
    def __init__(self, env, render_onscreen):
        self.env = env
        self.render_onscreen = render_onscreen

        if (is_v1 is False) and self.render_onscreen:
            self.env.viewer.set_camera(camera_id=2)

    def render(self, obs):
        if is_v1:
            vis_env = self.env.env
            robosuite_env = self.env.env.env
            robosuite_env.visualize(vis_settings=vis_env._vis_settings)
        else:
            robosuite_env = self.env.env

        if self.render_onscreen:
            self.env.render()
        else:
            # if is_v1:
            #     img_for_policy = obs['agentview_image']
            # else:
            #     img_for_policy = obs['image']
            # img_for_policy = img_for_policy[:,:,::-1]
            # img_for_policy = np.flip(img_for_policy, axis=0)
            # cv2.imshow('img for policy', img_for_policy)

            img = robosuite_env.sim.render(height=700, width=1000, camera_name="agentview")
            img = img[:,:,::-1]
            img = np.flip(img, axis=0)
            cv2.imshow('offscreen render', img)
            cv2.waitKey(1)

        if is_v1:
            robosuite_env.visualize(vis_settings=dict(
                env=False,
                grippers=False,
                robots=False,
            ))