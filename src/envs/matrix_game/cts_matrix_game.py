from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
import numpy as np
from gym.spaces import Box


class Matrixgame(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):
        # Unpack arguments from sacred
        args = kwargs["env_args"]
        if isinstance(args, dict):
            args = convert(args)

        # Define the agents and actions
        self.n_agents = 2
        self.n_actions = 1
        self.episode_limit = 1

        self.state = np.ones(5)
        self.action_space = [Box(low=-1, high=1,shape=(1,)), Box(low=-1, high=1,shape=(1,))]

    def _reward(self, x, y):
        r = -0.1 * (x**2 + y**2)
        # if np.sqrt(0.5**2 - x**2 - y**2) > 0:
        #     r += np.sqrt(0.5**2 - x**2 - y**2)
        if x > 0 and y > 0 and abs(x-y) < 0.01:
            r += max(0.1, 0.00001/(abs(x - y)+0.00001)) + (x + y)
            # r += (x + y)
        return r

    def reset(self):
        """ Returns initial observations and states"""
        return self.state, self.state

    def step(self, actions):
        """ Returns reward, terminated, info """
        reward = self._reward(actions[0], actions[1])

        info = {}
        terminated = True
        info["episode_limit"] = False
        info["x"] = actions[0]
        info["y"] = actions[1]

        return reward, terminated, info

    def get_obs(self):
        return [self.state for _ in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise NotImplementedError

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.get_state_size()

    def get_state(self):
        return self.state

    def get_state_size(self):
        """ Returns the shape of the state"""
        return len(self.state)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(self.n_actions)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def get_stats(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit,
                    "action_spaces": self.action_space,
                    "actions_dtype": np.float32,
                    "normalise_actions": False
                    }
        return env_info