import numpy as np

from ..interface import QLearningAlgoProtocol, StatefulTransformerAlgoProtocol
from ..types import GymEnv


from llm.algo.rule_base_2.gfootball import agent_dict as rule_base_2_agent_dict
from llm.utils.obs2text import observation_to_text_human, observation_to_text_raw, get_zons_240, format_code, imaginary_data_observation

from gfootball.env.wrappers import Simple115StateWrapper_ball_owned_player
from llm.utils.obs2text import imaginary_data_observation, imaginary_data_to_vector
__all__ = [
    "evaluate_qlearning_with_environment",
    "evaluate_transformer_with_environment",
    "evaluate_qlearning_with_img_obs_environment",
]

HORIZON = 1500

def rule_based_postprocess(wrapper_obs, obs, action):
    rule_policy_dic = rule_base_2_agent_dict(obs.copy(), ret_dict=True)
    rule_action = rule_policy_dic['action'].value
    team_owner = np.where(wrapper_obs[94:97] == 1)[0][0] # 0: no one, 1: left team, 2: right team
    team_owner = np.where(wrapper_obs[94:97] == 1)[0][0] # 0: no one, 1: left team, 2: right team
    ball_zone = get_zons_240(wrapper_obs[88], wrapper_obs[89])
    active_player_index = np.where(wrapper_obs[97:108] == 1)[0][0]
    active_player_zone = get_zons_240(wrapper_obs[2*active_player_index], wrapper_obs[2*active_player_index+1])
    if team_owner != 1 and active_player_zone == ball_zone:
        action = rule_action
    if action != rule_action:
        # if action == 16: # sliding
        #     action = rule_action
        if rule_action == 13: # sprint
            action = rule_action
        elif rule_action == 17: # dribble
            action = rule_action
        elif rule_action == 12: # shot
            action = rule_action
        # elif action == 0: # idle
        #     action = rule_action
    return action

def evaluate_qlearning_with_img_obs_environment(
    algo: QLearningAlgoProtocol,
    env_list: GymEnv,
    n_trials: int = 10,
    epsilon: float = 0.0,
    update_stack_obs=None,
    stack_obs_len=None
) -> float:
    """Returns average environment score.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.utility import evaluate_with_environment

        env = gym.make('CartPole-v0')

        cql = CQL()

        mean_episode_return = evaluate_with_environment(cql, env)


    Args:
        alg: algorithm object.
        env: gym-styled environment.
        n_trials: the number of trials.
        epsilon: noise factor for epsilon-greedy policy.

    Returns:
        average score.
    """
    

    res_dict = {}
    for env in env_list:
        episode_rewards = []
        wrapper_func = Simple115StateWrapper_ball_owned_player
        wrapper = wrapper_func(env)
        stack_obs = np.zeros(stack_obs_len)
        for n in range(n_trials):
            step = 0
            raw_obs = env.reset()
            episode_reward = 0.0
            print("evaluation trial", n, "in env", env.env_name)
            while True:
                wrap_obs = wrapper.observation(raw_obs)
                observation = imaginary_data_observation(wrap_obs[0], raw_obs[0], step, ret_type='vector', TODO_missing=True)
                # take action
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    if isinstance(observation, np.ndarray):
                        observation = np.expand_dims(observation, axis=0)
                    elif isinstance(observation, (tuple, list)):
                        observation = [
                            np.expand_dims(o, axis=0) for o in observation
                        ]
                    else:
                        raise ValueError(
                            f"Unsupported observation type: {type(observation)}"
                        )
                    stack_obs =update_stack_obs(observation[0], stack_obs)
                    action = algo.predict(np.expand_dims(stack_obs, axis=0))[0]
                action = rule_based_postprocess(wrap_obs[0], raw_obs[0], action)
                raw_obs, reward, done, info = env.step(action)
                episode_reward += float(reward)
                step += 1
                if done or step > HORIZON:
                    break
            episode_rewards.append(episode_reward)
            
        res_dict.update({
                f"rew_{env.env_name}": float(np.mean(episode_rewards)), 
                f"win_{env.env_name}": float(np.mean(np.array(episode_rewards)>0)), 
                f"draw_{env.env_name}": float(np.mean(np.array(episode_rewards)==0)), 
                f"lose_{env.env_name}": float(np.mean(np.array(episode_rewards)<0))})
    return res_dict


def evaluate_qlearning_with_environment(
    algo: QLearningAlgoProtocol,
    env_list: GymEnv,
    n_trials: int = 10,
    epsilon: float = 0.0,
) -> float:
    """Returns average environment score.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.utility import evaluate_with_environment

        env = gym.make('CartPole-v0')

        cql = CQL()

        mean_episode_return = evaluate_with_environment(cql, env)


    Args:
        alg: algorithm object.
        env: gym-styled environment.
        n_trials: the number of trials.
        epsilon: noise factor for epsilon-greedy policy.

    Returns:
        average score.
    """
    res_dict = {}
    for env in env_list:
        episode_rewards = []
        wrapper_func = Simple115StateWrapper_ball_owned_player
        wrapper = wrapper_func(env)
        
        for n in range(n_trials):
            step = 0
            observation = env.reset()
            episode_reward = 0.0
            print("evaluation trial", n, "in env", env.env_name)
            while True:
                observation = wrapper.observation(observation)
                # take action
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    if isinstance(observation, np.ndarray):
                        observation = np.expand_dims(observation, axis=0)
                    elif isinstance(observation, (tuple, list)):
                        observation = [
                            np.expand_dims(o, axis=0) for o in observation
                        ]
                    else:
                        raise ValueError(
                            f"Unsupported observation type: {type(observation)}"
                        )
                    action = algo.predict(observation)[0]

                observation, reward, done, info = env.step(action)
                episode_reward += float(reward)
                step += 1
                if done or step > HORIZON:
                    break
            episode_rewards.append(episode_reward)
        res_dict.update({
                f"rew_{env.env_name}": float(np.mean(episode_rewards)), 
                f"win_{env.env_name}": float(np.mean(np.array(episode_rewards)>0)), 
                f"draw_{env.env_name}": float(np.mean(np.array(episode_rewards)==0)), 
                f"lose_{env.env_name}": float(np.mean(np.array(episode_rewards)<0))})
    return res_dict


def evaluate_transformer_with_environment(
    algo: StatefulTransformerAlgoProtocol,
    env: GymEnv,
    n_trials: int = 10,
) -> float:
    """Returns average environment score.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.utility import evaluate_with_environment

        env = gym.make('CartPole-v0')

        cql = CQL()

        mean_episode_return = evaluate_with_environment(cql, env)


    Args:
        alg: algorithm object.
        env: gym-styled environment.
        n_trials: the number of trials.

    Returns:
        average score.
    """
    episode_rewards = []
    for _ in range(n_trials):
        algo.reset()
        observation, reward = env.reset()[0], 0.0
        episode_reward = 0.0

        while True:
            # take action
            action = algo.predict(observation, reward)

            observation, _reward, done, truncated, _ = env.step(action)
            reward = float(_reward)
            episode_reward += reward

            if done or truncated:
                break
        episode_rewards.append(episode_reward)
    return float(np.mean(episode_rewards))
