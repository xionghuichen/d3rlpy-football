import numpy as np

from ..interface import QLearningAlgoProtocol, StatefulTransformerAlgoProtocol
from ..types import GymEnv


from gfootball.env.wrappers import Simple115StateWrapper_ball_owned_player

__all__ = [
    "evaluate_qlearning_with_environment",
    "evaluate_transformer_with_environment",
]


def evaluate_qlearning_with_environment(
    algo: QLearningAlgoProtocol,
    env: GymEnv,
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
    episode_rewards = []
    wrapper_func = Simple115StateWrapper_ball_owned_player
    wrapper = wrapper_func(env)
    
    for _ in range(n_trials):
        step = 0
        observation = env.reset()
        episode_reward = 0.0
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
            if done or step > 3000:
                break
        episode_rewards.append(episode_reward)
    return {
            "rew": float(np.mean(episode_rewards)), 
            "win": float(np.mean(np.array(episode_rewards)>0)), 
            "draw": float(np.mean(np.array(episode_rewards)==0)), 
            "lose": float(np.mean(np.array(episode_rewards)<0))}


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
