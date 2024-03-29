import argparse

import gym

import d3rlpy

GAMMA = 0.99


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=1)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    env = gym.make(args.env)
    eval_env = gym.make(args.env)

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)
    d3rlpy.envs.seed_env(eval_env, args.seed)

    # setup algorithm
    sac = d3rlpy.algos.SACConfig(
        batch_size=256,
        gamma=GAMMA,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=3e-4,
        action_scaler=d3rlpy.preprocessing.MinMaxActionScaler(),
    ).create(device=args.gpu)

    # multi-step transition sampling
    transition_picker = d3rlpy.dataset.MultiStepTransitionPicker(
        n_steps=args.n_steps,
        gamma=GAMMA,
    )

    # replay buffer for experience replay
    buffer = d3rlpy.dataset.create_fifo_replay_buffer(
        limit=100000,
        env=env,
        transition_picker=transition_picker,
    )

    # start training
    sac.fit_online(
        env,
        buffer,
        eval_env=eval_env,
        n_steps=100000,
        n_steps_per_epoch=1000,
        update_interval=1,
        update_start_step=1000,
    )


if __name__ == "__main__":
    main()
