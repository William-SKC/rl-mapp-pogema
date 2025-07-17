import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MultiInputPolicy
from pogema import GridConfig, pogema_v0

# 1) Create the POGEMA env
env = pogema_v0(
    GridConfig(
        size=20,
        density=0.1,
        num_agents=1,
        max_episode_steps=50,
        integration="gymnasium",
        observation_type="POMAPF",
    )
)
dqn_agent = DQN(
    MultiInputPolicy,
    env,
    verbose=1,
    tensorboard_log="./dqn_pogema_tensorboard/"
)

dqn_agent.learn(
    total_timesteps=5_000_000,
    log_interval=1000,
    tb_log_name="baseline"
)
dqn_agent.save("dqn_pogema_model")