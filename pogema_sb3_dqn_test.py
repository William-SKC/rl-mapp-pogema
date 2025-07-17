from stable_baselines3 import DQN
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MultiInputPolicy
from pogema import GridConfig, pogema_v0, AnimationMonitor, AnimationConfig

import torch
print(torch.__version__)

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

# 2) Wrap in AnimationMonitor
env = AnimationMonitor(env)

# 3) Load your trained agent
model = DQN.load("./dqn_pogema_model", env=env)

# 4) Roll out one episode, collecting frames internally
obs, _info = env.reset()
print(obs)
done = False
step = 0
while not done and step < env.unwrapped.grid_config.max_episode_steps:
    print('Step:', step)
    print(obs['obstacles'].shape, obs['xy'], obs['target_xy'])

    action, _ = model.predict(obs, deterministic=True)
    print('Action:', action)
    obs, rewards, terminations, truncations, info = env.step(action)

    # Check if done
    done = terminations or truncations
    step+=1

# Save the recorded animation to an SVG file
env.save_animation('dqn.svg')
egocentric_idx = 0
env.save_animation(f'dqn_ego{egocentric_idx}.svg',
                               AnimationConfig(egocentric_idx=egocentric_idx))
print("Animation saved to dqn.svg")