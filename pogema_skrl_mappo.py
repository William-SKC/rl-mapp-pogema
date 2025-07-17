import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from pogema import pogema_v0, GridConfig

from skrl.envs.wrappers.torch import wrap_env
from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model
from skrl.memories.torch import RandomMemory
from skrl.multi_agents.torch.mappo.mappo_cnn import MAPPO_CNN, MAPPO_CNN_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.schedulers.torch import KLAdaptiveRL
# from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils import set_seed

# Set reproducible seed
set_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"


# 🧠 Setup Pogema environment
env = pogema_v0(
    GridConfig(
        size=20,
        density=0.1,
        num_agents=3,
        max_episode_steps=80, # 4*size
        # on_target = 'nothing',
    ),
    render_mode=None
)

env = wrap_env(env, wrapper="pogema")  # Wrap for SKRL compatibility
obs, _ = env.reset()
print("Environment:", env)
print(env.__dict__)

print("Agents:", env.possible_agents)
print("Observation Spaces:")
for agent_id, obs_space in env.observation_spaces.items():
    print(f"  Agent {agent_id}: {obs_space}")
print("State Spaces:")
for agent_id, space in env.state_spaces.items():
    print(f"  Agent {agent_id}: {space}")

# CNN Feature Extractor
class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_shape, output_dim=256):
        super().__init__()
        c, h, w = input_shape  # e.g. (4, 11, 11)
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),  # (32, h, w)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (64, h, w)
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            flat_dim = self.cnn(dummy_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(flat_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.cnn(x)
        return self.linear(x)
    

# 🧠 Define Policy Model
class Policy(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self)

        obs_shape = observation_space.shape  # (11, 11, 4)
        self.feature_extractor = CNNFeatureExtractor(input_shape=obs_shape, output_dim=256)

        self.policy_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )


    def compute(self, inputs, role):
        features = self.feature_extractor(inputs["states"])
        return self.policy_net(features), {}

# 🧠 Define Value Model
class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)

        obs_shape = observation_space.shape
        self.feature_extractor = CNNFeatureExtractor(input_shape=obs_shape, output_dim=256)

        self.value_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def compute(self, inputs, role):
        features = self.feature_extractor(inputs["states"])
        return self.value_net(features), {}


# 🔧 Define Separate Models for Each Agent
models = {
    agent_name : {
        "policy": Policy(env.observation_spaces[agent_name], env.action_spaces[agent_name], device),
        "value": Value(env.global_state_space, env.action_spaces[agent_name], device)
    }
    for agent_name in env.possible_agents
}

# instantiate memories as rollout buffer (any memory can be used for this)
memories = {}
for agent_name in env.possible_agents:
    memories[agent_name] = RandomMemory(memory_size=1024, num_envs=env.num_envs, device=device)


# 🎛 Configure MAPPO Agent
cfg = MAPPO_CNN_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 1024  # memory_size
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 8  
cfg["discount_factor"] = 0.95
cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.01
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0
# cfg["state_preprocessor"] = RunningStandardScaler
# cfg["state_preprocessor_kwargs"] = {"size": next(iter(env.observation_spaces.values())), "device": device}
# cfg["shared_state_preprocessor"] = RunningStandardScaler
# cfg["shared_state_preprocessor_kwargs"] = {"size": next(iter(env.state_spaces.values())), "device": device}
# cfg["value_preprocessor"] = RunningStandardScaler
# cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

# ✅ Set up logging & checkpoints
cfg["experiment"]["directory"] = "runs/torch/Pogema_MAPPO_CNN"
cfg["experiment"]["write_interval"] = 50000
cfg["experiment"]["checkpoint_interval"] = 100000

# print(env.observation_spaces)
# print(env.action_spaces)
# print(env.state_spaces)

training_agent = MAPPO_CNN(
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,  
        cfg=cfg,
        observation_spaces=env.observation_spaces,
        action_spaces=env.action_spaces,
        device=device,
        shared_observation_spaces=env.state_spaces
    )

# 🏋️ Configure & Start Training
cfg_trainer = {"timesteps": 1000000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=training_agent)

print("🚀 Starting MAPPO Training...")
trainer.train()
print("🎯 Training Complete!")