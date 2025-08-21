import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

from pogema import pogema_v0, GridConfig, AnimationMonitor, AnimationConfig

# SKRL imports
from skrl.envs.wrappers.torch import wrap_env
from skrl.multi_agents.torch.mappo.mappo_cnn_rnn import MAPPO_CNN_RNN, MAPPO_CNN_RNN_DEFAULT_CONFIG
from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model
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
        max_episode_steps=50,
    ),
    render_mode=None
)

env = AnimationMonitor(env)
env = wrap_env(env, wrapper="pogema")  # Wrap for SKRL compatibility
obs, _ = env.reset()


# 🧠 Define Policy and Value Models with RNNs
# These definitions must match the architecture of your trained model

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_shape, output_dim=256):
        super().__init__()
        c, h, w = input_shape.shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            flat_dim = self.cnn(dummy_input).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(flat_dim, output_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.linear(self.cnn(x))

class Policy(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, num_envs=1, num_rnn_layers=1, hidden_size=256, sequence_length=64):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self)

        self.num_envs = num_envs
        self.num_rnn_layers = num_rnn_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.feature_extractor = CNNFeatureExtractor(observation_space, output_dim=256)
        self.rnn = nn.GRU(
            input_size=256,
            hidden_size=self.hidden_size,
            num_layers=self.num_rnn_layers,
            batch_first=True
        )
        self.policy_net = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )

    def get_specification(self):
        return {"rnn": {"sequence_length": self.sequence_length, "sizes": [(self.num_rnn_layers, self.num_envs, self.hidden_size)]}}

    def compute(self, inputs, role):
        features = self.feature_extractor(inputs["states"])
        hidden_states = inputs["rnn"][0]
        rnn_input = features.view(-1, 1, features.shape[-1])
        rnn_output, new_hidden_states = self.rnn(rnn_input, hidden_states)
        logits = self.policy_net(rnn_output.squeeze(1))
        return logits, {"rnn": [new_hidden_states]}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, num_envs=1, num_rnn_layers=1, hidden_size=256, sequence_length=64):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)

        self.num_envs = num_envs
        self.num_rnn_layers = num_rnn_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.feature_extractor = CNNFeatureExtractor(observation_space, output_dim=256)
        self.rnn = nn.GRU(
            input_size=256,
            hidden_size=self.hidden_size,
            num_layers=self.num_rnn_layers,
            batch_first=True)
        self.value_net = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def get_specification(self):
        return {"rnn": {"sequence_length": self.sequence_length, "sizes": [(self.num_rnn_layers, self.num_envs, self.hidden_size)]}}

    def compute(self, inputs, role):
        features = self.feature_extractor(inputs["states"])
        hidden_states = inputs["rnn"][0]
        rnn_input = features.view(-1, 1, features.shape[-1])
        rnn_output, new_hidden_states = self.rnn(rnn_input, hidden_states)
        value = self.value_net(rnn_output.squeeze(1))
        return value, {"rnn": [new_hidden_states]}

# 🔧 Instantiate models
models = {}
for agent_id in env.possible_agents:
    models[agent_id] = {
        "policy": Policy(env.observation_spaces[agent_id], env.action_spaces[agent_id], device, num_envs=env.num_envs),
        "value": Value(env.state_spaces[agent_id], env.action_spaces[agent_id], device, num_envs=env.num_envs)
    }

# 🎛 Instantiate the agent
cfg = MAPPO_CNN_RNN_DEFAULT_CONFIG.copy()
# No training config needed for demo, but agent requires a cfg
cfg["experiment"]["directory"] = "runs/torch/Pogema_MAPPO_RNN_Demo"

agent = MAPPO_CNN_RNN(
    possible_agents=env.possible_agents,
    models=models,
    memories=None,  # No memory needed for inference
    cfg=cfg,
    observation_spaces=env.observation_spaces,
    action_spaces=env.action_spaces,
    device=device,
    shared_observation_spaces=env.state_spaces
)

# 💡 Load the trained agent's checkpoint
# TODO: Replace with the actual path to your trained agent file
try:
    agent.load("runs/torch/Pogema_MAPPO_CNN_RNN/25-08-19_21-16-08-297397_MAPPO_CNN_RNN/checkpoints/best_agent.pt")
except FileNotFoundError:
    print("Checkpoint not found. Running with randomly initialized agent.")


# 🎮 Run one episode for demonstration
obs, _ = env.reset()
agent.init() # Initialize the agent for evaluation

overall_done = False
step = 0
max_steps = env.unwrapped.grid_config.max_episode_steps

while not overall_done and step < max_steps:
    with torch.no_grad():
        # Get actions from the agent
        actions, _, _ = agent.act(obs, timestep=step, timesteps=max_steps)

    # Step the environment
    obs, reward, terminated, truncated, _ = env.step(actions)

    # Check if all agents are done
    all_terminated = all(terminated.values())
    all_truncated = all(truncated.values())
    overall_done = all_terminated or all_truncated
    step += 1

# 🎨 Save SVG animation of the episode
env.save_animation("mappo_rnn_skrl.svg")
env.save_animation("mappo_rnn_skrl_ego0.svg", AnimationConfig(egocentric_idx=0))
print("✅ SVG animation saved successfully!")