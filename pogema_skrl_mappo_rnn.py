import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from pogema import pogema_v0, GridConfig

# SKRL imports
from skrl.envs.wrappers.torch import wrap_env
from skrl.multi_agents.torch.mappo.mappo_cnn_rnn import MAPPO_CNN_RNN, MAPPO_CNN_RNN_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
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
        max_episode_steps=200,
        # on_target = 'nothing',
    ),
    render_mode=None
)

env = wrap_env(env, wrapper="pogema")  # Wrap for SKRL compatibility
obs, _ = env.reset()

# CHANGED: Simplified CNNFeatureExtractor
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

# CHANGED: Corrected Policy model with proper sequence handling during training
class Policy(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, num_envs=1, num_rnn_layers=1, hidden_size=128, sequence_length=64):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self)

        self.num_envs = num_envs
        self.num_rnn_layers = num_rnn_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.feature_extractor = CNNFeatureExtractor(observation_space, output_dim=128)
        self.rnn = nn.GRU(
            input_size=128,
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

        # Training mode: process full sequences
        if self.training:
            # Reshape features to (batch_size, sequence_length, feature_size)
            rnn_input = features.view(-1, self.sequence_length, features.shape[-1])
            # Reshape hidden states to (num_layers, batch_size, hidden_size)
            # We only need the hidden state from the start of the sequence
            hidden_states = hidden_states.view(self.num_rnn_layers, -1, self.sequence_length, self.hidden_size)
            hidden_states = hidden_states[:, :, 0, :].contiguous()
            
            rnn_output, new_hidden_states = self.rnn(rnn_input, hidden_states)
            # Reshape output back to (batch_size * sequence_length, hidden_size)
            rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)
        # Evaluation/Rollout mode: process one step at a time
        else:
            rnn_input = features.view(-1, 1, features.shape[-1])
            rnn_output, new_hidden_states = self.rnn(rnn_input, hidden_states)
            rnn_output = rnn_output.squeeze(1)

        logits = self.policy_net(rnn_output)
        return logits, {"rnn": [new_hidden_states]}

# CHANGED: Corrected Value model with proper sequence handling during training
class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, num_envs=1, num_rnn_layers=1, hidden_size=128, sequence_length=64):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)

        self.num_envs = num_envs
        self.num_rnn_layers = num_rnn_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.feature_extractor = CNNFeatureExtractor(observation_space, output_dim=128)
        self.rnn = nn.GRU(
            input_size=128,
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

        # Training mode: process full sequences
        if self.training:
            rnn_input = features.view(-1, self.sequence_length, features.shape[-1])
            hidden_states = hidden_states.view(self.num_rnn_layers, -1, self.sequence_length, self.hidden_size)
            hidden_states = hidden_states[:, :, 0, :].contiguous()
            
            rnn_output, new_hidden_states = self.rnn(rnn_input, hidden_states)
            rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)
        # Evaluation/Rollout mode: process one step at a time
        else:
            rnn_input = features.view(-1, 1, features.shape[-1])
            rnn_output, new_hidden_states = self.rnn(rnn_input, hidden_states)
            rnn_output = rnn_output.squeeze(1)

        value = self.value_net(rnn_output)
        return value, {"rnn": [new_hidden_states]}


# 🔧 Define models and memories for each agent
models = {}
memories = {}
for agent_id in env.possible_agents:
    models[agent_id] = {
        "policy": Policy(env.observation_spaces[agent_id], env.action_spaces[agent_id], device, num_envs=env.num_envs),
        "value": Value(env.state_spaces[agent_id], env.action_spaces[agent_id], device, num_envs=env.num_envs)
    }
    memories[agent_id] = RandomMemory(memory_size=1024, num_envs=env.num_envs, device=device)

# 🎛 Configure MAPPO_RNN Agent
cfg = MAPPO_CNN_RNN_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 1024
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 8
cfg["discount_factor"] = 0.95
cfg["lambda"] = 0.95
cfg["learning_rate"] = 2e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["grad_norm_clip"] = 0.5
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.02
cfg["value_loss_scale"] = 1.0
# No preprocessors needed for image-only input
cfg["state_preprocessor"] = None
cfg["shared_state_preprocessor"] = None

# ✅ Set up logging & checkpoints
cfg["experiment"]["directory"] = "runs/torch/Pogema_MAPPO_CNN_RNN"
cfg["experiment"]["write_interval"] = 50000
cfg["experiment"]["checkpoint_interval"] = 200000

agent = MAPPO_CNN_RNN(
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
cfg_trainer = {"timesteps": 2000000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

print("🚀 Starting MAPPO RNN Training...")
trainer.train()
print("🎯 Training Complete!")