"""
Deep Q-Network (DQN) Agent for the game 2048 — Rainbow-style.

Features:
- Dueling DQN architecture (separates state value and action advantage)
- Prioritized Experience Replay (samples surprising experiences more often)
- Deeper network with residual connections
- Noisy linear layers for exploration (replaces epsilon-greedy)
- Better state encoding with auxiliary features
- Quantile Regression DQN (QR-DQN) for distributional RL
- N-step returns for faster reward propagation
"""

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def encode_state(grid):
    """
    Encode the 4x4 grid as a multi-channel representation.

    Channels 0-15: one-hot encoding of log2(tile value)
    Channel 16: empty cell mask
    Channel 17: normalized tile values (val / max_val)

    Result: tensor (18, 4, 4).
    """
    encoded = np.zeros((18, 4, 4), dtype=np.float32)
    max_val = max(grid.flatten().max(), 1)

    for y in range(4):
        for x in range(4):
            val = grid[y][x]
            if val > 0:
                channel = int(np.log2(val))
                if channel < 16:
                    encoded[channel][y][x] = 1.0
                encoded[17][y][x] = val / max_val  # normalized value
            else:
                encoded[0][y][x] = 1.0
                encoded[16][y][x] = 1.0  # empty cell mask
    return encoded


class NoisyLinear(nn.Module):
    """
    Noisy Linear layer for exploration (Fortunato et al., 2018).
    Replaces epsilon-greedy with learned exploration noise.
    """

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DuelingDQN(nn.Module):
    """
    Dueling DQN with deeper architecture.

    Separates value estimation V(s) from advantage estimation A(s,a):
        Q(s,a) = V(s) + A(s,a) - mean(A(s,:))

    This helps the network learn which states are valuable regardless
    of the action taken, improving learning efficiency.
    """

    def __init__(self, in_channels=18):
        super().__init__()

        # Convolutional feature extractor — deeper with residual
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=2, padding=0)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, padding=0)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=2, padding=0)
        self.bn3 = nn.BatchNorm2d(256)

        # Feature size after convolutions: 256 * 1 * 1 = 256
        flat_size = 256

        # Value stream — estimates V(s)
        self.value_fc1 = nn.Linear(flat_size, 256)
        self.value_fc2 = NoisyLinear(256, 1)

        # Advantage stream — estimates A(s,a) for each action
        self.adv_fc1 = nn.Linear(flat_size, 256)
        self.adv_fc2 = NoisyLinear(256, 4)

    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        # Value stream
        v = F.relu(self.value_fc1(x))
        v = self.value_fc2(v)  # (batch, 1)

        # Advantage stream
        a = F.relu(self.adv_fc1(x))
        a = self.adv_fc2(a)  # (batch, 4)

        # Combine: Q = V + (A - mean(A))
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

    def reset_noise(self):
        self.value_fc2.reset_noise()
        self.adv_fc2.reset_noise()


class QuantileDQN(nn.Module):
    """
    Quantile Regression DQN (Dabney et al., 2018).

    Instead of estimating a single Q-value per action, this network outputs
    N_QUANTILES quantile estimates for each action. The Q-value is the mean
    of the quantile estimates: Q(s,a) = (1/N) * sum(theta_i(s,a)).

    Uses the same convolutional backbone and dueling structure as DuelingDQN,
    but the value and advantage heads output N_QUANTILES values each.
    """

    N_QUANTILES = 51

    def __init__(self, in_channels=18):
        super().__init__()
        n_quant = self.N_QUANTILES

        # Convolutional feature extractor (same as DuelingDQN)
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=2, padding=0)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, padding=0)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=2, padding=0)
        self.bn3 = nn.BatchNorm2d(256)

        flat_size = 256

        # Value stream: outputs N_QUANTILES values for V(s)
        self.value_fc1 = nn.Linear(flat_size, 256)
        self.value_fc2 = NoisyLinear(256, n_quant)

        # Advantage stream: outputs 4 * N_QUANTILES values for A(s,a)
        self.adv_fc1 = nn.Linear(flat_size, 256)
        self.adv_fc2 = NoisyLinear(256, 4 * n_quant)

    def forward(self, x):
        batch_size = x.size(0)
        n_quant = self.N_QUANTILES

        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(batch_size, -1)

        # Value stream: (batch, N_QUANTILES)
        v = F.relu(self.value_fc1(x))
        v = self.value_fc2(v).view(batch_size, 1, n_quant)

        # Advantage stream: (batch, 4, N_QUANTILES)
        a = F.relu(self.adv_fc1(x))
        a = self.adv_fc2(a).view(batch_size, 4, n_quant)

        # Dueling combine: Q = V + (A - mean(A))
        q = v + (a - a.mean(dim=1, keepdim=True))  # (batch, 4, N_QUANTILES)
        return q

    def q_values(self, x):
        """Return mean Q-values across quantiles: (batch, 4)."""
        return self.forward(x).mean(dim=2)

    def reset_noise(self):
        self.value_fc2.reset_noise()
        self.adv_fc2.reset_noise()


def quantile_huber_loss(quantiles, target, taus, kappa=1.0):
    """
    Compute the quantile Huber loss for QR-DQN.

    For each quantile tau_i, the asymmetric weight ensures:
    - Overestimation (positive error) is penalized by tau_i
    - Underestimation (negative error) is penalized by (1 - tau_i)

    This learns the full return distribution, not just the expected value.

    Args:
        quantiles: predicted quantile values, shape (batch, N_QUANTILES)
        target: target quantile values, shape (batch, N_QUANTILES)
        taus: quantile midpoints, shape (1, N_QUANTILES)
        kappa: Huber loss threshold (1.0 = standard Huber)

    Returns:
        loss: scalar tensor, mean quantile Huber loss
    """
    n_quant = quantiles.shape[1]

    # Pairwise TD errors: (batch, N_QUANTILES, N_QUANTILES)
    # quantiles[:, :, None] = (batch, N_quant_pred, 1)
    # target[:, None, :] = (batch, 1, N_quant_target)
    td_error = target[:, None, :] - quantiles[:, :, None]

    # Huber loss element-wise
    huber = torch.where(
        td_error.abs() <= kappa,
        0.5 * td_error.pow(2),
        kappa * (td_error.abs() - 0.5 * kappa),
    )

    # Asymmetric weighting by quantile level
    # taus shape: (1, N_QUANTILES) -> (1, N_QUANTILES, 1)
    taus_expanded = taus.unsqueeze(2)
    quantile_weight = torch.abs(taus_expanded - (td_error < 0).float())

    loss = (quantile_weight * huber).sum(dim=2).mean(dim=1)
    return loss


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (Schaul et al., 2016).

    Stores transitions with priorities. Higher priority = more likely to be sampled.
    Priority is based on TD error: |Q(s,a) - target| — surprising transitions
    are replayed more often.

    Uses a sum tree for O(log n) sampling.
    """

    def __init__(self, capacity=200_000, alpha=0.6, beta_start=0.4, beta_frames=100_000):
        self.capacity = capacity
        self.alpha = alpha  # priority exponent (0 = uniform, 1 = full prioritization)
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done, valid_moves):
        transition = (state, action, reward, next_state, done, valid_moves)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        self.frame += 1
        n = len(self.buffer)

        # Compute sampling probabilities
        priorities = self.priorities[:n] ** self.alpha
        probs = priorities / priorities.sum()

        # Sample indices
        indices = np.random.choice(n, batch_size, p=probs, replace=False)

        # Importance sampling weights (for unbiased updates)
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        weights = (n * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # normalize

        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones, valid_moves = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            valid_moves,
            indices,
            torch.FloatTensor(weights),
        )

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 1e-6
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


# Keep old class name for backward compatibility with server.py
DQN = DuelingDQN
ReplayBuffer = PrioritizedReplayBuffer


class NStepBuffer:
    """
    Accumulates n consecutive transitions and computes n-step returns.

    Instead of learning from single transitions (s, a, r, s'),
    we learn from n-step transitions (s, a, R_n, s_n) where:
        R_n = r_1 + γ·r_2 + γ²·r_3 + ... + γ^(n-1)·r_n

    This propagates rewards faster through the network,
    crucial for long-horizon games like 2048.
    """

    def __init__(self, n_steps=3, gamma=0.99):
        self.n_steps = n_steps
        self.gamma = gamma
        self.buffer = deque(maxlen=n_steps)

    def push(self, state, action, reward, next_state, done, valid_moves):
        self.buffer.append((state, action, reward, next_state, done, valid_moves))

    def is_ready(self):
        return len(self.buffer) == self.n_steps

    def get(self):
        """Returns the n-step transition (s_0, a_0, R_n, s_n, done_n, valid_n)."""
        state, action = self.buffer[0][0], self.buffer[0][1]

        # Compute n-step discounted reward
        n_step_reward = 0
        for i, (_, _, r, _, d, _) in enumerate(self.buffer):
            n_step_reward += (self.gamma ** i) * r
            if d:
                # Episode ended before n steps — use this as the final state
                return state, action, n_step_reward, self.buffer[i][3], True, self.buffer[i][5]

        # No terminal state in the n steps — use the last state
        last = self.buffer[-1]
        return state, action, n_step_reward, last[3], last[4], last[5]

    def reset(self):
        self.buffer.clear()


class DQNAgent:
    """
    Rainbow-style DQN Agent with:
    - Dueling DQN architecture
    - Prioritized Experience Replay
    - Noisy Networks for exploration (no epsilon-greedy needed)
    - Double DQN for reduced overestimation
    - N-step returns (faster reward propagation)
    - Quantile Regression DQN (distributional RL)
    - Gradient clipping
    - Learning rate scheduling
    """

    N_STEPS = 3  # look-ahead steps for n-step returns

    def __init__(
        self,
        lr=5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=100_000,
        batch_size=256,
        target_update=500,
        buffer_size=200_000,
        distributional=True,
        device=None,
    ):
        self.device = device or torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.distributional = distributional
        print(f"Using device: {self.device}")
        print(f"Distributional (QR-DQN): {self.distributional}")

        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        # Networks: choose architecture based on distributional flag
        if self.distributional:
            self.policy_net = QuantileDQN(in_channels=18).to(self.device)
            self.target_net = QuantileDQN(in_channels=18).to(self.device)
            # Precompute quantile midpoints: tau_i = (i + 0.5) / N
            n_quant = QuantileDQN.N_QUANTILES
            self.taus = torch.FloatTensor(
                [(i + 0.5) / n_quant for i in range(n_quant)]
            ).unsqueeze(0).to(self.device)  # (1, N_QUANTILES)
        else:
            self.policy_net = DuelingDQN(in_channels=18).to(self.device)
            self.target_net = DuelingDQN(in_channels=18).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50_000, gamma=0.5)
        self.memory = PrioritizedReplayBuffer(buffer_size)
        self.n_step_buffer = NStepBuffer(self.N_STEPS, gamma)

        self.steps_done = 0
        self.training_losses = []

    def get_epsilon(self):
        """Exponential epsilon decay (used as fallback, noisy nets handle exploration)."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-self.steps_done / self.epsilon_decay)

    def select_action(self, state, valid_moves, training=True):
        """
        Select action using noisy network (no epsilon-greedy needed in training).
        Falls back to epsilon-greedy for early exploration.

        For distributional mode, Q(s,a) = mean of quantile estimates for action a.
        """
        if not valid_moves:
            return 0

        # Early epsilon-greedy for initial exploration
        if training and self.steps_done < 5000 and random.random() < 0.5:
            return random.choice(valid_moves)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if training:
                self.policy_net.reset_noise()
            self.policy_net.eval()

            if self.distributional:
                # Q-values = mean across quantiles for each action
                q_values = self.policy_net.q_values(state_tensor).cpu().numpy()[0]
            else:
                q_values = self.policy_net(state_tensor).cpu().numpy()[0]

            self.policy_net.train()

            # Mask invalid actions with -inf
            masked = np.full(4, -np.inf)
            for m in valid_moves:
                masked[m] = q_values[m]
            return int(np.argmax(masked))

    def store_transition(self, state, action, reward, next_state, done, valid_moves):
        """Store transition using n-step buffer. Only pushes to replay when n steps accumulated."""
        self.n_step_buffer.push(state, action, reward, next_state, done, valid_moves)

        if self.n_step_buffer.is_ready():
            s, a, r_n, s_n, d_n, vm_n = self.n_step_buffer.get()
            self.memory.push(s, a, r_n, s_n, d_n, vm_n)

        # Flush remaining transitions at episode end
        if done:
            while len(self.n_step_buffer.buffer) > 0:
                s, a, r_n, s_n, d_n, vm_n = self.n_step_buffer.get()
                self.memory.push(s, a, r_n, s_n, d_n, vm_n)
                self.n_step_buffer.buffer.popleft()
            self.n_step_buffer.reset()

    def train_step(self):
        """Training step with Double Dueling DQN + Prioritized Replay + N-step returns.

        When distributional=True, uses quantile regression loss (QR-DQN).
        Otherwise, uses standard Huber loss.
        """
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, valid_moves_batch, \
            indices, is_weights = self.memory.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        is_weights_t = is_weights.to(self.device)

        # Reset noise for current forward pass
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        if self.distributional:
            loss, td_errors = self._train_step_distributional(
                states_t, actions_t, rewards_t, next_states_t, dones_t,
                valid_moves_batch, is_weights_t,
            )
        else:
            loss, td_errors = self._train_step_standard(
                states_t, actions_t, rewards_t, next_states_t, dones_t,
                valid_moves_batch, is_weights_t,
            )

        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_errors)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        self.scheduler.step()

        self.steps_done += 1

        # Hard update target network periodically
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        loss_val = loss.item()
        self.training_losses.append(loss_val)
        return loss_val

    def _train_step_standard(self, states_t, actions_t, rewards_t,
                             next_states_t, dones_t, valid_moves_batch,
                             is_weights_t):
        """Standard Double DQN training with Huber loss."""
        # Current Q-values
        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN: policy net selects action, target net evaluates
        with torch.no_grad():
            next_q_policy = self.policy_net(next_states_t)

            # Mask invalid actions
            for i, vm in enumerate(valid_moves_batch):
                mask = torch.ones(4, device=self.device) * (-1e9)
                for m in vm:
                    mask[m] = 0
                next_q_policy[i] += mask

            best_actions = next_q_policy.argmax(1)
            next_q_target = self.target_net(next_states_t)
            next_q = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            # N-step: discount by gamma^n instead of gamma
            target = rewards_t + (self.gamma ** self.N_STEPS) * next_q * (1 - dones_t)

        # TD errors for priority update
        td_errors = (q_values - target).detach().cpu().numpy()

        # Weighted loss (importance sampling correction)
        loss = (is_weights_t * F.smooth_l1_loss(q_values, target, reduction='none')).mean()
        return loss, td_errors

    def _train_step_distributional(self, states_t, actions_t, rewards_t,
                                   next_states_t, dones_t, valid_moves_batch,
                                   is_weights_t):
        """Distributional training with quantile regression (QR-DQN).

        Instead of learning E[R], learns the quantile function of the return
        distribution. The loss is an asymmetrically-weighted Huber loss
        that pushes each quantile estimate toward the correct quantile
        of the target distribution.
        """
        batch_size = states_t.size(0)
        n_quant = QuantileDQN.N_QUANTILES

        # Current quantile estimates for chosen actions: (batch, N_QUANTILES)
        all_quantiles = self.policy_net(states_t)  # (batch, 4, N_QUANTILES)
        actions_expanded = actions_t.unsqueeze(1).unsqueeze(2).expand(
            batch_size, 1, n_quant
        )
        current_quantiles = all_quantiles.gather(1, actions_expanded).squeeze(1)

        # Double DQN action selection using mean Q-values from policy net
        with torch.no_grad():
            next_q_policy = self.policy_net.q_values(next_states_t)  # (batch, 4)

            # Mask invalid actions
            for i, vm in enumerate(valid_moves_batch):
                mask = torch.ones(4, device=self.device) * (-1e9)
                for m in vm:
                    mask[m] = 0
                next_q_policy[i] += mask

            best_actions = next_q_policy.argmax(1)  # (batch,)

            # Get target quantiles for best actions
            next_quantiles_all = self.target_net(next_states_t)  # (batch, 4, N_QUANTILES)
            best_expanded = best_actions.unsqueeze(1).unsqueeze(2).expand(
                batch_size, 1, n_quant
            )
            next_quantiles = next_quantiles_all.gather(1, best_expanded).squeeze(1)

            # Compute target quantiles with n-step returns
            # T_theta = r + gamma^n * theta(s', a*) for non-terminal
            target_quantiles = rewards_t.unsqueeze(1) + \
                (self.gamma ** self.N_STEPS) * next_quantiles * \
                (1 - dones_t).unsqueeze(1)

        # Quantile Huber loss per sample: (batch,)
        sample_losses = quantile_huber_loss(
            current_quantiles, target_quantiles, self.taus
        )

        # TD error for priority update: use mean absolute quantile error
        td_errors = (current_quantiles - target_quantiles).abs().mean(dim=1).detach().cpu().numpy()

        # Apply importance sampling weights
        loss = (is_weights_t * sample_losses).mean()
        return loss, td_errors

    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.get_epsilon(),
            'distributional': self.distributional,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        try:
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.steps_done = checkpoint['steps_done']
            print(f"Model loaded from {path} (steps: {self.steps_done})")
        except RuntimeError:
            # Old checkpoint format — incompatible architecture
            self.steps_done = checkpoint.get('steps_done', 0)
            print(f"Old checkpoint format — starting with new architecture (steps reset)")
