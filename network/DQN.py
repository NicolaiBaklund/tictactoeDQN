from numpyml.nn import layers, activations, optimizers, losses # pyright: ignore[reportMissingImports]
from numpyml.nn.model import Sequential # pyright: ignore[reportMissingImports]



import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=bool)
        self.next_mask = np.zeros((capacity, 9), dtype=bool)  # 9 actions
        self.size = 0
        self.ptr = 0

    def add(self, s_obs, action, reward, s_next_obs, done, s_next_mask):
        i = self.ptr
        self.obs[i] = s_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = s_next_obs
        self.done[i] = done
        self.next_mask[i] = s_next_mask
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.obs[idx],
                self.actions[idx],
                self.rewards[idx],
                self.next_obs[idx],
                self.done[idx],
                self.next_mask[idx])


def select_action(q_values, legal_mask, epsilon, rng):
    if rng.random() < epsilon:
        legal = np.flatnonzero(legal_mask)
        return int(rng.choice(legal))
    q = q_values.copy()
    q[~legal_mask] = -np.inf
    return int(np.argmax(q))


def compute_targets_double_dqn(q_online, q_target, next_obs_batch, next_mask_batch, rewards, done, gamma):
    # q_online/ q_target: callables returning (B,9) from (B,obs_dim)
    q_online_next = q_online(next_obs_batch)            # (B, 9)
    q_online_next[~next_mask_batch] = -np.inf
    a_star = np.argmax(q_online_next, axis=1)           # (B,)

    q_target_next = q_target(next_obs_batch)            # (B, 9)
    batch_idx = np.arange(len(a_star))
    max_next = q_target_next[batch_idx, a_star]         # (B,)

    targets = rewards.copy()
    not_done = ~done
    targets[not_done] += gamma * max_next[not_done]
    return targets




class DQNAgent:
    def __init__(self, q_online: Sequential, optimizer, huber_loss, gamma=0.99,
                 target_update_k=500, action_dim=9):
        self.q_online = q_online          # Neural net approximating Q(s,a)
        self.q_target = q_online.copy()          # Target net used for "labels"
        self.opt = optimizer              # your optimizer bound to q_online params
        self.loss_fn = huber_loss         # instance of huber loss class
        self.gamma = gamma
        self.target_update_k = target_update_k
        self.action_dim = action_dim
        self.train_steps = 0

    # wrappers so compute_targets can call them
    def _forward_online(self, obs_batch):  # -> (B,9)
        return self.q_online.forward(obs_batch)

    def _forward_target(self, obs_batch):
        return self.q_target.forward(obs_batch)

    def update(self, batch, delta=1.0):
        obs, actions, rewards, next_obs, done, next_mask = batch

        # ---- forward current Q(s,·)
        q_pred = self.q_online.forward(obs)                 # (B, 9)
        B = q_pred.shape[0]
        batch_idx = np.arange(B)
        q_sa = q_pred[batch_idx, actions]                   # (B,)

        # ---- targets (Double DQN)
        targets = compute_targets_double_dqn(
            self._forward_online, self._forward_target,
            next_obs, next_mask, rewards, done, self.gamma
        )

        # ---- Huber on selected actions
        loss = self.loss_fn.forward(q_sa, targets, training=True)
        grad_sa = self.loss_fn.backward()              # (B,)

        # ---- scatter grad back to (B,9)
        grad_q = np.zeros_like(q_pred)
        grad_q[batch_idx, actions] = grad_sa

        # ---- backprop + step
        self.q_online.backward(grad_q)                 # adapt if your API differs
        self.opt.step()
        self.opt.zero_grad()
        self.train_steps += 1

        # ---- target hard update
        if self.train_steps % self.target_update_k == 0:
            self.copy_online_to_target()

        return float(loss)

    def copy_online_to_target(self):
        self.q_target = self.q_online.copy()

    

def linear_eps(step, start=1.0, end=0.05, decay_steps=20_000):
    t = min(1.0, step / decay_steps)
    return start + (end - start) * t
