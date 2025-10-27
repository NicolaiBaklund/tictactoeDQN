from numpyml.nn import layers, activations, optimizers, losses # pyright: ignore[reportMissingImports]
from numpyml.nn.model import Sequential # pyright: ignore[reportMissingImports]
from network.DQN import DQNAgent, linear_eps, ReplayBuffer, select_action
from environments.tictactoe import TicTacToeEnv
import numpy as np

q_online = Sequential([
    layers.Dense(9,64),
    activations.ReLU(),
    layers.Dense(64,64),
    activations.ReLU(),
    layers.Dense(64,9)
])

optim = optimizers.SGD(q_online.parameters(), q_online.gradients(), learning_rate=5e-4)
huber_loss = losses.Huber(delta=1.0)


rng = np.random.default_rng(42)
agent = DQNAgent(q_online, optim, huber_loss, gamma=0.99, target_update_k=500, action_dim=9)
env = TicTacToeEnv(seed=0, illegal_move_mode='raise', opponent='random')
buffer = ReplayBuffer(capacity=50000, obs_dim=9)


learn_start = 1000
batch_size = 64
learn_freq = 1
episodes = 10_000
step = 0


for episode in range(episodes):
    state = env.reset()
    obs = state["obs"]
    mask = state["mask"]

    done = False

    while not done:
        # act
        q_vals = q_online.forward(obs[None, :])[0]      # (9,)
        a = select_action(q_vals, mask, linear_eps(step, decay_steps=50_000), rng)

        # step env
        s_next, r, done, info = env.step(a)
        next_obs  = s_next["obs"].astype(np.float32)
        next_mask = s_next["mask"]

        # store
        buffer.add(obs, a, r, next_obs, done, next_mask)

        # learn
        if buffer.size >= learn_start and (step % learn_freq) == 0:
            batch = buffer.sample(batch_size)
            loss = agent.update(batch)

        obs, mask = next_obs, next_mask
        step += 1

    # simple logging every 50 eps
    if (episode + 1) % 50 == 0:
        w = info.get("winner")
        print(f"Ep {episode+1} | steps {step} | eps {linear_eps(step):.3f} | last winner: {w}")




q_online.save("tictactoe_dqn_model.pkl")

