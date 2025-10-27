# test_model.py
import argparse
from numpyml.nn.model import Sequential # pyright: ignore[reportMissingImports]
from environments.tictactoe import TicTacToeEnv
import numpy as np


def select_best_action(q_values, legal_mask):
    q = q_values.copy()
    q[~legal_mask] = -np.inf
    return int(np.argmax(q))


def evaluate_model(model_path: str, rounds: int = 1000, seed: int | None = None):
    """Run the agent (X) vs opponent='random' and report win/draw/loss rates.

    Args:
        model_path: path to the saved Sequential model (Sequential.load).
        rounds: number of episodes to play.
        seed: optional RNG seed for reproducibility (None => non-deterministic).
    """
    model: Sequential = Sequential.load(model_path)

    wins = draws = losses = 0

    # create env with random opponent
    env = TicTacToeEnv(seed=seed, illegal_move_mode="raise", opponent="random")

    for i in range(rounds):
        state = env.reset()

        done = False
        while not done:
            obs = state["obs"].astype(np.float32)
            mask = state["mask"]
            q_vals = model.forward(obs[None, :])[0]  # (9,)
            a = select_best_action(q_vals, mask)

            state, reward, done, info = env.step(a)

        winner = info.get("winner")
        if winner == +1:
            wins += 1
        elif winner is None:
            draws += 1
        else:
            losses += 1

        # small progress print occasionally
        if (i + 1) % max(1, rounds // 10) == 0:
            print(f"Played {i+1}/{rounds} episodes...")

    total = wins + draws + losses
    assert total == rounds

    print("\n=== Evaluation Results ===")
    print(f"Episodes: {rounds}")
    print(f"Wins:   {wins} ({wins/rounds:.2%})")
    print(f"Draws:  {draws} ({draws/rounds:.2%})")
    print(f"Losses: {losses} ({losses/rounds:.2%})")


def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate DQN TicTacToe model vs random opponent")
    p.add_argument("model_path", nargs="?", default="tictactoe_dqn_model.pkl",
                   help="Path to saved Sequential model (default: tictactoe_dqn_model.pkl)")
    p.add_argument("-n", "--rounds", type=int, default=100000, help="Number of episodes to run")
    p.add_argument("--seed", type=int, default=None, help="Optional RNG seed for env")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_model(args.model_path, rounds=args.rounds, seed=args.seed)
