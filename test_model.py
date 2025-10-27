# test_model.py
from numpyml.nn.model import Sequential # pyright: ignore[reportMissingImports]
from environments.tictactoe import TicTacToeEnv
import numpy as np

def select_best_action(q_values, legal_mask):
    q = q_values.copy()
    q[~legal_mask] = -np.inf
    return int(np.argmax(q))

def play_many(model_path: str, rounds: int = 10):
    model: Sequential = Sequential.load(model_path)

    wins = draws = losses = 0
    for i in range(rounds):
        print(f"\n--- Game {i+1}/{rounds} ---")
        # Human plays O; env will prompt for O via opponent="player"
        env = TicTacToeEnv(seed=None, illegal_move_mode="raise", opponent="player")
        state = env.reset()

        done = False
        while not done:
            # Agent (X) move – YOU do this part
            obs = state["obs"].astype(np.float32)
            mask = state["mask"]
            q_vals = model.forward(obs[None, :])[0]  # (9,)
            a = select_best_action(q_vals, mask)

            state, reward, done, info = env.step(a)  # env will then prompt the human (O) automatically
            # After env.step, either the game ended, or it advanced through O's reply already.
            # render() is already called inside the env for the player's turn.

        winner = info.get("winner")
        if winner == -1:       # human is O
            print("You WIN 🎉")
            wins += 1
        elif winner is None:
            print("Draw 😐")
            draws += 1
        else:
            print("You LOSE 💀")
            losses += 1

    print("\n=== Results ===")
    print(f"Wins:   {wins}")
    print(f"Draws:  {draws}")
    print(f"Losses: {losses}")

if __name__ == "__main__":
    play_many("tictactoe_dqn_model.pkl", rounds=10)
