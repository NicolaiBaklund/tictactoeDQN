import numpy as np
from typing import Callable, Dict, Optional, Tuple

Tensor = np.ndarray
State = Dict[str, Tensor]

WIN_TRIPLETS = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),   # cols
    (0, 4, 8), (2, 4, 6),              # diagonals
)


class TicTacToeEnv:
    """
    Agent is always 'X' (+1). Opponent is 'O' (-1).
    One env.step() = agent move (X), optional opponent reply (O).
    Rewards (agent-centric): +1 win, -1 loss, 0 draw/continue.
    """
    def __init__(self, seed: Optional[int] = None,
                 illegal_move_mode: str = "raise", opponent: str = "random") -> None:
        """
        illegal_move_mode:
            - "penalize": illegal action ends episode with reward -1.
            - "raise": raise ValueError on illegal action (useful for debugging).
        """
        self.board: Tensor = np.zeros(9, dtype=np.int8)
        self.done: bool = False
        self._winner: Optional[int] = None

        if opponent == "random":
            self._opponent = self._random_opponent
        elif opponent == "player":
            self._opponent = self._player_opponent
        else:
            raise ValueError(f"Unknown opponent='{opponent}'. Use 'random' or 'player'.")
    
        self.illegal_move_mode = illegal_move_mode
        self.rng = np.random.default_rng(seed)

    # API methods

    def reset(self) -> State:
        self.board.fill(0)
        self.done = False
        self._winner = None
        return self._state()

    def step(self, action: int) -> Tuple[State, float, bool, Dict]:
        if self.done:
            raise ValueError("Episode is done. Call reset().")


        # Agent move (X = +1)
        if not self._is_legal(action):
            if self.illegal_move_mode == "raise":
                raise ValueError(f"Illegal action {action}.")
            # Penalize illegal move and end
            self.done = True
            self._winner = -1  # treat as a loss for the agent
            return self._state(), -1.0, True, {"winner": self._winner, "illegal_action": action}

        self.board[action] = +1
        if self._is_win(+1):
            self.done, self._winner = True, +1
            return self._state(), +1.0, True, {"winner": +1}
        if self._is_draw():
            self.done, self._winner = True, None
            return self._state(), 0.0, True, {"winner": None}

        # Opponent move (O = -1)
        opp_action = self._opponent(self.board, self.rng)
        if not self._is_legal(opp_action):  # defensive fallback, should not happen
            legal = np.flatnonzero(self.board == 0)
            opp_action = int(self.rng.choice(legal))
        self.board[opp_action] = -1
        if self._is_win(-1):
            self.done, self._winner = True, -1
            return self._state(), -1.0, True, {"winner": -1}
        if self._is_draw():
            self.done, self._winner = True, None
            return self._state(), 0.0, True, {"winner": None}

        # Non terminal state
        return self._state(), 0.0, False, {}

    def legal_action_mask(self) -> Tensor:
        return (self.board == 0)

    def set_opponent(self, policy_fn: Callable[[Tensor, np.random.Generator], int]) -> None:
        """
        policy_fn(board, rng) -> legal action in [0..8]
        """
        self._opponent = policy_fn

    def seed(self, s: int) -> None:
        self.rng = np.random.default_rng(s)

    def render(self) -> None:
        sym = {+1: "X", -1: "O", 0: " "}
        for r in range(3):
            row = [sym[int(self.board[3*r + c])] for c in range(3)]
            print(" | ".join(row))
            if r < 2: print("--+---+--")
        print()

    # helpers

    def _state(self) -> State:
        # Return obs + mask for DQN (agent-centric)
        return {
            "obs": self.board.copy(),  # shape (9,), values in {-1,0,+1}
            "mask": self.legal_action_mask().copy(),  # bool[9]
        }

    def _is_legal(self, action: int) -> bool:
        return 0 <= action < 9 and self.board[action] == 0

    def _is_win(self, mark: int) -> bool:
        b = self.board
        for a, c, d in WIN_TRIPLETS:
            if b[a] == b[c] == b[d] == mark:
                return True
        return False

    def _is_draw(self) -> bool:
        return bool(np.all(self.board != 0))

    def _random_opponent(self, board: Tensor, rng: np.random.Generator) -> int:
        legal = np.flatnonzero(board == 0)
        # In a properly called step(), there is always at least one legal move here
        return int(rng.choice(legal))
    def _player_opponent(self, board: Tensor, rng: np.random.Generator) -> int:
        print("Opponent's turn.")
        # Dependent on being called from step(), so there is at least one legal move
        self.render()
        legal = set(np.flatnonzero(board == 0).tolist())
        while True:
            try:
                action = int(input("Enter your move (0-8): "))
                if action in legal:
                    return action
                else:
                    print("Illegal move. Try again.")
            except ValueError:
                print("Invalid input. Enter an integer between 0 and 8. Space must be unoccupied.")


# --- tiny manual test ---
if __name__ == "__main__":
    env = TicTacToeEnv(seed=0, illegal_move_mode="raise", opponent="player")
    s = env.reset()
    env.render()
    # random-vs-random smoke test
    done = False
    while not done:
        # agent picks random legal move
        legal = np.flatnonzero(s["mask"])
        a = int(np.random.choice(legal))
        s, r, done, info = env.step(a)
        env.render()
        if done:
            print(f"Reward: {r}, Winner: {info.get('winner')}")
