"""Automated online self-training against computer on play.xiangqi.com."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Optional

from my_chess.chess_ai.action_space import move_to_action_id
from my_chess.chess_ai.online_learning import (
    MistakeMemoryStore,
    OnlineLearningSession,
    PersistentReplayBuffer,
)
from my_chess.chess_ai.player import AIPlayer
from my_chess.chess_ai.train import (
    DEFAULT_CHECKPOINT_PATH,
    TrainingConfig,
    incremental_train_on_samples,
)
from my_chess.chess_core.chessboard import Chessboard
from my_chess.chess_core.move import Move
from my_chess.chess_core.wxf_notation import parse_wxf_move

BRIDGE_SCRIPT_PATH = Path(__file__).parent / "browser_bridge_server.mjs"
SERVER_URL = "http://127.0.0.1:45678"


class BrowserBridgeClient:
    """Client for controlling the ego-browser Xiangqi session."""

    def __init__(self, bridge_script: Path = BRIDGE_SCRIPT_PATH) -> None:
        self.bridge_script = bridge_script
        self.proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        """Launches the Node.js bridge server."""
        print(f"[online-agent] Starting browser bridge server...")
        # Use cmd /c redirection so ego-browser reads the mjs file cleanly on Windows
        cmd = f'cmd.exe /c "ego-browser nodejs < {self.bridge_script.resolve()}"'
        self.proc = subprocess.Popen(cmd, shell=True)

        # Wait for server to come online
        for _ in range(15):
            time.sleep(1)
            try:
                resp = urllib.request.urlopen(f"{SERVER_URL}/ping", timeout=2)
                data = json.loads(resp.read().decode())
                if data.get("status") == "pong":
                    print("[online-agent] Browser bridge server online!")
                    return
            except Exception:
                pass
        raise RuntimeError("Failed to connect to browser bridge server on port 45678")

    def stop(self) -> None:
        """Closes the browser TaskSpace and stops the server."""
        try:
            req = urllib.request.Request(
                f"{SERVER_URL}/close",
                data=b"{}",
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass
        if self.proc:
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        print("[online-agent] Browser bridge stopped.")

    def new_game(self, level: int = 1) -> dict:
        """Opens a new computer match on play.xiangqi.com with the specified bot level (1-10)."""
        payload = json.dumps({"level": level}).encode("utf-8")
        req = urllib.request.Request(
            f"{SERVER_URL}/start_game",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read().decode())

    def play_move(
        self,
        from_col: int,
        from_row: int,
        to_col: int,
        to_row: int,
        expected_count: int,
    ) -> dict:
        """Plays our move on the webpage and waits for opponent's response."""
        payload = json.dumps(
            {
                "from_col": from_col,
                "from_row": from_row,
                "to_col": to_col,
                "to_row": to_row,
                "expected_move_count": expected_count,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{SERVER_URL}/play_move",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read().decode())


def run_online_training_session(
    num_games: int = 1,
    num_simulations: int = 800,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    max_moves: int = 300,
    level: int = 1,
) -> None:
    """Runs automated games against play.xiangqi.com and updates the checkpoint."""
    client = BrowserBridgeClient()
    client.start()

    learning_session = OnlineLearningSession()
    replay_store = PersistentReplayBuffer()
    mistake_store = MistakeMemoryStore()

    ai = AIPlayer(
        is_red=True,
        num_simulations=num_simulations,
        checkpoint_path=checkpoint_path,
        mistake_memory=mistake_store.load(),
    )

    try:
        for game_idx in range(1, num_games + 1):
            game_info = client.new_game(level=level)
            bot_name = game_info.get("bot", f"Level {level}")

            print(f"\n{'='*60}")
            print(
                f"[online-agent] Starting Game {game_idx}/{num_games} against play.xiangqi.com ({bot_name})..."
            )
            print(f"{'='*60}")

            board = Chessboard(f"online_game_{game_idx}")
            board.init_board()
            learning_session.reset()

            move_index = 0
            winner: Optional[str] = None

            while not board.is_end() and move_index < max_moves:
                # 1. AI's turn (Red)
                selected_move, visit_distribution, root_value = (
                    ai.choose_move_with_distribution(board)
                )
                learning_session.record_distribution(board, visit_distribution)
                learning_session.record_ai_action(
                    board, move_to_action_id(selected_move)
                )

                print(
                    f"[{move_index+1}] Red (Our AI): {selected_move.to_ucci()} "
                    f"eval={root_value:+.2f}"
                )

                expected_count = len(board.moves_history) + 2
                res = client.play_move(
                    selected_move.from_point.x,
                    selected_move.from_point.y,
                    selected_move.to_point.x,
                    selected_move.to_point.y,
                    expected_count,
                )

                board.apply_move(selected_move)
                move_index += 1

                winner = board.get_winner()
                if winner or res.get("gameOver"):
                    if not winner and res.get("gameOver"):
                        txt = res["gameOver"].lower()
                        winner = "Red" if "red" in txt or "win" in txt else "Draw"
                    break

                # 2. Computer's turn (Black)
                opp_wxf = res.get("latestOpponentMove")
                if not opp_wxf:
                    print(
                        "[online-agent] Waiting for opponent move timed out or game ended."
                    )
                    break

                opp_move = parse_wxf_move(board, opp_wxf)
                if opp_move is None:
                    print(
                        f"[online-agent] Warning: could not parse WXF move '{opp_wxf}', picking legal fallback"
                    )
                    legal = board.get_legal_moves()
                    if legal:
                        opp_move = legal[0]
                    else:
                        break

                print(
                    f"[{move_index+1}] Black (Web Bot): {opp_move.to_ucci()} ({opp_wxf})"
                )

                # Record opponent move as an expert/teacher move
                learning_session.record_selected_move(board, opp_move)
                board.apply_move(opp_move)
                move_index += 1

                winner = board.get_winner()
                if winner or res.get("gameOver"):
                    if not winner and res.get("gameOver"):
                        txt = res["gameOver"].lower()
                        winner = "Black" if "black" in txt else "Draw"
                    break

            winner = winner or board.get_winner() or "Draw"
            print(f"\n[online-agent] Game {game_idx} finished! Result: {winner}")

            # Online Learning & Model Update
            learned_samples = learning_session.finalize(winner)
            if learned_samples:
                replay_buffer = replay_store.load()
                replay_buffer.extend(learned_samples)
                replay_store.save(replay_buffer.samples())

                steps = max(24, min(96, len(learned_samples) * 2 + 16))
                metrics = incremental_train_on_samples(
                    replay_buffer.samples(),
                    device=ai.device,
                    checkpoint_path=checkpoint_path,
                    training_config=TrainingConfig(
                        learning_rate=1e-3, weight_decay=1e-4
                    ),
                    steps=steps,
                    batch_size=min(64, max(16, len(learned_samples))),
                    recent_samples=learned_samples,
                    recent_weight=5,
                    verbose=True,
                )
                print(
                    f"[online-agent] Checkpoint updated with {len(learned_samples)} samples: {metrics}"
                )

                # Refresh AI player with latest weights
                ai = AIPlayer(
                    is_red=True,
                    num_simulations=num_simulations,
                    checkpoint_path=checkpoint_path,
                    mistake_memory=mistake_store.load(),
                )

    finally:
        client.stop()


def main():
    parser = argparse.ArgumentParser(description="Automated online Xiangqi AI training")
    parser.add_argument("--games", type=int, default=1, help="Number of games to play")
    parser.add_argument(
        "--simulations", type=int, default=800, help="MCTS simulations per move"
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=300,
        help="Maximum half-moves (plies) per game (default: 300, i.e. 150 rounds)",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        choices=range(1, 11),
        metavar="[1-10]",
        help="Bot difficulty level on play.xiangqi.com (1 to 10, default: 1)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Path to checkpoint",
    )
    args = parser.parse_args()

    run_online_training_session(
        num_games=args.games,
        num_simulations=args.simulations,
        checkpoint_path=Path(args.checkpoint),
        max_moves=args.max_moves,
        level=args.level,
    )


if __name__ == "__main__":
    main()
