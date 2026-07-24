import sys

from bootstrap import setup_project_imports


def main():
    setup_project_imports()

    # Check arguments to decide which mode to run (GUI or CLI)
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        try:
            from my_chess.chess_ui import cli_game

            print("Starting CLI Game...")
            cli_game.main()
        except ImportError as e:
            print(f"Error importing CLI game: {e}")
            sys.exit(1)
    elif len(sys.argv) > 1 and sys.argv[1] == "train":
        try:
            from my_chess.chess_ai import train

            sys.argv = [sys.argv[0], *sys.argv[2:]]
            train.main()
        except ImportError as e:
            print(f"Error importing training entrypoint: {e}")
            sys.exit(1)
    else:
        try:
            from my_chess.chess_ui import win_game

            args = {arg.lower() for arg in sys.argv[1:]}
            enable_ai = bool({"ai", "ai-red", "ai-black"} & args)
            ai_is_red = "ai-red" in args
            enable_learning = "learn" in args
            print("Starting GUI Game...")
            win_game.main(
                enable_ai=enable_ai,
                ai_is_red=ai_is_red,
                enable_learning=enable_learning,
            )
        except ImportError as e:
            print(f"Error importing GUI game: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
