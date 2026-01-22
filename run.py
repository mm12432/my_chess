import os
import sys
from pathlib import Path


def setup_path():
    """
    Add the project's parent directory to sys.path to ensure
    top-level package 'MyChess' can be imported.
    """
    # Get the directory containing this script (project root)
    root_dir = Path(__file__).resolve().parent
    # We need to add the PARENT of the root dir, so that 'MyChess'
    # can be resolved as a package.
    parent_dir = root_dir.parent
    if str(parent_dir) not in sys.path:
        print(f"Adding {parent_dir} to sys.path")
        sys.path.insert(0, str(parent_dir))


def main():
    setup_path()

    # Check arguments to decide which mode to run (GUI or CLI)
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        try:
            from MyChess.Chess_UI import cli_game

            print("Starting CLI Game...")
            cli_game.main()
        except ImportError as e:
            print(f"Error importing CLI game: {e}")
            sys.exit(1)
    else:
        try:
            from MyChess.Chess_UI import win_game

            print("Starting GUI Game...")
            win_game.main()
        except ImportError as e:
            print(f"Error importing GUI game: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
