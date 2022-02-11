from subprocess import run
import sys
import pathlib

import ms_reader

def main():
    """The main routine"""

    path_to_app = pathlib.Path(ms_reader.__path__[0])
    path_to_app = path_to_app / "app.py"
    run(["streamlit", "run", str(path_to_app)])

if __name__ == "__main__":
    sys.exit(main())
