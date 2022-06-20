from subprocess import run
import sys
import pathlib
from threading import Thread
import requests

import ms_reader


def get_last_version():
    """Get last package version."""
    try:
        pf_path = pathlib.Path(ms_reader.__file__).parent
        # Get the version from pypi
        response = requests.get('https://pypi.org/pypi/ms-reader/json')
        latest_version = response.json()['info']['version']
        with open(str(pathlib.Path(pf_path, "last_version.txt")), "w") as f:
            f.write(latest_version)
    except Exception:
        print("Error checking version from pypi")


def main():
    """The main routine"""

    thread = Thread(target=get_last_version)
    thread.start()

    path_to_app = pathlib.Path(ms_reader.__file__).parent
    path_to_app = path_to_app / "app.py"
    run(["streamlit", "run", str(path_to_app)])


if __name__ == "__main__":
    sys.exit(main())
