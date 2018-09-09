import os
from contextlib import contextmanager
from pathlib import Path
from typing import Union


@contextmanager
def pushd(new_dir: Union[Path, str]):
    previous_dir = os.getcwd()
    os.chdir(str(new_dir))
    try:
        yield
    finally:
        os.chdir(previous_dir)
