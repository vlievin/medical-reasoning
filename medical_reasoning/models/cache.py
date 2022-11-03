import os
import shutil
import time
from copy import copy
from pathlib import Path
from typing import Any
from typing import Callable

import click
import dill
import loguru
from datasets.fingerprint import Hasher


def update_hash(obj: Any, hasher: Hasher):
    if isinstance(obj, (set, tuple, list)):
        for el in obj:
            update_hash(el, hasher)
    elif isinstance(obj, dict):
        for k, v in sorted(obj.items(), key=lambda x: x[0]):
            update_hash(k, hasher)
            update_hash(v, hasher)
    else:
        hasher.update(obj)


class CachedFunction(object):
    def __init__(self, cache_dir: os.PathLike, reset_cache: bool = False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        if self.cache_dir.exists():
            n_files = len(list(self.cache_dir.iterdir()))
            loguru.logger.info(
                f"Using cache at {self.cache_dir}, found {n_files} cached files"
            )

        if reset_cache:
            if self.cache_dir.exists():
                n_files = len(list(self.cache_dir.iterdir()))
                size = sum(f.stat().st_size for f in self.cache_dir.iterdir())
                msg = (
                    f"Are you sure you want to erase the cache "
                    f"({n_files} files, {size / 1e6} MB)?"
                )
                if click.confirm(msg, default=False):
                    shutil.rmtree(self.cache_dir)
                    self.cache_dir.mkdir(exist_ok=True, parents=True)

    def __call__(
            self, fn: Callable, *args, retries: bool = True, **kwargs
    ) -> (Any, bool):

        # save the arguments
        data = copy(kwargs)
        data["__args__"] = list(args)
        data["__fn__"] = fn

        # fingerprint
        hasher = Hasher()
        update_hash(data, hasher)
        fingerprint = hasher.hexdigest()
        filename = f"{fingerprint}.pkl"
        cache_file = self.cache_dir / filename

        if cache_file.exists():
            return dill.load(open(cache_file, "rb")), True
        else:
            if not retries:
                result = fn(*args, **kwargs)
            else:
                sleep_time = 1
                n_tries = 0
                while True:
                    try:
                        result = fn(*args, **kwargs)
                        break
                    except Exception as exc:
                        time.sleep(sleep_time)
                        sleep_time *= 2
                        n_tries += 1
                        if n_tries % 10 == 0 and n_tries > 0:
                            loguru.logger.warning(
                                f"Failed to run {fn} after {n_tries} tries: {exc}")

            dill.dump(result, open(cache_file, "wb"))
            return result, False
