import os
from copy import copy
from pathlib import Path
from typing import Any
from typing import Callable

import dill
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
    def __init__(self, cache_dir: os.PathLike):
        self.cache_dir = Path(cache_dir)

    def __call__(self, fn: Callable, *args, **kwargs) -> (Any, bool):

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
            result = fn(*args, **kwargs)
            dill.dump(result, open(cache_file, "wb"))
            return result, False