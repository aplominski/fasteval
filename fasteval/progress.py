import os

from tqdm import tqdm


class ProgressBar:
    def __init__(self, total: int, desc: str = "", disable: bool | None = None):
        if total < 0:
            raise ValueError(f"total must be >= 0, got {total}")
        if disable is None:
            disable = not os.isatty(1) or os.environ.get("TERM") == "dumb"
        self._closed = False
        try:
            self._pbar = tqdm(
                total=total,
                desc=desc,
                unit="sample",
                disable=disable,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )
        except Exception:
            self._pbar = None
            raise

    def update(self, n: int = 1):
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if self._pbar is not None:
            self._pbar.update(n)

    def set_postfix(self, **kwargs):
        if self._pbar is not None:
            self._pbar.set_postfix(**kwargs)

    def close(self):
        if not self._closed:
            self._closed = True
            if self._pbar is not None:
                self._pbar.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
