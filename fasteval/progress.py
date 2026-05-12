from tqdm import tqdm


class ProgressBar:
    def __init__(self, total: int, desc: str = ""):
        self._pbar = tqdm(
            total=total,
            desc=desc,
            unit="sample",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

    def update(self, n: int = 1):
        self._pbar.update(n)

    def set_postfix(self, **kwargs):
        self._pbar.set_postfix(**kwargs)

    def close(self):
        self._pbar.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
