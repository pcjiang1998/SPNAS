from typing import Dict
from pymoo.core.algorithm import Algorithm
from pymoo.core.callback import Callback
from recorder import Recorder
from utils.log_util import Logger


class MyCallback(Callback):

    def __init__(self, verbose=False) -> None:
        super().__init__()
        self.count = 0
        self.history: Dict[str, Recorder] = {}
        self.current: Dict[str, Recorder] = {}
        self.verbose: bool = verbose

    def notify(self, algorithm: Algorithm):
        if self.verbose:
            Logger.info(f'Gen {algorithm.n_gen}: {self.count}, '
                        f'duplicates {len(self.current)}, '
                        f'history length {len(self.history)}')
        else:
            Logger.info(f'Gen {algorithm.n_gen}')
        self.count = 0
        self.current = {}

    def reset(self):
        self.count = 0
        self.history: Dict[str, Recorder] = {}
        self.current: Dict[str, Recorder] = {}
