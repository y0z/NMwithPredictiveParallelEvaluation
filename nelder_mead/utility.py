import concurrent.futures
import json
import numpy as np
from functools import partial


class CacheError(Exception):
    pass


array64 = partial(np.array, dtype=np.float64)


class Function:
    def __init__(self, f, num_parallels=1):
        self.f = f
        self.keys = []
        self.values = []
        self.cache = {}
        self.count = 0
        self.num_parallels = num_parallels
        if num_parallels > 0:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_parallels)
            #self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.num_parallels)

    def bulk_call(self, xs, raise_if_not_found=False):
        ret = []
        if raise_if_not_found:
            for x in xs:
                f = self.executor.submit(self.__call__, x, raise_if_not_found, False)
                ret.append(f.result())
        else:
            for x in xs:
                f = self.executor.submit(self.__call__, x, raise_if_not_found, False)
                ret.append(f.result())
            self.count += int(len(xs) / self.num_parallels) \
                    + int(len(xs) % self.num_parallels != 0)
        return array64(ret)

    def check_cache(self, x):
        return tuple(x) in self.cache

    def __call__(self, x, raise_if_not_found=False, count=True):
        if self.check_cache(x):
            return self.cache[tuple(x)]
        elif raise_if_not_found:
            raise CacheError(x)
        ret = self.f(x)
        self.keys.append(x)
        self.values.append(ret)
        self.cache[tuple(x)] = ret
        if count:
            self.count += 1
        return ret
