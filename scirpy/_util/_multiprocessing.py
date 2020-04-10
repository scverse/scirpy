import multiprocessing.pool as mpp
from tqdm import tqdm


class EnhancedPool(mpp.Pool):
    def istarmap(self, func, iterable, chunksize=1):
        """starmap-version of imap

        From https://stackoverflow.com/a/57364423/2340703. 
        """
        if self._state != mpp.RUN:
            raise ValueError("Pool not running")

        if chunksize < 1:
            raise ValueError("Chunksize must be 1+, not {0:n}".format(chunksize))

        task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
        try:
            result = mpp.IMapIterator(self._cache)
        except AttributeError:
            # python >=3.8 should pass `pool` instead of `cache`
            result = mpp.IMapIterator(self)
        self._taskqueue.put(
            (
                self._guarded_task_generation(
                    result._job, mpp.starmapstar, task_batches
                ),
                result._set_length,
            )
        )
        return (item for chunk in result for item in chunk)

    def starmap_progress(self, func, iterable, chunksize=1, total=None):
        """Implementation of starmap with progressbar"""
        return list(tqdm(self.istarmap(func, iterable), total=total))
