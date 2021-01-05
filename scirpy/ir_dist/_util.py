from collections.abc import MutableMapping
import numpy as np


class SetDict(MutableMapping):
    """A dictionary that supports set operations"""

    def __init__(self, *args, **kwargs):
        self.store = dict(*args, **kwargs)
        # self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __or__(self, other):
        if isinstance(other, set):
            raise NotImplementedError(
                "Cannot combine SetDict and set using 'or' (wouldn't know how to handle the score)"
            )
        elif isinstance(other, SetDict):
            return SetDict(
                (
                    (k, np.min(self.store.get(k, np.inf), other.get(k, np.inf)))
                    for k in set(self.store) | set(other.store)
                )
            )
        else:
            raise NotImplementedError("Operation implementedonly for SetDict. ")

    def __ror__(self, other):
        raise NotImplementedError()

    def __and__(self, other):
        if isinstance(other, set):
            return SetDict(((k, self.store[k]) for k in set(self.store) & other))
        elif isinstance(other, SetDict):
            return SetDict(
                (
                    (k, np.max(self.store.get(k, np.inf), other.get(k, np.inf)))
                    for k in set(self.store) & set(other.store)
                )
            )
        else:
            raise NotImplementedError(
                "Operation implemented only for SetDict and set. "
            )

    def __rand__(self, other):
        raise NotImplementedError()
