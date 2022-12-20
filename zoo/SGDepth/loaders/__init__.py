import torch
import itertools as it

class LoaderList(object):
    def __init__(self, loaders):
        self.loaders = tuple(loaders)

    def __iter__(self):
        raise NotImplementedError()

class FixedLengthLoaderList(LoaderList):
    def __init__(self, loaders, length):
        super().__init__(loaders)

        self.length = length

    def __iter__(self):
        infinite_iters = tuple(
            self._make_infinite(domain_idx, loader)
            for domain_idx, loader in enumerate(self.loaders)
        )

        length_iter = range(self.length)

        for batch_idx, *group in zip(length_iter, *infinite_iters):
            yield tuple(group)

    def _make_infinite(self, domain_idx, loader):
        while True:
            for batch in loader:
                batch['domain_idx'] = torch.tensor(domain_idx)

                yield batch


class ChainedLoaderList(LoaderList):
    def __iter__(self):
        for domain_idx, loader in enumerate(self.loaders):
            for batch in loader:
                batch['domain_idx'] = torch.tensor(domain_idx)

                yield (batch, )
