import copy as copy_mod

from tensorpack.dataflow.common import MapData
from tensorpack.dataflow.image import check_dtype, ExceptionHandler
from tensorpack.dataflow.imgaug.base import AugmentorList


class MapDataComponentTwice(MapData):
    """
    Apply two mappers/filters on a same datapoint component.

    Note:
        1. This dataflow itself doesn't modify the datapoints.
           But please make sure func doesn't modify its arguments in place,
           unless you're certain it's safe.
        2. If you discard some datapoints, ``len(ds)`` will be incorrect.

    Example:

        .. code-block:: none

            ds = Mnist('train)
            ds = MapDataComponent(ds, lambda img: img * 255, 0)
    """
    def __init__(self, ds, func0, func1, func2, index=0):
        """
        Args:
            ds (DataFlow): input DataFlow which produces either list or dict.
            func (TYPE -> TYPE|None): takes ``dp[index]``, returns a new value for ``dp[index]``.
                Return None to discard/skip this datapoint.
            index (int or str): index or key of the component.
        """
        self._index = index
        self._func0 = func0
        self._func1 = func1
        self._func2 = func2
        super(MapDataComponentTwice, self).__init__(ds, self._mapper)

    def _mapper(self, dp):
        r = self._func0(dp[self._index])
        if r is None:
            return r
        r1 = self._func1(r)
        r2 = self._func2(r)
        if r1 is None or r2 is None:
            return None
        dp = copy_mod.copy(dp)   # shallow copy to avoid modifying the datapoint
        if isinstance(dp, tuple):
            dp = list(dp)  # to be able to modify it in the next line
        dp[self._index] = r1
        dp.append(r2)
        return dp


class AugmentImageComponentTwice(MapDataComponentTwice):
    """
    Apply two different image augmentors on one image component.
    """

    def __init__(self, ds, augmentors0, augmentors1, augmentors2, index=0, copy=True, catch_exceptions=False):
        """
        Args:
            ds (DataFlow): input DataFlow.
            augmentors (AugmentorList): a list of :class:`imgaug.ImageAugmentor` to be applied in order.
            index (int): the index of the image component to be augmented in the datapoint.
            copy (bool): Some augmentors modify the input images. When copy is
                True, a copy will be made before any augmentors are applied,
                to keep the original images not modified.
                Turn it off to save time when you know it's OK.
            catch_exceptions (bool): when set to True, will catch
                all exceptions and only warn you when there are too many (>100).
                Can be used to ignore occasion errors in data.
        """
        self.augs0 = augmentors0 if isinstance(augmentors0, AugmentorList) else AugmentorList(augmentors0)
        self.augs1 = augmentors1 if isinstance(augmentors1, AugmentorList) else AugmentorList(augmentors1)
        self.augs2 = augmentors2 if isinstance(augmentors2, AugmentorList) else AugmentorList(augmentors2)
        self._copy = copy
        self._exception_handler = ExceptionHandler(catch_exceptions)
        super(AugmentImageComponentTwice, self).__init__(ds, self._aug_mapper0, self._aug_mapper1, self._aug_mapper2, index)

    def reset_state(self):
        self.ds.reset_state()
        self.augs0.reset_state()
        self.augs1.reset_state()
        self.augs2.reset_state()

    def _aug_mapper0(self, x):
        check_dtype(x)
        with self._exception_handler.catch():
            if self._copy:
                x = copy_mod.deepcopy(x)
            return self.augs0.augment(x)

    def _aug_mapper1(self, x):
        check_dtype(x)
        with self._exception_handler.catch():
            if self._copy:
                x = copy_mod.deepcopy(x)
            return self.augs1.augment(x)

    def _aug_mapper2(self, x):
        check_dtype(x)
        with self._exception_handler.catch():
            if self._copy:
                x = copy_mod.deepcopy(x)
            return self.augs2.augment(x)
