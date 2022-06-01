import collections
import math


def _add(a, b):
    a = 0 if a is None else a
    b = 0 if b is None else b
    return a + b


def _min(a, b):
    a = float('inf') if a is None else a
    b = float('inf') if b is None else b
    return min(a, b)


class PointerError(IndexError):
    pass


class SegmentTree(collections.abc.Collection):

    def __init__(self, maxlen, /, operator):
        self._maxlen = maxlen
        self._operator = operator
        self._height = math.ceil(math.log2(self._maxlen))
        self._capacity = 2 ** self._height
        self._nodes = [None for _ in range(2 * self._capacity)]  # perfect binary tree
        self._len = 0
        self._next_index = 0

    @property
    def maxlen(self):
        return self._maxlen

    def __repr__(self):
        return '%s(%s, maxlen=%d)' % (
            self.__class__.__name__,
            repr(self._nodes[self._capacity:self._capacity + self._len]),
            self._maxlen
        )

    def __iter__(self):
        return iter(self._nodes[self._capacity:self._capacity + self._len])

    def __len__(self):
        return self._len

    def __contains__(self, x):
        return x in self._nodes[self._capacity:self._capacity + self._len]

    @property
    def root(self):
        return self._nodes[1]

    def _pointer_checker(self, pointer, assignment=False):
        if pointer < 1 or pointer >= len(self._nodes):
            raise PointerError(
                "'%s' pointer out of range" % self.__class__.__name__
            )
        if assignment and pointer < self._capacity:
            raise PointerError(
                "'%s' assignment pointer out of range" % self.__class__.__name__
            )

    def get(self, pointer):
        self._pointer_checker(pointer)
        return self._nodes[pointer]

    def set(self, pointer, value):
        self._pointer_checker(pointer, assignment=True)
        self._nodes[pointer] = value
        while pointer >= 2:
            pointer //= 2
            self._nodes[pointer] = self._operator(
                self._nodes[2 * pointer],
                self._nodes[2 * pointer + 1]
            )

    def _index_checker(self, index):
        if index < -self._len or index >= self._len:
            raise IndexError(
                "'%s' index out of range" % self.__class__.__name__
            )

    def __getitem__(self, index):
        self._index_checker(index)
        index %= self._len
        pointer = index + self._capacity
        return self._get(pointer)

    def __setitem__(self, index, value):
        self._index_checker(index)
        index %= self._len
        pointer = index + self._capacity
        self._set(pointer, value)

    def append(self, value):
        index = self._next_index
        pointer = index + self._capacity
        self._set(pointer, value)
        self._len = min(self._len + 1, self._maxlen)
        self._next_index = (self._next_index + 1) % self._maxlen


class SumSegmentTree(SegmentTree):

    def __init__(self, maxlen):
        super().__init__(maxlen, operator=_add)

    def sum(self):
        return self.root

    def retrieve_index(self, upperbound, eps):
        # Point to the root
        pointer = 1
        # While non-leaf
        while pointer < self._capacity:
            left = 2 * pointer
            right = left + 1
            if self._nodes[left] > upperbound:
                # Point to the left child
                pointer = left
            else:
                upperbound -= self._nodes[left]
                # Point to the right child
                pointer = right
        return pointer - self._capacity


class MinSegmentTree(SegmentTree):

    def __init__(self, maxlen):
        super().__init__(maxlen, operator=_min)

    def min(self):
        return self.root
