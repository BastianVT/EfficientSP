from abc import abstractmethod
from enum import Enum

from .base_solution import Solution


class ProblemType(Enum):
    PCTSP = 1
    KP = 2


class Problem:

    def __init__(self, container: dict):
        self.container = container

    def __getitem__(self, item):
        return self.container[item]

    def __setitem__(self, key, value):
        self.container[key] = value

    def __delitem__(self, key):
        del self.container[key]

    @abstractmethod
    def get_preferred_solution(self) -> Solution:
        pass

    @abstractmethod
    def build_solution(self, y):
        pass

    @staticmethod
    @abstractmethod
    def read_data(filepath):
        pass
