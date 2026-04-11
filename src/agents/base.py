from abc import ABC, abstractmethod
from typing import TypeVar

S = TypeVar("S")


class BaseAgent(ABC):
    """
    Contract every agent in this platform must satisfy.

    Agents receive the current state, perform their work, and return
    the (mutated) state.  The type parameter S is bound at the call site
    so that sub-graphs can use their own state type without losing
    static-analysis coverage.
    """

    @abstractmethod
    def run(self, state: S) -> S: ...
