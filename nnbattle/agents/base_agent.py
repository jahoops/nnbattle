# FILE: agents/base_agent.py

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, team):
        self.team = team

    @abstractmethod
    def select_move(self, game):
        pass


class Agent(ABC):
    @abstractmethod
    def select_move(self, board):
        """
        Given the current board state, return the column number (0-6) where the agent wants to drop its piece.
        """
        pass

