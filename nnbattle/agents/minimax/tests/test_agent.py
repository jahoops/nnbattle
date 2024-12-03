# FILE: minimax/tests/test_agent.py
import unittest

from nnbattle.agents.minimax import MinimaxAgent
from nnbattle.game.connect_four_game import ConnectFourGame 


class TestMinimaxAgent(unittest.TestCase):
    def setUp(self):
        self.game = ConnectFourGame()
        self.agent = MinimaxAgent(depth=2)

    def test_select_move(self):
        move = self.agent.select_move(self.game)
        self.assertIn(move, range(7))  # Assuming 7 columns (0-6)

    def test_win_detection(self):
        # Simulate a simple win scenario
        PLAYER_PIECE = 1  # Define PLAYER_PIECE
        for i in range(4):
            row = self.game.get_next_open_row(0)
            self.game.drop_piece(row, 0, PLAYER_PIECE)
        self.assertTrue(self.game.check_win(PLAYER_PIECE))

if __name__ == '__main__':
    unittest.main()