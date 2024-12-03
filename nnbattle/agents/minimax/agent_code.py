# agents/agent_minimax/agent_code.py

import math
import random
import logging
import copy
from nnbattle.agents.base_agent import BaseAgent
from nnbattle.constants import EMPTY, ROW_COUNT, COLUMN_COUNT

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

from nnbattle.game.connect_four_game import ConnectFourGame 
from nnbattle.constants import RED_TEAM, YEL_TEAM

class MinimaxAgent(BaseAgent):
    def __init__(self, depth=4, team=YEL_TEAM):
        """
        Initializes the MinimaxAgent with a specified search depth and team number.
        
        :param depth: The depth to which the Minimax algorithm will search.
        :param team: The team number (1 or 2) that the agent is playing for.
        """
        super().__init__(team)
        self.depth = depth
        self.team = team  # Assign team to the agent
        logger.info(f"MinimaxAgent initialized with team {self.team} and depth {self.depth}")

    def select_move(self, game: ConnectFourGame):
        """
        Selects the best move by running the Minimax algorithm with Alpha-Beta pruning.
        
        :param game: The current state of the game.
        :return: The column number (0-6) where the agent decides to drop its piece.
        """
        score, column = self.minimax(game, self.depth, -math.inf, math.inf, True)
        logger.info(f"Selected move: {column} with score: {score}")
        return column

    def minimax(self, game: ConnectFourGame, depth, alpha, beta, maximizingPlayer):
        """
        The Minimax algorithm with Alpha-Beta pruning.
        
        :param game: The current game state.
        :param depth: The current depth in the game tree.
        :param alpha: The alpha value for pruning.
        :param beta: The beta value for pruning.
        :param maximizingPlayer: Boolean indicating if the current layer is maximizing or minimizing.
        :return: Tuple of (score, column)
        """
        valid_moves = game.get_valid_moves()
        result = game.get_game_state()
        if result != "ONGOING":
            if result == self.team:
                return (math.inf, None)
            elif result == 3 - self.team:
                return (-math.inf, None)
            else:  # Game is over, no more valid moves
                return (0, None)
        elif depth == 0:  # Depth is zero
            return (self.score_position(game), None)
        else:
            if maximizingPlayer:
                value = -math.inf
                best_column = random.choice(valid_moves)
                for col in valid_moves:
                    temp_game = copy.deepcopy(game)
                    move_successful = temp_game.make_move(col, self.team)
                    if not move_successful:
                        continue  # Skip invalid moves
                    new_score, _ = self.minimax(temp_game, depth-1, alpha, beta, False)
                    if new_score > value:
                        value = new_score
                        best_column = col
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
                return value, best_column
            else:
                value = math.inf
                best_column = random.choice(valid_moves)
                for col in valid_moves:
                    temp_game = copy.deepcopy(game)  # Changed from game.new_game() to copy.deepcopy(game)
                    move_successful = temp_game.make_move(col, 3 - self.team)
                    if not move_successful:
                        continue  # Skip invalid moves
                    new_score, _ = self.minimax(temp_game, depth-1, alpha, beta, True)
                    if new_score < value:
                        value = new_score
                        best_column = col
                    beta = min(beta, value)
                    if alpha >= beta:
                        break
                return value, best_column

    def score_position(self, game: ConnectFourGame):
        """
        Evaluates the board and returns a score from the agent's perspective.
        
        :param game: The current game state.
        :return: Integer score representing the desirability of the board.
        """
        score = 0
        board = game.get_board()
        
        ## Score center column
        center_array = [int(i) for i in list(board[:, COLUMN_COUNT//2])]
        center_count = center_array.count(self.team)
        score += center_count * 3  # Weight center positions higher
        
        ## Score Horizontal
        for r in range(ROW_COUNT):
            row_array = [int(i) for i in list(board[r,:])]
            for c in range(COLUMN_COUNT-3):
                window = row_array[c:c+4]
                score += self.evaluate_window(window)
        
        ## Score Vertical
        for c in range(COLUMN_COUNT):
            col_array = [int(i) for i in list(board[:,c])]
            for r in range(ROW_COUNT-3):
                window = col_array[r:r+4]
                score += self.evaluate_window(window)
        
        ## Score positive sloped diagonals
        for r in range(ROW_COUNT-3):
            for c in range(COLUMN_COUNT-3):
                window = [board[r+i][c+i] for i in range(4)]
                score += self.evaluate_window(window)
        
        ## Score negative sloped diagonals
        for r in range(3, ROW_COUNT):
            for c in range(COLUMN_COUNT-3):
                window = [board[r-i][c+i] for i in range(4)]
                score += self.evaluate_window(window)
        
        return score
    
    def evaluate_window(self, window):
        """
        Evaluates a 4-cell window and assigns a score based on the contents.
        
        :param window: List of 4 cells from the board.
        :return: Integer score for the window.
        """
        score = 0
        opp_team = 3 - self.team
        
        if window.count(self.team) == 4:
            score += 100
        elif window.count(self.team) == 3 and window.count(EMPTY) == 1:
            score += 5
        elif window.count(self.team) == 2 and window.count(EMPTY) == 2:
            score += 2
        
        if window.count(opp_team) == 3 and window.count(EMPTY) == 1:
            score -= 4
        
        return score