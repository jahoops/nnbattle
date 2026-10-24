# game/connect_four_game.py

import copy
import logging
import numpy as np
from nnbattle.constants import RED_TEAM, YEL_TEAM, EMPTY, ROW_COUNT, COLUMN_COUNT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class InvalidMoveError(Exception):
    """Exception raised when an invalid move is made."""
    pass

class InvalidTurnError(Exception):
    """Exception raised when an invalid turn is attempted."""
    pass

class ConnectFourGame:
    def __init__(self):
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=np.int8)
        self.last_team = None
        self.enforce_turns = True  # Always enforce turns

    def new_game(self):
        """Creates and returns a new game instance."""
        new_game = ConnectFourGame()
        new_game.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=np.int8)
        new_game.last_team = None
        return new_game

    def reset(self):
        """Resets the game to initial state."""
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=np.int8)
        self.last_team = None
        return self.board.copy()

    def get_initial_state(self):
        """Return the initial state of the board."""
        return self.board.copy()

    def make_move(self, column: int, team: int) -> bool:
        """Make a move in the specified column for the given team."""
        logger.debug(f"Attempting move by Team {team} in column {column}. Last team: {self.last_team}")
        
        # Validate team
        if team not in [RED_TEAM, YEL_TEAM]:
            logger.error(f"Invalid team: {team}. Must be {RED_TEAM} or {YEL_TEAM}.")
            raise InvalidMoveError(f"Invalid team: {team}. Must be {RED_TEAM} or {YEL_TEAM}.")

        # Validate turn order
        if self.enforce_turns and self.last_team is not None:
            if self.last_team == team:
                logger.error(f"Invalid turn: team {team} cannot move twice in a row.")
                raise InvalidTurnError(f"Invalid turn: team {team} cannot move twice in a row.")
        
        # Validate move
        if not self.is_valid_move(column):
            logger.error(f"Invalid move: Column {column} is full or out of bounds.")
            raise InvalidMoveError(f"Invalid move: Column {column} is full or out of bounds.")

        # Make the move
        row = self.get_next_open_row(column)
        if row is None:
            return False

        # Update board and last_team
        self.board[row][column] = team
        self.last_team = team
        logger.debug(f"Move made by Team {team}. Updated last_team to {self.last_team}.")
        return True

    def is_valid_move(self, column):
        """Check if a move is valid."""
        if column < 0 or column >= COLUMN_COUNT:
            return False
        return self.board[0][column] == EMPTY

    def get_next_open_row(self, column):
        """Get the next available row in the given column starting from the bottom."""
        for r in range(ROW_COUNT-1, -1, -1):
            if self.board[r][column] == EMPTY:
                return r
        raise InvalidMoveError(f"Column {column} is full.")

    def check_win(self, team):
        """Check if the given team has won."""
        # Check horizontal
        for r in range(ROW_COUNT):
            for c in range(COLUMN_COUNT-3):
                if all(self.board[r][c+i] == team for i in range(4)):
                    return True

        # Check vertical
        for r in range(ROW_COUNT-3):
            for c in range(COLUMN_COUNT):
                if all(self.board[r+i][c] == team for i in range(4)):
                    return True

        # Check positive diagonal
        for r in range(ROW_COUNT-3):
            for c in range(COLUMN_COUNT-3):
                if all(self.board[r+i][c+i] == team for i in range(4)):
                    return True

        # Check negative diagonal
        for r in range(3, ROW_COUNT):
            for c in range(COLUMN_COUNT-3):
                if all(self.board[r-i][c+i] == team for i in range(4)):
                    return True

        return False

    def is_board_full(self):
        """Check if the board is full."""
        return not any(self.board[0][c] == EMPTY for c in range(COLUMN_COUNT))

    def get_game_state(self):
        """Return the current game state."""
        if self.check_win(RED_TEAM):
            return RED_TEAM
        elif self.check_win(YEL_TEAM):
            return YEL_TEAM
        elif self.is_board_full():
            return "Draw"
        return "ONGOING"

    def get_valid_moves(self, state=None):
        """Get a list of valid moves based on the current state."""
        if state is None:
            state = self.board
        valid_moves = []
        for col in range(COLUMN_COUNT):
            if state[0][col] == EMPTY:
                valid_moves.append(col)
        return valid_moves

    def get_next_state(self, current_state, action):
        """Return the next state after applying the action."""
        next_state = current_state.copy()
        for row in range(ROW_COUNT-1, -1, -1):
            if next_state[row][action] == EMPTY:
                next_state[row][action] = self.last_team  # Current team has already been set
                self.last_team = 3 - self.last_team  # Switch team
                break
        return next_state

    def get_board(self):
        """Return copy of current board state."""
        return self.board.copy()

    def get_value(self, state):
        """Evaluate the board state and return the value."""
        # Implement a heuristic or game outcome evaluation
        # For simplicity, return 1.0 for a win, -1.0 for a loss, 0.0 for draw
        game_state = self.get_game_state()
        if game_state == RED_TEAM:
            return 1.0
        elif game_state == YEL_TEAM:
            return -1.0
        else:
            return 0.0

    def is_terminal(self, state):
        """Check if the game is over."""
        return self.get_game_state() != "ONGOING"

    def board_to_string(self):
        """Return string representation of board with colored circles."""
        symbols = {EMPTY: '.', RED_TEAM: 'R', YEL_TEAM: 'Y'}
        board_str = ''
        for row in self.board:
            board_str += ' '.join(symbols[cell] for cell in row) + '\n'
        return board_str