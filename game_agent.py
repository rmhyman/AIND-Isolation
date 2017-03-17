"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
from random import randint
import logging

logging.basicConfig(filename='debug.log',format='%(levelname)s:\n%(message)s',level=logging.DEBUG)

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    s1 = set(game.get_legal_moves(player))
    s2 = set(game.get_legal_moves(game.get_opponent(player)))
    return overlapping_move_count(s1,s2)


def overlapping_move_count(s1,s2):
    '''
    This heuristic returns the number of non overlapping next moves player 1 has with player 2
    :param s1: set of moves for player 1
    :param s2: set of moves for player
    :return: The number of moves reachable by player 1 not reachable by player 2
    '''
    return float(len(s1&s2))


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = 0.
        self.best_current_move = (-1,-1)
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters

        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left


        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring


            self.best_current_move = (-1,-1)
            random_choice_threshold = 1
            mid_game_boundary = 4
            end_game_search_depth = 5
            if len(legal_moves) == 0:
                return self.best_current_move
            max_depth = -1
            self.best_current_move = legal_moves[randint(0, len(legal_moves) - 1)]

            if self.method == 'minimax':
                if self.iterative:
                    self.search_depth = 6
                    for depth_level in range(1,self.search_depth + 1):
                        _, self.best_current_move = self.minimax(game, depth_level, True)
                else:
                    _, self.best_current_move = self.minimax(game,self.search_depth,True)
            else:

                if self.iterative:
                    self.search_depth = 12
                    for depth_level in range(1,self.search_depth+1):
                        _,self.best_current_move = self.alphabeta(game,depth_level,True)
                        max_depth = depth_level


                else:
                    _,_ = self.alphabeta(game,self.search_depth,True)




        except Timeout:
            if self.best_current_move == None:
                self.best_current_move = legal_moves[0]



        # Return the best move from the last completed search iteration
        logging.info("max depth level: {}".format(max_depth))
        return self.best_current_move

    def find_midpoint(self,game):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        width = game.width
        height = game.height
        return (int(height/2),int(width/2))
    def terminial_test(self,game):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        moves = game.get_legal_moves()
        return len(moves) == 0

    def perform_action(self,game,depth):
        '''

        :param game:
        :param depth:
        :return: score: (float), best_move: tuple(int,int)

        This function simulates the actions being carried as you traverse
        the minimax tree.  We perform the max/min action based on the depth level.
        '''
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        if self.terminial_test(game) or depth > self.search_depth:
            return self.score(game,self),game.get_player_location(self)


        moves = game.get_legal_moves()
        score, results =0., []
        for move in moves:
            updated_board = game.forecast_move(move)
            score,_ = self.perform_action(updated_board,depth + 1)
            results.append([score,move])

            results = sorted(results, key=lambda v: v[0]) if depth % 2 == 0 \
                        else sorted(results, key=lambda v: v[0],reverse=True)

        return results[0]



    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        moves = game.get_legal_moves()

        #logging.info("Root's Legal moves:%s",moves)
        #logging.debug(game.to_string())

        self.search_depth = depth
        return self.perform_action(game,1)









    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        moves = game.get_legal_moves()
        self.search_depth = depth

        f,alpha,beta,self.best_current_move = self.calculate_value(game, 1,alpha,beta)

        return f, self.best_current_move

    def calculate_value(self,game, depth,alpha=float("-inf"), beta=float("inf")):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if self.terminial_test(game) or depth > self.search_depth:
            leaf_score = self.score(game,self)
            return leaf_score, alpha, beta,game.get_player_location(self)

        moves = game.get_legal_moves()

        v = (float('-inf'),float('inf'))[depth % 2 == 0]

        results = []
        for move in moves:
            updated_board = game.forecast_move(move)
            tmp, _, _,_ = self.calculate_value(updated_board, depth + 1, alpha, beta)

            if depth % 2 == 0 :
                v = (min(v, tmp))
                if v <= alpha:
                    return v, alpha, beta,updated_board.get_player_location(self)
                beta = min(beta,v)
                results.append([beta, move])
            else:
                v = (max(v, tmp))
                if v >= beta:
                    return v, alpha, beta, updated_board.get_player_location(self)
                alpha = max(alpha, v)
                results.append([alpha,move])

        best_action = self.find_best_move(results,beta) if depth % 2 == 0 else self.find_best_move(results,alpha)


        return v, alpha, beta,best_action


    def find_best_move(self,results,value):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        actions = [a for a in results if a[0] == value]
        return actions[0][1]

