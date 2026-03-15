#Zijie Zhang, Sep.24/2023

import numpy as np
import socket, pickle
import time
from reversi import reversi

class TimeUp(Exception):
    pass

def evaluate(board, root_turn):
    # simple piece difference from root player's perspective
    return int(np.sum(board == root_turn) - np.sum(board == -root_turn))

def has_any_move(game, board, turn):
    game.board = board
    for i in range(8):
        for j in range(8):
            if game.step(i, j, turn, False) > 0:
                return True
    return False

def is_terminal(game, board):
    # terminal if board full OR no moves for both players
    if not (board == 0).any():
        return True
    return (not has_any_move(game, board, 1)) and (not has_any_move(game, board, -1))

def alphabeta(game, board, depth, alpha, beta, turn, root_turn, deadline):
    # time cutoff
    if time.perf_counter() >= deadline:
        raise TimeUp

    if depth == 0 or is_terminal(game, board):
        return evaluate(board, root_turn)

    game.board = board
    moves = [(i, j) for i in range(8) for j in range(8) if game.step(i, j, turn, False) > 0]

    # pass if no legal moves
    if not moves:
        return alphabeta(game, board, depth - 1, alpha, beta, -turn, root_turn, deadline)

    if turn == root_turn:  # maximizing
        value = -10**18
        for (i, j) in moves:
            if time.perf_counter() >= deadline:
                raise TimeUp

            new_board = np.array(board, copy=True)
            game.board = new_board
            game.step(i, j, turn, True)  # commit on copy

            value = max(
                value,
                alphabeta(game, game.board, depth - 1, alpha, beta, -turn, root_turn, deadline)
            )
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value
    else:  # minimizing
        value = 10**18
        for (i, j) in moves:
            if time.perf_counter() >= deadline:
                raise TimeUp

            new_board = np.array(board, copy=True)
            game.board = new_board
            game.step(i, j, turn, True)

            value = min(
                value,
                alphabeta(game, game.board, depth - 1, alpha, beta, -turn, root_turn, deadline)
            )
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value

def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    game = reversi()

    while True:
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        if turn == 0:
            game_socket.close()
            return

        # Debug
        # print(turn)
        # print(board)

        # Minimax algorithm w/ alpha-beta pruning + time cutoff
        x = -1
        y = -1
        best_val = -10**18
        depth = 4

        # buffer under 5 seconds
        deadline = time.perf_counter() + 4.8

        game.board = board
        try:
            for i in range(8):
                for j in range(8):
                    # quick time check at root too
                    if time.perf_counter() >= deadline:
                        raise TimeUp

                    cur = game.step(i, j, turn, False)
                    if cur > 0:
                        # Fallback: lock in the first legal move so we never pass by accident
                        if x == -1:
                            x, y = i, j

                        new_board = np.array(board, copy=True)
                        game.board = new_board
                        game.step(i, j, turn, True)

                        val = alphabeta(
                            game, game.board, depth - 1,
                            -10**18, 10**18,
                            -turn, turn,
                            deadline
                        )

                        if val > best_val:
                            best_val = val
                            x, y = i, j
        except TimeUp:
            # If time runs out, we just play the best move found so far (or the fallback legal move)
            pass

        game_socket.send(pickle.dumps([x, y]))

if __name__ == '__main__':
    main()