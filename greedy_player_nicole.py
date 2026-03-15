import numpy as np
import socket, pickle
import time
from reversi import reversi

# Strategic weights: Prioritizes corners and avoids giving them away.
WEIGHTS = np.array([
    [100, -30,  10,   5,   5,  10, -30, 100],
    [-30, -50,  -2,  -2,  -2,  -2, -50, -30],
    [ 10,  -2,   5,   1,   1,   5,  -2,  10],
    [  5,  -2,   1,   1,   1,   1,  -2,   5],
    [  5,  -2,   1,   1,   1,   1,  -2,   5],
    [ 10,  -2,   5,   1,   1,   5,  -2,  10],
    [-30, -50,  -2,  -2,  -2,  -2, -50, -30],
    [100, -30,  10,   5,   5,  10, -30, 100]
])

class TournamentBot:
    def __init__(self):
        self.game = reversi()
        self.start_time = 0

    def evaluate(self, board, turn):
        empty_spaces = np.count_nonzero(board == 0)
        # ENDGAME: Maximize total piece lead
        if empty_spaces <= 12:
            return np.sum(board) * turn
        
        # MIDGAME: Positional strategy + Mobility Starvation
        positional_score = np.sum(board * WEIGHTS)
        opp_moves = self.get_valid_moves(board, -turn)
        mobility_score = -len(opp_moves) * 15
        
        return (positional_score + mobility_score) * turn
    
    def get_valid_moves(self, board, turn):
        valid_moves = []
        for i in range(8):
            for j in range(8):
                self.game.board = board.copy()
                if self.game.step(i, j, turn, commit=False) > 0:
                    valid_moves.append((i, j))
        return valid_moves
    
    def minimax(self, board, depth, alpha, beta, is_maximizing, turn):
        # 4.3s Emergency Exit
        if time.time() - self.start_time > 4.3:
            return self.evaluate(board, turn), None

        curr_p = turn if is_maximizing else -turn
        moves = self.get_valid_moves(board, curr_p)
        
        if depth == 0 or not moves:
            return self.evaluate(board, turn), None
        
        best_move = None
        if is_maximizing:
            max_eval = -float('inf')
            for m in moves:
                temp_board = board.copy()
                self.game.board = temp_board
                self.game.step(m[0], m[1], turn, commit=True)
                ev, _ = self.minimax(temp_board, depth - 1, alpha, beta, False, turn)
                if ev > max_eval:
                    max_eval, best_move = ev, m
                alpha = max(alpha, ev)
                if beta <= alpha: break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for m in moves:
                temp_board = board.copy()
                self.game.board = temp_board
                self.game.step(m[0], m[1], -turn, commit=True)
                ev, _ = self.minimax(temp_board, depth - 1, alpha, beta, True, turn)
                if ev < min_eval:
                    min_eval, best_move = ev, m
                beta = min(beta, ev)
                if beta <= alpha: break
            return min_eval, best_move

def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    bot = TournamentBot()
    
    # Initialize Log File
    log_filename = f"reversi_log_{int(time.time())}.txt"
    print(f"Logging to {log_filename}")

    while True:
        try:
            data = game_socket.recv(4096)
            if not data: break
            turn, board = pickle.loads(data)

            if turn == 0:
                print("Game ended.")
                game_socket.close()
                return
            
            bot.start_time = time.time()
            
            # --- ADAPTIVE DEPTH ---
            empty_spaces = np.count_nonzero(board == 0)
            target_depth = 4 # Standard
            if empty_spaces < 14:
                target_depth = 6 # Deeper for endgame
            
            score, move = bot.minimax(board, target_depth, -float('inf'), float('inf'), True, turn)

            if move is None:
                move = (-1, -1)
            
            elapsed = time.time() - bot.start_time
            
            # Formatting log and console output
            status = f"Turn: {turn} | Move: {move} | Eval: {score:.1f} | Time: {elapsed:.2f}s"
            print(status)
            with open(log_filename, "a") as f:
                f.write(status + "\n")

            game_socket.send(pickle.dumps(list(move)))

        except (EOFError, ConnectionResetError, KeyboardInterrupt):
            break

if __name__ == '__main__':
    main()