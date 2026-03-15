import numpy as np
import socket, pickle
from reversi import reversi

#Checks and returns all available moves
def get_avail_moves(game, board, turn):
    moves = []
    game.board = board
    for i in range(8):
        for j in range(8):
            if game.step(i, j, turn, False) > 0:
                moves.append((i, j))
    return moves

#Checks current score
def evaluate(board, turn):
    return np.sum(board) * turn

#Main Algorithm
def minimax(game, board, turn):
    ideal_score = -float('inf')
    ideal_move = (-1, -1)

    avail_moves = get_avail_moves(game, board, turn)

    if not avail_moves:
        return (-1, -1)
    
    #Make temp game for each move, check opponent responses
    for move in avail_moves:
        temp_game = reversi()
        temp_game.board = np.copy(board)

        temp_game.step(move[0], move[1], turn, True)
        new_board = np.copy(temp_game.board)

        opponent_moves = get_avail_moves(temp_game, new_board, -turn)

        if not opponent_moves:
            score = evaluate(new_board, turn)

        #Opponent move sim
        else:
            worst_score = float('inf')

            for opp_move in opponent_moves:
                opp_game = reversi()
                opp_game.board = np.copy(new_board)

                opp_game.step(opp_move[0], opp_move[1], -turn, True)
                opp_board = np.copy(opp_game.board)

                score = evaluate(opp_board, turn)
                worst_score = min(worst_score, score)

            score = worst_score
        
        #Track current best move
        if score > ideal_score:
            ideal_score = score
            ideal_move = move

    return ideal_move

def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    game = reversi()

    while True:

        #Receive play request from the server
        #turn : 1 --> you are playing as white | -1 --> you are playing as black
        #board : 8*8 numpy array
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        #Turn = 0 indicates game ended
        if turn == 0:
            game_socket.close()
            return
        
        #Debug info
        print(turn)
        print(board)

        #Minimax Algorithm
        x, y = minimax(game, board, turn)
        

        #Send your move to the server. Send (x,y) = (-1,-1) to tell the server you have no hand to play
        game_socket.send(pickle.dumps([x,y]))
        
if __name__ == '__main__':
    main()
