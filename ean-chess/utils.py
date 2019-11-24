import numpy as np
import chess
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import json
import os

RESOURCES_PATH = os.path.abspath(__file__ + '../resources')

# CONSTANTS #
piece_map = {
    1: "PAWN",
    2: "KNIGHT",
    3: "BISHOP",
    4: "ROOK",
    5: "QUEEN",
    6: "KING"
}

# adjust these values if need be
piece_value_map = {
    1: 1,
    2: 3,
    3: 3,
    4: 5,
    5: 9,
    6: 1000
}

promotion_value_map = {
    'n': 3,
    'b': 3,
    'r': 5,
    'q': 9
}


# UTILITY FUNCTIONS
def load_action_map():
    with open(os.path.join(RESOURCES_PATH, 'action_map.json')) as fp:
        return json.load(fp)


def update_action_map(num_iter=1000):
    """
    Get all the possible actions from num_iter simulated games
    Update the current map with any new actions found
    """
    board = chess.Board()
    map_path = os.path.join(RESOURCES_PATH, 'action_map.json')
    # if map already exists, load it; otherwise make a brand new one
    try:
        with open(map_path) as fp:
            action_set = set(json.load(fp).values())
    except FileNotFoundError:
        action_set = set()
    for i in range(num_iter):
        while not board.is_stalemate() and not board.is_game_over():
            legal_moves = [str(x) for x in board.legal_moves]
            action_set.update(set(legal_moves))
            np.random.shuffle(legal_moves)
            curr_action = chess.Move.from_uci(np.random.choice(legal_moves))
            board.push(curr_action)

    action_map = {ind: action for ind, action in enumerate(sorted(action_set))}
    with open(map_path, 'w') as fp:
        json.dump(action_map, fp)
    return action_map


def get_board_vector(board):
    """
    Get a length-64 vector representing the board
    Vector values represent the type of piece at that positon
    White pieces are positive int's, black player's pieces are negative
    :param board: chess.Board
    :return: np.array (64,)
    """
    vec = np.zeros((64,), dtype="float32")
    for pos in range(64):
        piece = board.piece_at(pos)
        # white pieces are positive, black pieces are negative
        if piece:
            if piece.color:
                piece_mult = 1
            else:
                piece_mult = -1
            vec[pos] = piece_mult*int(piece.piece_type)
    return vec


def make_move(board, move):
    """Return a new board vector, reward, and whether it finishes the game
    Assume the move is already validated and legal"""
    reward = 0
    done = False
    player = board.turn
    init_pos, new_pos, promotion = move[:2], move[2:4], move[4:5]
    piece_to_move = board.piece_at(
        chess.SQUARE_NAMES.index(init_pos)).piece_type

    if board.piece_at(chess.SQUARE_NAMES.index(new_pos)):
        reward += piece_value_map[board.piece_at(
            chess.SQUARE_NAMES.index(new_pos)).piece_type]
    if promotion:
        reward += promotion_value_map[promotion]

    board.push(chess.Move.from_uci(move))
    if board.is_checkmate():
        reward += piece_value_map[6]
    elif board.is_check():
        reward += piece_value_map[6] / 2

    # negative reward for putting piece in threatened position
    num_attackers = len(board.attackers(not player,
                                        chess.SQUARE_NAMES.index(new_pos)))
    reward -= (num_attackers/64 * piece_value_map[piece_to_move])

    new_state = get_board_vector(board)
    if board.is_game_over():
        done = True
    return new_state, reward, done


def neural_network_model(input_size, output_size, layers=None,
                         learning_rate=.005):
    network = input_data(shape=[None, input_size], name='input')
    if layers:
        for layer_size in layers:
            network = fully_connected(network, layer_size, activation='relu')
    network = fully_connected(network, output_size, activation='softmax')
    network = regression(network,
                         optimizer='adam',
                         learning_rate=learning_rate,
                         loss='mean_square',
                         name='targets')
    model = tflearn.DNN(network, tensorboard_dir='logs')
    return model

