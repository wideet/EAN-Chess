import numpy as np
import chess
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import warnings
warnings.filterwarnings("ignore")

piece_map = {
    1: "PAWN",
    2: "KNIGHT",
    3: "BISHOP",
    4: "ROOK",
    5: "QUEEN",
    6: "KING"
}


def get_board_vector(board):
    vec = np.zeros((64,))
    curr_player = board.turn
    for pos in range(64):
        piece = board.piece_at(pos)
        if piece:
            if piece.color == curr_player:
                mult = 1
            else:
                mult = -1
            vec[pos] = mult*int(piece.piece_type)
    return vec


def get_training_data(required_wins=10):
    board = chess.Board()
    training_vectors = []
    training_actions = []
    num_wins = 0

    while num_wins < required_wins:
        board.reset()
        white_player_vectors = []
        black_player_vectors = []
        white_player_actions = []
        black_player_actions = []
        while not board.is_stalemate() and not board.is_game_over():
            curr_player = board.turn
            curr_vector = get_board_vector(board)
            legal_moves = list(board.legal_moves)
            np.random.shuffle(legal_moves)
            curr_action = np.random.choice(legal_moves)
            action_str = str(curr_action)
            if curr_player:
                white_player_vectors.append(curr_vector)
                white_player_actions.append(action_str)
            else:
                black_player_vectors.append(curr_vector)
                black_player_actions.append(action_str)
            board.push(curr_action)
        if board.is_checkmate():
            # if it's white's turn, black won, so we want black's data
            if board.turn:
                training_vectors.extend(black_player_vectors)
                training_actions.extend(black_player_actions)
            else:
                training_vectors.extend(white_player_vectors)
                training_actions.extend(white_player_actions)
            num_wins += 1
            print(num_wins)
    return training_vectors, training_actions


def neural_network_model(input_size, output_size):

    network = input_data(shape=[None, input_size], name='input')
    network = fully_connected(network, 128, activation='relu')
    network = fully_connected(network, 256, activation='relu')
    network = fully_connected(network, 512, activation='relu')
    network = fully_connected(network, 256, activation='relu')
    network = fully_connected(network, 128, activation='relu')
    network = fully_connected(network, output_size, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=.0001,
                         loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_vectors, training_actions, action_map, model=False):
    X = np.array([i for i in training_vectors], dtype="float32")
    y = []
    for action in training_actions:
        vec = np.zeros((len(action_map),))
        np.put(vec, action_map[action], 1)
        y.append(vec)

    if not model:
        model = neural_network_model(input_size=64,
                                     output_size=len(set(training_actions)))

    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500,
              show_metric=True, run_id='chess_model')
    model.save('chess_model.tflean')
    return model


