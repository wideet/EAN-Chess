import numpy as np
import chess
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from datetime import datetime
import warnings
import pickle
import json
import os
warnings.filterwarnings("ignore")


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
    6: int(1e5)
}

promotion_value_map = {
    'n': 3,
    'b': 3,
    'r': 5,
    'q': 9
}

def get_action_map(num_iter=1000, curr_map_path=None):
    """
    Get all the possible actions from num_iter simulated games
    Update the current map with any new actions found
    """
    if curr_map_path:
        with open(curr_map_path) as fp:
            action_set = set(json.load(fp).values())
    else:
        action_set = set()
    board = chess.Board()
    for i in range(num_iter):
        while not board.is_stalemate() and not board.is_game_over():
            legal_moves = [str(x) for x in board.legal_moves]
            action_set.update(set(legal_moves))
            np.random.shuffle(legal_moves)
            curr_action = chess.Move.from_uci(np.random.choice(legal_moves))
            board.push(curr_action)

    action_map = {ind: action for ind, action in enumerate(sorted(action_set))}
    with open(curr_map_path, 'w') as fp:
        json.dump(action_map, fp)
    return action_map


def get_board_vector(board, flip=False):
    """
    Get a length-64 vector representing the board
    Vector values represent the type of piece at that positon
    Current player pieces are positive int's, other player's pieces are negative
    :param board: chess.Board
    :param flip: int
    :return: np.array (64,)
    """
    vec = np.zeros((64,), dtype="float32")
    curr_player = board.turn
    flip_mult = 1 if not flip else -1
    for pos in range(64):
        piece = board.piece_at(pos)
        if piece:
            if piece.color == curr_player:
                piece_mult = 1
            else:
                piece_mult = -1
            vec[pos] = flip_mult*piece_mult*int(piece.piece_type)
    return vec


def make_move(board, move, piece_value_map, promotion_value_map):
    """Return a new board vector and reward and whether it finishes the game
    Assume the move is already validated and legal"""
    reward = 0
    done = False
    init_pos, new_pos, promotion = move[:2], move[2:4], move[4:5]
    if board.piece_at(chess.SQUARE_NAMES.index(new_pos)):
        reward += piece_value_map[board.piece_at(
            chess.SQUARE_NAMES.index(new_pos)).piece_type]
    if promotion:
        reward += promotion_value_map[promotion]
    board.push(chess.Move.from_uci(move))
    new_state = get_board_vector(board, flip=True)
    if board.is_game_over():
        done = True
    return new_state, reward, done


def neural_network_model(input_size, output_size, learning_rate):
    network = input_data(shape=[None, input_size], name='input')
    network = fully_connected(network, 128, activation='relu')
    network = fully_connected(network, 128, activation='relu')
    network = fully_connected(network, output_size, activation='softmax')
    network = regression(network,
                         optimizer='adam',
                         learning_rate=learning_rate,
                         loss='mean_square',
                         name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')
    return model


def get_training_data(action_map, model, piece_value_map, promotion_value_map,
                      games=100, epsilon=0.7):
    board = chess.Board()
    data = []
    action_map_value_set = set(
                        [chess.Move.from_uci(v) for v in action_map.values()])
    for i in range(games):
        board.reset()
        turns = 0
        while not board.is_stalemate() and not board.is_game_over():
            state = get_board_vector(board)
            if np.random.rand() <= epsilon:
                action = str(np.random.choice(list(board.legal_moves)))
            elif set(board.legal_moves).isdisjoint(action_map_value_set):
                # If there are no legal moves from the action_map, pick first
                # random legal move to continue the game
                print("Had to choose legal move")
                action = str(np.random.choice(list(board.legal_moves)))
            else:
                action = None
                ind = 0
                sorted_action_index_ls = np.flip(
                    np.argsort(model.predict(state.reshape(1, -1)))[0])
                while not action:
                    action = action_map[sorted_action_index_ls[ind]]
                    if chess.Move.from_uci(action) not in board.legal_moves:
                        action = None
                        ind += 1
            new_state, reward, done = make_move(board, action, piece_value_map,
                                                promotion_value_map)
            data.append((state, action, reward, new_state, done))
            turns += 1
            if turns % 200 == 0:
                print(turns)
        print(i)
    return data


# data.append((state, action, reward, new_state, done))
def train_model(num_games, model, action_map, piece_value_map,
                promotion_value_map, epsilon, gamma=0.9):
    board = chess.Board()
    action_map_value_set = set(
        [chess.Move.from_uci(v) for v in action_map.values()])
    reverse_action_map = {v: k for k, v in action_map.items()}
    for game in range(num_games):
        board.reset()
        game_data = []
        turns = 0
        while not board.is_stalemate() and not board.is_game_over():
            state = get_board_vector(board)
            if np.random.rand() <= epsilon:
                action = str(np.random.choice(list(board.legal_moves)))
            elif set(board.legal_moves).isdisjoint(action_map_value_set):
                # If there are no legal moves from the action_map, pick first
                # random legal move to continue the game
                print("Had to choose legal move")
                action = str(np.random.choice(list(board.legal_moves)))
            else:
                action = None
                ind = 0
                sorted_action_index_ls = np.flip(
                    np.argsort(model.predict(state.reshape(1, -1)))[0])
                while not action:
                    action = action_map[sorted_action_index_ls[ind]]
                    if chess.Move.from_uci(action) not in board.legal_moves:
                        action = None
                        ind += 1
            new_state, reward, done = make_move(board, action, piece_value_map,
                                                promotion_value_map)
            game_data.append((state, action, reward, new_state, done))
            turns += 1
            if turns % 200 == 0:
                print("GAME: ", game)
                print("TURN: ", turns)
        print("GAME {} COMPLETE".format(game))
        print("TRAINING ON {} MOVES".format(len(game_data)))

        # TRAIN ON GAME DATA
        X = np.zeros((len(game_data), 64))
        y = np.zeros((len(game_data), len(action_map)))
        for i, data_point in enumerate(game_data):
            state = data_point[0].reshape(1, -1)
            action = data_point[1]
            reward = data_point[2]
            new_state = data_point[3].reshape(1, -1)
            done = data_point[4]
            # If the move that was made is not in the action map, can't train
            # on it, so just ignore it
            try:
                action_index = reverse_action_map[action]
            except:
                continue
            X[i] = state
            y[i] = model.predict(state)
            Q_sa = model.predict(new_state)
            if done:
                y[i, action_index] = reward
            else:
                y[i, action_index] = reward + gamma * np.max(Q_sa)
        model.fit({'input': X}, {'targets': y}, n_epoch=1,
                  show_metric=True, run_id='chess_model')
    return model


games_observed = 50
epsilon = 0.7   # Prob of choosing a random move during simulation
gamma = 0.1     # (0-1) Amount we care about future moves
lr = 0.005

date_str = str(datetime.now()).replace(' ', '_').replace(':', '')
action_map = get_action_map(100000, 'action_map.json')
print(len(action_map))
init_model = neural_network_model(input_size=64,
                                  output_size=len(action_map),
                                  learning_rate=lr)
# training_data = get_training_data(action_map, init_model, piece_value_map,
#                                   promotion_value_map, games=games_observed,
#                                   epsilon=epsilon)
# with open(os.path.join('training_data', date_str + '.pkl'), 'wb') as fp:
#     pickle.dump(training_data, fp)

trained_model = train_model(num_games=games_observed,
                            model=init_model,
                            action_map=action_map,
                            piece_value_map=piece_value_map,
                            promotion_value_map=promotion_value_map,
                            epsilon=epsilon,
                            gamma=gamma)
trained_model.save(
    os.path.join('models', date_str, 'chess_model.tflearn'))


