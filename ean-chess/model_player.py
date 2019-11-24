"""Class that represents a player backed by a model"""

from .utils import *


class ModelPlayer:
    def __int__(self, color, model=None, score=0):
        self.color = color
        self.model = model
        self.score = score

    def reset_score(self):
        self.score = 0

    def get_model(self, layers=None, lr=None):
        if not self.model:
            action_map = load_action_map()
            self.model = neural_network_model(input_size=64,
                                              output_size=len(action_map),
                                              layers=layers,
                                              learning_rate=lr)
        return self.model

    def train(self, num_games, epsilon, gamma):
        board = chess.Board()
        action_map = load_action_map()
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
                    # If there are no legal moves from the action_map,
                    # pick first random legal move to continue the game
                    print("Had to choose random legal move")
                    action = str(np.random.choice(list(board.legal_moves)))
                else:
                    action = None
                    ind = 0
                    # list of indices sorted by model's recommendation
                    sorted_action_index_ls = np.flip(np.argsort(
                        self.model.predict(state.reshape(1, -1)))[0])
                    while not action:
                        action = action_map[sorted_action_index_ls[ind]]
                        if chess.Move.from_uci(action) not in \
                                board.legal_moves:
                            action = None
                            ind += 1
                new_state, reward, done = make_move(board, action)
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
                y[i] = self.model.predict(state)
                Q_sa = self.model.predict(new_state)
                if done:
                    y[i, action_index] = reward
                else:
                    y[i, action_index] = reward + gamma * np.max(Q_sa)
            self.model.fit({'input': X}, {'targets': y}, n_epoch=1,
                           show_metric=True, run_id='chess_model')
