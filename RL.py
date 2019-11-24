

from datetime import datetime
import warnings
import json
import os
warnings.simplefilter(action='ignore', category=FutureWarning)




games_observed = 10000
epsilon = 0.9   # Prob of choosing a random move during simulation
gamma = 0.7     # (0-1) Amount we care about future moves
lr = 0.0005

date_str = str(datetime.now()).replace(' ', '_').replace(':', '')
action_map = get_action_map(100000, 'action_map.json')
print(len(action_map))
init_model = neural_network_model(input_size=64,
                                  output_size=len(action_map),
                                  learning_rate=lr)

trained_model = train_model(num_games=games_observed,
                            model=init_model,
                            action_map=action_map,
                            piece_value_map=piece_value_map,
                            promotion_value_map=promotion_value_map,
                            epsilon=epsilon,
                            gamma=gamma)
os.makedirs(os.path.join('models', date_str))
trained_model.save(
    os.path.join('models', date_str, 'chess_model.tflearn'))
