import random
import time
from os import listdir

import torch

from gamestate import GameState
from memory import Memory
from models.actor_critic_lstm import ActorCriticNetworkLSTM
from players.random_coward_player import RandomCowardPlayer
from players.rl_player import RlPlayer
from players.rule_based_player import RuleBasedPlayer
from rules import Rules
import numpy as np


class SchafkopfGame:
    """
    This class is responsible for managing a round of 4 players to play the game.
    The round is created given the policies of the 4 players.
    The main function is play_one_game.
  """

    def __init__(self, player0, player1, player2, player3, seed=None):
        self.players = [player0, player1, player2, player3]

        for p in self.players:
            if isinstance(p, RlPlayer):
                p.policy.eval()

        self.rules = Rules()
        self.setSeed(seed)

        # some logging counts
        self.game_count = [0, 0, 0, 0]  # weiter, sauspiel, wenz, solo
        self.won_game_count = [0, 0, 0, 0]
        self.contra_retour = [0, 0]

    def play_one_game(self):
        """
    Simulates one game of Schafkopf between 4 players with given policies:
    1) selects a dealer
    2) asks every player to call a game and selects the highest
    3) performs the trick phase where each player plays one card and the highest card gets the trick
    4) determins winners and rewards

    :return: return the final game_state of the game
    :rtype: game_state
    """
        dealer = random.choice([0, 1, 2, 3])
        game_state = GameState(dealer)

        # deal cards
        random.shuffle(self.rules.cards)
        for p in range(4):
            self.players[p].take_cards(self.rules.cards[8 * p:8 * (p + 1)])

        # every player beginning with the one left of the dealer calls his game
        game_state.game_stage = GameState.BIDDING
        current_highest_game = [None, None]
        game_player = None
        for p in range(4):
            current_player_id = (game_state.first_player + p) % 4
            current_player = self.players[current_player_id]
            game_type, prob = current_player.call_game_type(game_state)
            game_state.bidding_round[current_player_id] = game_type
            game_state.action_probabilities[0][current_player_id] = prob
            if current_highest_game[1] == None or (not game_type[1] == None and game_type[1] > current_highest_game[1]):
                current_highest_game = game_type
                game_player = current_player_id
        game_state.game_player = game_player
        game_state.game_type = current_highest_game

        # play the game if someone is playing
        if game_state.game_type != [None, None]:
            # gegenspieler can now double the game
            game_state.game_stage = GameState.CONTRA
            for p in range(4):
                current_player_id = (game_state.first_player + p) % 4
                current_player = self.players[current_player_id]
                contra, prob = current_player.contra_retour(game_state)
                game_state.action_probabilities[1][current_player_id] = prob
                if contra:
                    game_state.contra_retour.append(current_player_id)

            game_state.game_stage = GameState.RETOUR
            for p in range(4):
                current_player_id = (game_state.first_player + p) % 4
                current_player = self.players[current_player_id]
                retour, prob = current_player.contra_retour(game_state)
                game_state.action_probabilities[2][current_player_id] = prob
                if retour:
                    game_state.contra_retour.append(current_player_id)

            # trick phase
            game_state.game_stage = GameState.TRICK
            first_player_of_trick = game_state.first_player
            for trick_number in range(8):
                game_state.trick_number = trick_number
                for p in range(4):
                    current_player_id = (first_player_of_trick + p) % 4
                    current_player = self.players[current_player_id]
                    card, prob = current_player.select_card(game_state)
                    game_state.player_plays_card(current_player_id, card, prob)
                first_player_of_trick = game_state.trick_owner[trick_number]

        # determine winner(s) and rewards
        player_rewards = game_state.get_rewards()
        for p in range(4):
            self.players[p].retrieve_reward(player_rewards[p], game_state)

        # update statistics just for logging purposes
        self.update_statistics(game_state)

        return game_state

    def setSeed(self, seed):
        if seed == None:
            seed = int(time.time() * 1000) % 2**32
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        self.rules = Rules()

    def get_player_memories(self, ids=None):
        """
    summarizes the memories (states, actions, rewards, ...) of selected players
    :param ids: the ids of the player memories
    :type ids: list
    :return: the combined memories of the selected players
    :rtype: memory
    """
        memory = Memory()
        if ids == None:
            ids = range(4)
        for i in ids:
            memory.append_memory(self.players[i].memory)
        return memory

    def print_game(self, game_state):
        br = "Bidding Round: "
        for i in range(4):
            if game_state.first_player == i:
                br += "(" + str(i) + "*)"
            else:
                br += "(" + str(i) + ")"
            if game_state.bidding_round[i][1] != None:
                if game_state.bidding_round[i][0] != None:
                    br += self.rules.card_color[game_state.bidding_round[i][0]] + " "
                br += self.rules.game_names[game_state.bidding_round[i][1]] + " "
            else:
                br += "weiter! "
            br += "[{:0.3f}]  ".format(game_state.action_probabilities[0][i])
        print(br + "\n")

        played_game_str = "Played Game: "
        if game_state.game_type[1] != None:
            if game_state.game_type[0] != None:
                played_game_str += self.rules.card_color[game_state.game_type[0]] + " "
            played_game_str += self.rules.game_names[game_state.game_type[1]] + " "
        else:
            played_game_str += "no game "
        print(played_game_str + "played by player: " + str(game_state.game_player) + "\n")
        contra_str = "Contra/Retour: "
        for i in range(len(game_state.contra_retour)):
            player = game_state.contra_retour[i]
            contra_str += "player "+str(player)
            contra_str += "[{:0.3f}]".format(game_state.action_probabilities[i + 1][player])
            contra_str += "  |   "
        print(contra_str + "\n")

        if game_state.game_type[1] != None:
            print("Course of game")
            for trick in range(8):
                trick_str = ""
                for player in range(4):
                    trick_str_ = "(" + str(player)
                    if (trick == 0 and game_state.first_player == player) or (
                            trick != 0 and game_state.trick_owner[trick - 1] == player):
                        trick_str_ += "^"
                    if game_state.trick_owner[trick] == player:
                        trick_str_ += "*"
                    trick_str_ += ")"

                    if game_state.course_of_game_playerwise[trick][player] in self.rules.get_sorted_trumps(
                            game_state.game_type):
                        trick_str_ += '\033[91m'

                    trick_str_ += self.rules.card_color[game_state.course_of_game_playerwise[trick][player][0]] + " " + \
                                  self.rules.card_number[game_state.course_of_game_playerwise[trick][player][1]]

                    trick_str_ += "[{:0.3f}]".format(game_state.action_probabilities[trick+3][player])
                    if game_state.course_of_game_playerwise[trick][player] in self.rules.get_sorted_trumps(
                            game_state.game_type):
                        trick_str_ += '\033[0m'
                        trick_str += trick_str_.ljust(39)
                    else:
                        trick_str += trick_str_.ljust(30)
                print(trick_str)

            print("\nScores: " + str(game_state.scores) + "\n")
        rewards = game_state.get_rewards()
        print("Rewards: " + str(rewards))

    def update_statistics(self, game_state):
        if game_state.game_type[1] == None:
            self.game_count[0] += 1
        else:
            self.game_count[game_state.game_type[1] + 1] += 1
            if game_state.get_rewards()[game_state.game_player] > 0:
                self.won_game_count[game_state.game_type[1] + 1] += 1
            if len(game_state.contra_retour) >= 1:
                self.contra_retour[0] += 1
            if len(game_state.contra_retour) == 2:
                self.contra_retour[1] += 1


def main():
    all_rewards = np.array([0, 0, 0, 0])

    '''
    policy = ActorCriticNetwork6_ego()
    # take the newest generation available
    # file pattern = policy-000001.pt
    generations = [int(f[:8]) for f in listdir("../policies") if f.endswith(".pt")]
    if len(generations) > 0:
        max_gen = max(generations)
        policy.load_state_dict(torch.load("../policies/" + str(max_gen).zfill(8) + ".pt"))

    # policy.eval()
    policy.to(device='cuda')
    '''
    gs = SchafkopfGame(RlPlayer(0, policy), RlPlayer(1, policy), RlPlayer(2, policy), RlPlayer(3, policy), 1)
    #gs = SchafkopfGame(RandomCowardPlayer(0), RuleBasedPlayer(1), RandomCowardPlayer(2),  RuleBasedPlayer(3), 1)
    #gs = SchafkopfGame(RuleBasedPlayer(0), RuleBasedPlayer(1), RuleBasedPlayer(2), RuleBasedPlayer(3), 1)
    #gs = SchafkopfGame(policy, policy, policy, policy, 1)

    for i in range(10):
        print("playing game " + str(i))
        game_state = gs.play_one_game()
        rewards = np.array(game_state.get_rewards())
        all_rewards += rewards
        gs.print_game(game_state)
    print(all_rewards)
    print(sum(all_rewards))


if __name__ == '__main__':
    main()
