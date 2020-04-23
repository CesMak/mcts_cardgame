import random
import numpy as np
import math

# Author Markus Lamprecht (www.simact.de) 08.12.2019
# A card game named Witches:

class card(object):
    def __init__(self, color, val):
        self.color = color
        self.value = val

    # Implementing build in methods so that you can print a card object
    def __unicode__(self):
        return self.show()
    def __str__(self):
        return self.show()
    def __repr__(self):
        return self.show()

    def show(self):
        return str("'{} of {}'".format(self.value, self.color))


class deck(object):
    def __init__(self):
        self.cards = []
        self.build()

    # Display all cards in the deck
    def show(self):
        for card in self.cards:
            print(card.show())

    # Generate 60 cards
    # Green Yellow Blue Red
    def build(self):
        self.cards = []
        for color in ['G', 'Y', 'B', 'R']:
            for val in range(1, 16):# choose different deck size here!
                self.cards.append(card(color, val))

    # Shuffle the deck
    def shuffle(self, num=1):
        length = len(self.cards)
        for _ in range(num):
            # This is the fisher yates shuffle algorithm
            for i in range(length-1, 0, -1):
                randi = random.randint(0, i)
                if i == randi:
                    continue
                self.cards[i], self.cards[randi] = self.cards[randi], self.cards[i]
            # You can also use the build in shuffle method
            # random.shuffle(self.cards)

    # Return the top card
    def deal(self):
        return self.cards.pop()


class player(object):
    def __init__(self, name, style=0):
        self.name         = name
        self.hand         = []
        self.offhand      = [] # contains won cards of each round (for 4 players 4 cards!)
        self.total_result = 0  # the total result as noted down in a book!
        self.take_hand    = [] # cards in first phase cards to take!
        self.colorFree    = [0.0, 0.0, 0.0, 0.0] # 1.0 means other know that your are free of this color B G R Y
        #self.player_style = style # 0: play random card, 1: play card with lowest value, 2: ai player

    def sayHello(self):
        print ("Hi! My name is {}".format(self.name))
        return self

    # Draw n number of cards from a deck
    # Returns true in n cards are drawn, false if less then that
    def draw(self, deck, num=1):
        for _ in range(num):
            card = deck.deal()
            if card:
                self.hand.append(card)
            else:
                return False
        return True

    # Display all the cards in the players hand
    def showHand(self):
        print ("{}'s hand: {}".format(self.name, self.getHandCardsSorted()))
        return self

    def discard(self):
        # returns most upper card and removes it from the hand!
        return self.hand.pop()

    def getHandCardsSorted(self):
        return sorted(self.hand, key = lambda x: ( x.color,  x.value))

    def getRandomOption(self, incolor):
        options_list = [0]*60
        options      = self.getOptions(incolor)
        rand_card    = random.randrange(len(options))
        [_, card]    = options[rand_card]
        options_list[self.getCardIndex(card)] = 1
        return options_list

    def convertAllCardState(self, state60_in):
        #in comes a state matrix with len = 60 with 0...1..0...1
        indices = [i for i, x in enumerate(state60_in) if int(x) == 1]
        result  = []
        for j in indices:
            result.append(self.getIndexOfCard(j))
        return result

    def getBinaryOptions(self, incolor, shifting_phase=False):
        #returns 0....1... 60x1 array BGRY 0...15 sorted
        options_list = [0]*60
        if shifting_phase:
            options =self.hand
        else:
            options      = self.getOptions(incolor)
        #print("Binary Options:")
        for opt in options:
            # i is the index of the card in the hand!
            if shifting_phase:
                card = opt
            else:
                [i, card] = opt
            #print(card)
            options_list[self.getCardIndex(card)] = 1
        return options_list

    def getBinaryHand(self, input_cards):
        #return hand state as 0..1...0..1 sorted BGRY 1....15
        hand_list = [0]*60
        for card in input_cards:
            hand_list[self.getCardIndex(card)] = 1
        return hand_list

    def getIndexOfCard(self, cardindex):
        my_card = card("G", 1)
        if cardindex<15:
            my_card.color ="B"
            my_card.value = cardindex-15*0+1
        elif cardindex<30 and cardindex>=15:
            my_card.color ="G"
            my_card.value = cardindex-15*1+1
        elif cardindex<45 and cardindex>=30:
            my_card.color ="R"
            my_card.value = cardindex-15*2+1
        elif cardindex<60 and cardindex>=45:
            my_card.color ="Y"
            my_card.value = cardindex-15*3+1
        return my_card

    def playBinaryCardIndex(self, cardindex):
        #input cardindex 0...1....60
        #play correct card from hand (if possible)
        cardtoplay = self.getIndexOfCard(cardindex)
        idx_of_card =  self.specificIndexHand(cardtoplay)
        return self.hand.pop(idx_of_card)

    def playRandomCard(self, incolor):
        options = (self.getOptions(incolor))
        if len(options) == 0:
            print("Error has no options left!", options, self.hand)
            return None
        rand_card = random.randrange(len(options))
        card_idx = 0
        card_idx  = options[rand_card][0]
        return self.hand.pop(card_idx)

    def getCardIndex(self, card):
        #return sorted index of a card BGRY
        result_idx = 0
        if card.color == "B":
            result_idx = card.value-1
        elif card.color =="G":
            result_idx =15+card.value-1
        elif card.color =="R":
            result_idx = 15*2+card.value-1
        elif card.color =="Y":
            result_idx = 15*3+card.value-1
        return result_idx

    def setColorFree(self, color):
        if color =="B":
            self.colorFree[0] = 1.0
        elif color =="G":
            self.colorFree[1] = 1.0
        elif color == "R":
            self.colorFree[2] = 1.0
        elif color =="Y":
            self.colorFree[3]  = 1.0

    def getOptions(self, incolor, orderOptions=0):
        # incolor = None -> Narr was played played before
        # incolor = None -> You can start!
        #    In both cases return all options as cards
        options = []
        hasColor = False
        if incolor is None:
            for i, card in enumerate(self.hand):
                options.append([i, card])
        else:
            for i, card in enumerate(self.hand):
                if card.color == incolor and card.value <15:
                    options.append([i, card])
                    hasColor = True
                if card.value == 15: # append all joker
                    options.append([i, card])

        # if has not color and no joker append all cards!
        # wenn man also eine Farbe aus ist!
        if not hasColor:
            options = [] # necessary otherwise joker double!
            for i, card in enumerate(self.hand):
                options.append([i, card])
            if not self.hasJoker() and incolor is not None:
                self.setColorFree(incolor)
        if orderOptions: return sorted(options, key = lambda x: ( x[1].color,  x[1].value))
        return options

    def getMinimumValue(self, options):
        # return card index with minimum value.
        minimum_value = options[0][1].value
        index          = options[0][0]
        for opt in options:
            [i, card] = opt
            if card.value<minimum_value:
                minimum_value=card.value
                index        = i
        return index

    def hasJoker(self):
        for i in ["Y", "R", "G", "B"]:
            if self.hasSpecificCard(14, i):
                return True
        return False

    def hasYellowEleven(self):
        return self.hasSpecificCard(11, "Y")

    def hasRedEleven(self):
        return self.hasSpecificCard(11, "R")

    def hasBlueEleven(self):
        return self.hasSpecificCard(11, "B")

    def specificIndexHand(self, incard):
        # return card index of the incard of the hand
            for i, card in enumerate(self.hand):
                if card.color == incard.color and card.value == incard.value:
                    return i
            return 0

    def hasSpecificCardOnHand(self, card):
        # return True if the hand has this card!
        for i in self.hand:
            if card.color == i.color and card.value == i.value:
                return True
        return False

    def hasSpecificCard(self, cardValue, cardColor):
        # return True if the offhand has this card!
        for stich in self.offhand:
            for card in stich:
                if card is not None:
                    if card.color == cardColor and card.value == cardValue:
                        return True
        return False

    def countResult(self, input_cards):
        #input_cards = [[card1, card2, card3, card4], [stich2], ...]
        # in class player
        # get the current Reward (Evaluate offhand cards!)
        negative_result = 0
        # input_cards = self.offhand
        for stich in input_cards:
            for card in stich:
                if card is not None:
                    if card.color == "R" and card.value <15 and card.value!=11 and card.value and not self.hasRedEleven():
                        negative_result -=1
                    if card.color == "R" and card.value <15 and card.value!=11 and card.value and self.hasRedEleven():
                        negative_result -=1*2
                    if not self.hasBlueEleven():
                        if card.color == "G" and card.value == 11:
                            negative_result -= 5
                        if card.color == "G" and card.value == 12:
                            negative_result -= 10
                    if card.color == "Y" and card.value == 11:
                        negative_result+=5
        return negative_result

    def appendCards(self, stich):
        # add cards to the offhand.
        self.offhand.append(stich)

class game(object):
    def __init__(self, options_dict):
        self.names_player      = options_dict["names"]
        self.nu_players        = len(self.names_player)
        self.current_round     = 0
        self.nu_games_played   = 0
        self.players           = []  # stores players object
        self.on_table_cards    = []  # stores card on the table
        self.active_player     = 0   # due to gym reset =3 stores which player is active (has to give a card)
        self.played_cards      = []  # of one game # see also in players offhand!
        self.gameOver          = 0
        self.neuralNetworkInputs = {}
        self.game_start_player = self.active_player
        self.ai_player         = options_dict["type"]
        self.rewards           = np.zeros((self.nu_players,))
        self.total_rewards     = np.zeros((self.nu_players,))
        #Shifting:
        self.shifted_cards     = 0 # counts
        self.nu_shift_cards    = 2 # shift 2 cards!
        self.shifting_phase    = True
        self.shift_option      = 0 # due to gym reset=2 ["left", "right", "opposide"]
        self.correct_moves     = -1

        myDeck = deck()
        myDeck.shuffle()
        self.total_rounds      = int(len(myDeck.cards)/self.nu_players)
        self.setup_game(myDeck)

    # generate a Game:
    def setup_game(self, myDeck):
        for i in range (self.nu_players):
            play = player(self.names_player[i])
            play.draw(myDeck, self.total_rounds)
            play.hand = play.getHandCardsSorted()
            self.players.append(play)

        # print("Show HANDDD")
        # for p in self.players:
        #     p.showHand()

    def reset_game(self):
        #not used yet.
        myDeck = deck()
        myDeck.shuffle()
        self.nu_games_played +=1
        self.shifted_cards  = 0

        if self.shift_option <2:
            self.shift_option += 1
        else:
            self.shift_option  = 0
        self.shifting_phase    = True
        self.players           = []  # stores players object
        self.on_table_cards    = []  # stores card on the table
        self.played_cards      = []  # of one game # see also in players offhand!
        self.gameOver          = 0
        self.rewards           = np.zeros((self.nu_players,))
        self.current_round     = 0
        self.setup_game(myDeck)
        self.active_player     = self.nextGamePlayer()
        self.correct_moves     = -1

    def init_Random_TestGame(self, ai_player=["RANDOM", "RL", "RANDOM", "RANDOM"]):
        self.ai_player         = ai_player

    def play_ai_move(self, ai_card_idx, print_=False):
        'card idx from 0...60'
        current_player    =  self.active_player
        valid_options_idx = self.getValidOptions(current_player)
        card              = self.players[current_player].getIndexOfCard(ai_card_idx)
        player_has_card   = self.players[current_player].hasSpecificCardOnHand(card)
        tmp               = self.players[current_player].specificIndexHand(card)
        if player_has_card and tmp in valid_options_idx:
            if print_:
                if self.shifting_phase:
                    print("[{}] {} {}\t shifts {}\tCard {}\tCard Index {}\t len {}".format(self.current_round, current_player, self.names_player[current_player], self.ai_player[current_player], card, ai_card_idx, len(self.players[current_player].hand)))
                else:
                    print("[{}] {} {}\t plays {}\tCard {}\tCard Index {}\t len {}".format(self.current_round, current_player, self.names_player[current_player], self.ai_player[current_player], card, ai_card_idx, len(self.players[current_player].hand)))
            self.correct_moves +=1
            rewards, round_finished, gameOver = self.step_idx_with_shift(tmp)
            if print_ and round_finished:
                print(rewards, self.correct_moves, gameOver, "\n")
            return rewards, round_finished, gameOver
        else:
            return {"state": "play_or_shift", "ai_reward": None}, False, True # rewards round_finished, game_over

    def playUntilAI(self, print_=False):
        rewards        = {"state": "play_or_shift", "ai_reward": None}
        gameOver       = False
        round_finished = False
        while len(self.players[self.active_player].hand) > 0:
            current_player = self.active_player
            if "RANDOM" in self.ai_player[current_player]:
                if  self.shifting_phase:
                    action = self.getRandomCards()[0] # is a hand index
                    if print_:
                        card              = self.players[current_player].getIndexOfCard(action)
                        print("[{}] {} {}\t shifts {}\tCard {}\tHand Index {}\t len {}".format(self.current_round, current_player, self.names_player[current_player], self.ai_player[current_player], self.players[current_player].hand[action], action, len(self.players[current_player].hand)))
                else:
                    action = self.getRandomOption_()
                    if print_:
                        card              = self.players[current_player].getIndexOfCard(action)
                        print("[{}] {} {}\t plays {}\tCard {}\tHand Index {}\t len {}".format(self.current_round, current_player, self.names_player[current_player], self.ai_player[current_player], self.players[current_player].hand[action], action, len(self.players[current_player].hand)))
                rewards, round_finished, gameOver = self.step_idx_with_shift(action)
                if print_ and round_finished:
                    print(rewards, self.correct_moves, gameOver, "\n")
            else:
                return rewards, round_finished, gameOver
        # Game is over!
        #CAUTION IF GAME OVER NO REWARDS ARE RETURNED
        #rewards = {'state': 'play_or_shift', 'ai_reward': None}
        return rewards, True, True

    def stepRandomPlay(self, action_ai, print_=False):
        rewards, round_finished, gameOver = self.play_ai_move(action_ai, print_=print_)
        if rewards["ai_reward"] is None: # illegal move
            corr_moves = self.correct_moves
            return None, corr_moves, True
        else:
            rewards, round_finished, gameOver = self.playUntilAI(print_=print_)
            ai_reward = 0
            if rewards["state"]=="play":
                ai_reward = rewards["trick_rewards"][1]
            return ai_reward, self.correct_moves, gameOver

    def getInColor(self):
        # returns the leading color of the on_table_cards
        # if only joker are played None is returned
        for i, card in enumerate(self.on_table_cards):
            if card is not None:
                if card.value <15:
                    return card.color
        return None

    def evaluateWinner(self):
        #uses on_table_cards to evaluate the winner of one round
        #returns winning card
        #player_win_idx: player that one this game! (0-3)
        #on_table_win_idx: player in sequence that one!
        highest_value    = 0
        winning_card     = self.on_table_cards[0]
        incolor          = self.getInColor()
        on_table_win_idx = 0
        if  incolor is not None:
            for i, card in enumerate(self.on_table_cards):
                # Note 15 is a Jocker
                if card is not None and ( card.value > highest_value and card.color == incolor and card.value<15):
                    highest_value = card.value
                    winning_card = card
                    on_table_win_idx = i

        player_win_idx = self.active_player
        for i in range(self.nu_players-on_table_win_idx-1):
            player_win_idx = self.getPreviousPlayer(player_win_idx)
        return winning_card, on_table_win_idx, player_win_idx

    def nextGamePlayer(self):
        if self.game_start_player < self.nu_players-1:
            self.game_start_player+=1
        else:
            self.game_start_player = 0
        return self.game_start_player

    def getPreviousPlayer(self, input_number):
        if input_number == 0:
            prev_player = self.nu_players-1
        else:
            prev_player = input_number -1
        return prev_player

    def getShiftPlayer(self):
        # works FOR 4 Players only!
        if self.shift_option==0:
            return self.getNextPlayer_()
        elif self.shift_option==1:
            return self.getPreviousPlayer(self.active_player)
        elif self.shift_option==2: # opposide
            return self.getPreviousPlayer(self.getPreviousPlayer(self.active_player))
        else:
            print("ERROR!!!! TO BE IMPLEMENTED!")
            return 0

    def getNextPlayer_(self):
        tmp = self.active_player
        if tmp < self.nu_players-1:
            tmp+=1
        else:
            tmp = 0
        return tmp

    def getNextPlayer(self):
        if self.active_player < self.nu_players-1:
            self.active_player+=1
        else:
            self.active_player = 0
        return self.active_player

    def getRandomOption_(self):
        incolor = None
        if len(self.on_table_cards)>0:
            incolor = self.on_table_cards[0].color
        options = self.players[self.active_player].getOptions(incolor)
        if len(options) == 0:
            print("Error has no options left!", options, self.players[self.active_player].hand)
            return None
        rand_card = random.randrange(len(options))
        card_idx = 0
        card_idx  = options[rand_card][0]
        return card_idx

    def getCardIndex(self, card):
        result_idx = 0
        if card.color == "B":
            result_idx = card.value-1
        elif card.color =="G":
            result_idx =15+card.value-1
        elif card.color =="R":
            result_idx = 15*2+card.value-1
        elif card.color =="Y":
            result_idx = 15*3+card.value-1
        return result_idx

    def convertAvailableActions(self, availAcs):
        #convert from (1,0,0,1,1...) to (0, -math.inf, -math.inf, 0,0...) etc
        for j, i in enumerate(availAcs):
            if i == 1:
                availAcs[j]=0
            if i == 0:
                availAcs[j]=-math.inf
        return np.asarray(availAcs)

    def getState_240(self):
        #    return self.playersGo, self.neuralNetworkInputs[self.playersGo].reshape(1,412), convertAvailableActions(self.returnAvailableActions()).reshape(1,1695)
        # return active_player, neuronNetworkInputs of active player and available actions of active player
        play_options = self.players[self.active_player].getBinaryOptions(self.getInColor(), self.shifting_phase)
        on_table, on_hand, played = self.getmyState(self.active_player)
        return np.asarray([on_table, on_hand, played, play_options])

    def getState_303(self):
        play_options = self.players[self.active_player].getBinaryOptions(self.getInColor(), self.shifting_phase)
        on_table, on_hand, played = self.getmyState(self.active_player)
        add_states = [] #48= with has cards,+12=60 for does not have card anymore...
        for i in range(len(self.players)):
            if i!=self.active_player:
                add_states.extend(self.getAdditionalState(i))
        return np.asarray([on_table+ on_hand+ played+ play_options+ add_states])

    def getState(self):
        #    return self.playersGo, self.neuralNetworkInputs[self.playersGo].reshape(1,412), convertAvailableActions(self.returnAvailableActions()).reshape(1,1695)
        # return active_player, neuronNetworkInputs of active player and available actions of active player
        play_options = self.players[self.active_player].getBinaryOptions(self.getInColor(), self.shifting_phase)
        #play_options = self.convertAvailableActions(play_options)
        on_table, on_hand, played = self.getmyState(self.active_player)
        add_states = [] #48= with has cards,+12=60 for does not have card anymore...
        for i in range(len(self.players)):
            if i!=self.active_player:
                add_states.extend(self.getAdditionalState(i))
        # if self.shifting_phase:
        #     for i in range(len(on_table)):
        #         on_table[i]    = 1.00
        return np.asarray([on_table+ on_hand+ played+ play_options+ add_states])

    def getBinaryStateFirstCard(self, playeridx, action):
        hand   = self.players[playeridx].getBinaryHand(self.players[playeridx].hand)
        actions = [self.players[playeridx].hand[action[0]], self.players[playeridx].hand[action[1]]]
        action_output = self.players[playeridx].getBinaryHand(actions)
        return [hand]+[action_output]

    def getBinaryState(self, playeridx, action_output, bestq):
        bin_options   = self.players[playeridx].getBinaryOptions(self.getInColor())
        bin_on_table  = self.players[playeridx].getBinaryHand(self.on_table_cards)
        bin_played    = self.players[playeridx].getBinaryHand(self.played_cards)
        action_output = self.players[playeridx].getBinaryHand([self.players[playeridx].hand[action_output]])
        return [bin_options+bin_on_table+bin_played]+[action_output+[round(bestq, 2)]]

    def getPlayerState(self, playeridx):
        return [self.players[playeridx].hand, self.on_table_cards, self.played_cards]

    def getGameState(self):
        return [self.players, self.rewards, self.on_table_cards, self.played_cards, self.shifted_cards, self.total_rewards]

    def setState(self, game_state):
        [players, rewards, on_table_cards, played_cards, shifted_cards, total_rewards, active_player] = game_state
        self.active_player  = active_player
        self.on_table_cards = on_table_cards
        self.players        = players
        self.played_cards   = played_cards
        self.rewards        = rewards
        self.shifted_cards  = shifted_cards
        self.total_rewards  = total_rewards

    def isGameFinished(self):
        cards = 0
        for player in self.players:
            cards += len(player.hand)
        if cards == 0:
            return True
        else:
            return False

    def getShiftOptions(self):
        # Return all options to shift 2 not unique card idx.
        # returns:  [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14], [1, 2], [1
        n   = len(self.players[self.active_player].hand)
        i   = 0
        options = []
        for j in range(0, n-1):
            tmp = i
            while tmp<n-1:
                options.append([j, tmp+1])
                tmp +=1
            i = i+1
        return options

    def getValidOptions(self, player):
        if self.shifting_phase:
            options = [x for x in range(len(self.players[player].hand))]
            return options
        else:
            options =  self.players[player].getOptions(self.getInColor())
        action_idx_list = []
        for option in options:
            action_idx, card = option
            action_idx_list.append(action_idx)
        return action_idx_list

    def step_idx_with_shift(self, card_idx):
        #Note that card_idx is a Hand Card IDX!
        self.shifting_phase = (self.shifted_cards<=self.nu_players*self.nu_shift_cards)
        if self.shifting_phase:
            shift_round   = int(self.shifted_cards/self.nu_players)
            self.shiftCard(card_idx, self.active_player, self.getShiftPlayer())
            self.shifted_cards +=1

            round_finished = False
            if self.shifted_cards%self.nu_players == 0:
                round_finished = True
            #print("Shift Round:", shift_round, "Shifted Cards:", self.shifted_cards, "round_finished", round_finished)
            if shift_round == (self.nu_shift_cards)-1 and round_finished:
                #print("\nShifting PHASE FINISHED!!!!!!\n")
                for player in self.players:
                    #print(player.name, player.take_hand)
                    player.hand.extend(player.take_hand)
                self.shifted_cards  = 100
                self.shifting_phase = False
            self.active_player = self.getNextPlayer()
            return {"state": "shift", "ai_reward": 0}, round_finished, False # rewards, round_finished, gameOver
        else:
            return self.step_idx(card_idx)

    def step_idx(self, card_idx):
        """
        card_idx = hand card!
        """
        # in case card_idx is a simple int value
        round_finished = False
        # play the card_idx:
        played_card = self.players[self.active_player].hand.pop(card_idx)
        self.on_table_cards.append(played_card)
        self.played_cards.append(played_card)
        # Case round finished:
        trick_rewards    = [0, 0, 0, 0]
        on_table_win_idx = -1
        player_win_idx   = -1
        if len(self.on_table_cards) == self.nu_players:
            winning_card, on_table_win_idx, player_win_idx = self.evaluateWinner()
            trick_rewards[player_win_idx] = self.players[player_win_idx].countResult([self.on_table_cards])
            self.current_round +=1
            self.players[player_win_idx].appendCards(self.on_table_cards)
            self.on_table_cards = []
            self.active_player  = player_win_idx
            round_finished = True
        else:
            self.active_player = self.getNextPlayer()

        #yes this is the correct ai reward in case all players are ai players.
        return {"state": "play", "ai_reward": trick_rewards[player_win_idx], "on_table_win_idx": on_table_win_idx, "trick_rewards": trick_rewards, "player_win_idx": player_win_idx}, round_finished, self.isGameFinished()

    def step_idx_old(self, card_idx):
      """
      card_idx = hand card!
      """
      # in case card_idx is a simple int value
      round_finished = False
      # play the card_idx:
      played_card = self.players[self.active_player].hand.pop(card_idx)
      self.on_table_cards.append(played_card)
      self.played_cards.append(played_card)
      # Case round finished:
      trick_rewards = [0, 0, 0, 0]
      if len(self.on_table_cards) == self.nu_players:
        winning_card, on_table_win_idx, player_win_idx = self.evaluateWinner()
        trick_rewards[player_win_idx] = self.players[player_win_idx].countResult([self.on_table_cards])
        self.current_round +=1
        self.players[player_win_idx].appendCards(self.on_table_cards)
        self.on_table_cards = []
        self.active_player  = player_win_idx
        round_finished = True
      else:
        self.active_player = self.getNextPlayer()

      finished = self.isGameFinished()
      if finished is not None:
        self.assignRewards()
        self.total_rewards += self.rewards
        # CAUTION CHANGED: before:return self.rewards, True
        return trick_rewards, True
      else:
        # trick value is not 100% the local value as 11 red etc. have future effects!
        return trick_rewards, round_finished

    def shiftCard(self, card_idx, current_player, next_player):
        # shift round = 0, 1, ... (for 2 shifted cards)
        #print("I shift now hand idx", card_idx, "from", self.players[current_player].name, "to", self.players[next_player].name)
        card = self.players[current_player].hand.pop(card_idx) # wenn eine Karte weniger index veringern!
        self.players[next_player].take_hand.append(card)

    def getRandomCards(self):
        # get random indices of cards! (not the same!)
        shift_cards = []
        while len(shift_cards)< self.nu_shift_cards:
            shift_cards.append(random.randrange(len(self.players[self.active_player].hand)))
            shift_cards = list(dict.fromkeys(shift_cards))
        return shift_cards

    def randomStep(self):
        'Out: None    -> game not finished'
        '     rewards -> game finished'
        'No shifting phase allowed here - happens in first call of step_idx'
        incolor = None
        # play the card:
        if len(self.on_table_cards)>0 and self.on_table_cards[0] is not None:
            incolor = self.on_table_cards[0].color
        played_card = self.players[self.active_player].playRandomCard(incolor)
        self.on_table_cards.append(played_card)
        self.played_cards.append(played_card)

        # Case round finished:
        if len(self.on_table_cards) == self.nu_players:
            winning_card, on_table_win_idx, player_win_idx = self.evaluateWinner()
            self.players[player_win_idx].appendCards(self.on_table_cards)
            self.on_table_cards = []
            self.active_player  = player_win_idx
        else:
            self.active_player = self.getNextPlayer()

        finished = self.isGameFinished()
        if finished is not None:
            self.assignRewards()
            return self.rewards
        else:
            return None

    def assignRewards(self):
        for i, player in enumerate(self.players):
            #print(i, player.offhand)
            self.rewards[i] = player.countResult(player.offhand)

    def getAdditionalState(self, playeridx):
        result = []
        player = self.players[playeridx]
        for i in ["B","G","R","Y"]:
            for j in range(11,15):
                if player.hasSpecificCard(j, i):
                    result.append(1.0)
                else:
                    result.append(0.0)

        #extend if this player would win the current cards
        player_win_idx = playeridx
        if len(self.on_table_cards)>0:
            winning_card, on_table_win_idx, player_win_idx = self.evaluateWinner()
        if player_win_idx == playeridx:
            result.extend([1.0])
        else:
            result.extend([0.0])
        result.extend(player.colorFree) # 4 per player
        return result


    def getmyState(self, playeridx):
        on_table =[0.00]*60
        on_hand  =[0.00]*60
        played   =[0.00]*60
        for card in self.on_table_cards:
            on_table[self.getCardIndex(card)]    = 1.00

        for card in self.players[playeridx].hand:
            on_hand[self.getCardIndex(card)] =1.00

        for card in self.played_cards:
            played[self.getCardIndex(card)] = 1.00
        return on_table, on_hand, played

    def getCurrentPlayerState(self, playeridx):
        # get Current State as bool (binary) values
        # 15*4 card x is currently on the table                see self.cards_of_round
        # 15*4 card x is currently in ai hand                see self.player.hand
        # 15*4 card x is already in offhand (being played)    see self.played_cards
        # sort by: (alphabet:) Blue, Green, Red, Yellow
        # 180 Input vector!
        state = [0]*180
        for card in self.on_table_cards:
            state[self.getCardIndex(card)]    = 1

        for card in self.players[playeridx].hand:
            state[self.getCardIndex(card)+60] = 1

        for card in self.played_cards:
            state[self.getCardIndex(card)+120] = 1
        return state

    def step(self, action_list):
        #active player plays a card
        #inputs: player_idx index of the player to play  a card
        #action_idx BGRY 0...1....0...1 of the card to be played!
        #returns: reward, done, info
        if type(action_list) is list:
            action_idx = action_list.index(1)
        else:
            action_idx = action_list

        self.rewards = np.zeros((self.nu_players,))
        info = {}

        #0. check if gameOver
        if self.gameOver:
            print(">>>Is already game over! Reset Game now")
            info = self.reset_game()
            #return self.rewards, self.gameOver, info

        else:
            #1. just play the card:
            played_card = self.players[self.active_player].playBinaryCardIndex(action_idx)
            #print(self.names_player[self.active_player],"plays random option: ", played_card,"action idx:", action_idx, "player:", self.active_player)
            self.played_cards.append(played_card)
            self.on_table_cards.append(played_card)

            #2. Check if round is finished
            if len(self.on_table_cards) == self.nu_players:
                #evaluate winner
                winning_card, on_table_win_idx, player_win_idx = self.evaluateWinner()
                self.current_round +=1
                self.gameOver   =  ((self.total_rounds)==self.current_round)
                self.players[player_win_idx].appendCards(self.on_table_cards)
                # assign rewards
                self.rewards[player_win_idx]  =  self.players[player_win_idx].countResult([self.on_table_cards])

                print(">>>"+ str(winning_card)+", "+str(self.names_player[player_win_idx])+" ("+str(player_win_idx)+") wins this round ["+str(self.current_round)+"/"+str(self.total_rounds)+"] rewards:", self.rewards, "\n")
                self.on_table_cards = []
                self.active_player  = player_win_idx
                #if this player wins! do not call next player cause this player has to start next Round!
            else:
                #3. set next player:
                self.active_player = self.getNextPlayer()

        # fill neuronal network inputs:
        for i in range(self.nu_players):
            self.neuralNetworkInputs[i] = np.asarray(self.getCurrentPlayerState(i), dtype=int)

        return self.rewards, self.gameOver, {} #<-return info here!

    def play_rounds(self, nu_rounds=3):
        #plays multiple rounds in steps!
        for i in range(0, self.total_rounds*self.nu_players*nu_rounds+1*nu_rounds):
            if not self.gameOver:
                rand_option       = self.players[self.active_player].getRandomOption(self.getInColor())
            rewards, gameOver, _ = self.step(rand_option)

    def play_one_game(self, start_player=0):
        j = 0
        for i in range(0, self.total_rounds*self.nu_players):
            #print("\nRound", i, "Player to start:", j)
            j = self.play_round(player_to_start=j)

        # print("\nGame finished with:")
        # for player in self.players:
        #     print(player.name, "Hand:", player.hand, "Result:", player.countResult(), "\noffHand:", player.offhand, "\n\n")

        return self.getNextPlayer()

    def play_multiple_games(self, nu_games):
        curr_player = 0
        for i in range(0, nu_games):
            print("\nGame", i, "Player to start:", curr_player)
            curr_player = self.play_one_game(start_player=curr_player)
            self.reset_game()

        for player in self.players:
            print(player.name, "\tTotal Result:", player.total_result)



####################################
#
#Testing
#
####################################

# game = game(["Tim", "Bob", "Lena", "Anja"])
# game.play_rounds(nu_rounds=3)
