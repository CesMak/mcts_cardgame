import random
import numpy as np
import math

# Author Markus Lamprecht (www.simact.de) 08.12.2019
# A card game named Witches:

class card(object):
    def __init__(self, color, val, idx):
        self.color = color
        self.value = val
        self.idx   = idx # unique card index used for getState
        self.player= ""  # this card is owned by this player!

    # Implementing build in methods so that you can print a card object
    def __unicode__(self):
        return self.show()
    def __str__(self):
        return self.show()
    def __repr__(self):
        return self.show()

    def show(self):
        if self.value == 15:
            val = "J"
        elif self.value == 11:
            val =">11<"
        elif self.value == 12 and self.color =="Green":
            val ="°12°"
        else:
            val = self.value
        return str("{} of {}_{}".format(val, self.color, self.idx))


class deck(object):
    def __init__(self, nu_cards, seed=None):
        self.cards    = []
        self.nu_cards = nu_cards # e.g. 4 or maximum is 16?
        #todo assert max cards here
        if seed is not None:
            random.seed(seed)
        self.build()

    # Display all cards in the deck
    def show(self):
        for card in self.cards:
            print(card.show())

    # Green Yellow Blue Red
    def build(self):
        self.cards = []
        idx        = 0
        for color in ['B', 'G', 'R', 'Y']:
            # CHANGED range(0, ...) is WRONG!!!!
            for val in range(1, self.nu_cards+1):# choose different deck size here! max is 16
                self.cards.append(card(color, val, idx))
                idx +=1

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

    def sayHello(self):
        print ("Hi! My name is {}".format(self.name))
        return self

    # Draw n number of cards from a deck
    # Returns true in n cards are drawn, false if less then that
    def draw(self, deck, num=1):
        for _ in range(num):
            card = deck.deal()
            if card:
                card.player = self.name
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

    def getBinaryOptions(self, incolor, players, cards, shifting):
        #returns 0....1... x1 array BGRY 0...15 sorted
        options_list = [0]*players*cards
        if shifting:
            unique_idx = self.cards2Idx(self.hand)
        else:
            unique_idx = self.hand2Idx(self.getOptions(incolor))
        for idx in unique_idx:
            options_list[idx] = 1
        return options_list

    def playRandomCard(self, incolor):
        options = (self.getOptions(incolor))
        if len(options) == 0:
            print("Error has no options left!", options, self.hand)
            return None
        rand_card = random.randrange(len(options))
        card_idx = 0
        card_idx  = options[rand_card][0]
        return self.hand.pop(card_idx)

    def setColorFree(self, color):
        if color =="B":
            self.colorFree[0] = 1.0
        elif color =="G":
            self.colorFree[1] = 1.0
        elif color == "R":
            self.colorFree[2] = 1.0
        elif color =="Y":
            self.colorFree[3]  = 1.0

    def getOptions(self, incolor, orderOptions=False):
        # incolor = None -> Narr was played played before
        # incolor = None -> You can start!
        # Return Hand index
        options = []
        hasColor = False
        if incolor is None:
            for i, card in enumerate(self.hand):
                options.append(i)
        else:
            for i, card in enumerate(self.hand):
                if card.color == incolor and card.value <15:
                    options.append(i)
                    hasColor = True
                if card.value == 15: # append all joker
                    options.append(i)

        # if has not color and no joker append all cards!
        # wenn man also eine Farbe aus ist!
        if not hasColor:
            options = [] # necessary otherwise joker double!
            for i, card in enumerate(self.hand):
                options.append(i)
            if not self.hasJoker() and incolor is not None:
                self.setColorFree(incolor)
        if orderOptions: return sorted(options, key = lambda x: ( x[1].color,  x[1].value))
        return options

    def cards2Idx(self, cardlist):
        result = []
        for i in cardlist:
            result.append(i.idx)
        return result

    def hand2Idx(self, hand_idx):
        #convert hand index to unique card index
        result = []
        for i in hand_idx:
            result.append(self.hand[i].idx)
        return result

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


    def hasSpecificCardOnHand(self, idx):
        # return True if the hand has this card!
        for i in self.hand:
            if i.idx == idx:
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
                    if card.color == "R" and card.value <15 and card.value!=11 and not self.hasRedEleven():
                        negative_result -=1
                    if card.color == "R" and card.value <15 and card.value!=11 and self.hasRedEleven():
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
        self.active_player     =  3  # due to gym reset =3 stores which player is active (has to give a card)
        self.played_cards      = []  # of one game # see also in players offhand!
        self.gameOver          = 0
        self.game_start_player = self.active_player
        self.player_type       = options_dict["type"] # set here the player type RANDOM or RL (reinforcement player)
        self.rewards           = np.zeros((self.nu_players,))
        self.total_rewards     = np.zeros((self.nu_players,))
        #Shifting:
        self.shifted_cards     = 0 # counts
        self.nu_shift_cards    = options_dict["nu_shift_cards"] # shift 2 cards!  # set to 0 to disable
        self.shifting_phase    = True
        self.shift_option      = 2 # due to gym reset=2 ["left", "right", "opposide"]
        self.correct_moves     = 0
        #Number of cards
        self.nu_cards          = options_dict["nu_cards"] # e.g. 15(maximum), or 4 -> 12,13,14,15 of each color is given.
        self.seed              = options_dict["seed"] # none for not using it

        myDeck = deck(self.nu_cards, self.seed)
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

    def reset(self):
        myDeck = deck(self.nu_cards, self.seed)
        myDeck.shuffle()
        self.nu_games_played +=1
        self.shifted_cards  = 0

        if self.shift_option <2:
            self.shift_option += 1
        else:
            self.shift_option  = 0
        if self.nu_shift_cards>0:
            self.shifting_phase    = True
        else:
            self.shifting_phase    = False
        self.players           = []  # stores players object
        self.on_table_cards    = []  # stores card on the table
        self.played_cards      = []  # of one game # see also in players offhand!
        self.gameOver          = 0
        self.rewards           = np.zeros((self.nu_players,))
        self.current_round     = 0
        self.setup_game(myDeck)
        self.active_player     = self.nextGamePlayer()
        self.correct_moves     = 0

        # Delete in favour of other .json file!
    # def init_Random_TestGame(self, player_type=["RANDOM", "RL", "RANDOM", "RANDOM"]):
    #     self.player_type         = player_type

    def idx2Card(self, idx):
        # input unique card index output: card object
        myDeck = deck(self.nu_cards, self.seed)
        for card in myDeck.cards:
            if card.idx == idx:
                return card

    def idx2Hand(self, idx, player_idx):
        #returns hand index of unique idx
        for i, card in enumerate(self.players[player_idx].hand):
            if card.idx == idx:
                return i

    def idxList2Cards(self, idxlist):
        result  = []
        for j in idxlist:
            result.append(self.idx2Card(j))
        return result

    def play_ai_move(self, ai_card_idx, print_=False):
        'card idx from 0....'
        current_player    =  self.active_player
        valid_options_idx = self.getValidOptions(current_player)# hand index
        card              = self.idx2Card(ai_card_idx)
        player_has_card   = self.players[current_player].hasSpecificCardOnHand(ai_card_idx)
        tmp               = card.idx
        card_options      = [self.players[current_player].hand[i].idx for i in valid_options_idx]
        card_options__    = [self.players[current_player].hand[i] for i in valid_options_idx]
        if player_has_card and tmp in card_options and "RL" in self.player_type[current_player]:
            if print_:
                if self.shifting_phase and self.nu_shift_cards>0:
                    print("[{}] {} {}\t shifts {}\tCard {}\tCard Index {}\t len {}".format(self.current_round, current_player, self.names_player[current_player], self.player_type[current_player], card, ai_card_idx, len(self.players[current_player].hand)))
                else:
                    print("[{}] {} {}\t plays {}\tCard {}\tCard Index {}\t len {}  options {} on table".format(self.current_round, current_player, self.names_player[current_player], self.player_type[current_player], card, ai_card_idx, len(self.players[current_player].hand), card_options__), self.on_table_cards)
            self.correct_moves +=1
            rewards, round_finished, gameOver = self.step(self.idx2Hand(tmp, current_player), print_)
            # if print_ and round_finished:
            #     print(rewards, self.correct_moves, gameOver, "\n")
            return rewards, round_finished, gameOver
        else:
            if print_:
                if not player_has_card:
                    print("Caution player does not have card:", card, " choose one of:", self.idxList2Cards(card_options))
                if not tmp in valid_options_idx:
                    print("Caution option idx", tmp, "not in (idx)", card_options)
                if not "RL" in self.player_type[current_player]:
                    print("Caution", self.player_type[current_player], self.active_player, "is not of type RL", self.player_type)
            return {"state": "play_or_shift", "ai_reward": None}, False, True # rewards round_finished, game_over

    def playUntilAI(self, print_=False):
        rewards        = {"state": "play_or_shift", "ai_reward": None}
        gameOver       = False
        round_finished = False
        while len(self.players[self.active_player].hand) > 0:
            current_player = self.active_player
            if "RANDOM" in self.player_type[current_player]:
                if  self.shifting_phase and self.nu_shift_cards>0:
                    hand_idx_action = self.getRandomCard()
                    card            =        self.players[current_player].hand[hand_idx_action]
                    if print_:
                        print("[{}] {} {}\t shifts {}\tCard {}\tHand Index {}\t len {}".format(self.current_round, current_player, self.names_player[current_player], self.player_type[current_player], card, hand_idx_action, len(self.players[current_player].hand)))
                else:
                    hand_idx_action = self.getRandomValidOption()
                    card            = self.players[self.active_player].hand[hand_idx_action]
                    if print_:
                        print("[{}] {} {}\t plays {}\tCard {}\tHand Index {}\t len {}".format(self.current_round, current_player, self.names_player[current_player], self.player_type[current_player], card, hand_idx_action, len(self.players[current_player].hand)))
                rewards, round_finished, gameOver = self.step(hand_idx_action, print_)
                if print_ and round_finished:
                    print("")
            else:
                return rewards, round_finished, gameOver
        # Game is over!
        #CAUTION IF GAME OVER NO REWARDS ARE RETURNED
        #rewards = {'state': 'play_or_shift', 'ai_reward': None}
        return rewards, True, True

    def state2Cards(self, state_in):
        #in comes a state matrix with len = 60 with 0...1..0...1
        indices = [i for i, x in enumerate(state_in) if int(x) == 1]
        result  = []
        for j in indices:
            result.append(self.idx2Card(j))
        return result

    def printCurrentState(self):
        #Note: ontable, onhand played play_options laenge = players* cards
        state = self.getState().flatten().astype(np.int)
        ll    = self.nu_players * self.nu_cards
        on_table, on_hand, played, play_options, add_states = state[0:ll], state[ll:2*ll], state[ll*2:3*ll], state[3*ll:4*ll], state[4*ll:len(state)]
        for i,j in zip([on_table, on_hand, played, play_options], ["on_table", "on_hand", "played", "options"]):
             #print(j, i, self.state2Cards(i))
             print("\t", j, self.state2Cards(i))

    def stepRandomPlay(self, action_ai, print_=False):
        # fängt denn ai überhaupt an???
        # teste ob correct_moves korrekt hochgezählt werden?!
        rewards, round_finished, gameOver = self.play_ai_move(action_ai, print_=print_)
        if rewards["ai_reward"] is None: # illegal move
            return None, self.correct_moves, True
        elif gameOver and "final_rewards" in rewards:
            # case that ai plays last card:
            mean_random = (sum(rewards["final_rewards"])- rewards["final_rewards"][1])/3
            return [rewards["final_rewards"][1], mean_random], self.correct_moves, gameOver
        else:
            #case that random player plays last card:
            if "RL" in self.player_type[self.active_player]:
                return [0, 0], self.correct_moves, gameOver
            else:
                rewards, round_finished, gameOver = self.playUntilAI(print_=print_)
                ai_reward   = 0
                mean_random = 0
                if gameOver and "final_rewards" in rewards:
                    mean_random = (sum(rewards["final_rewards"])- rewards["final_rewards"][1])/3
                    ai_reward = rewards["final_rewards"][1]
                return [ai_reward, mean_random], self.correct_moves, gameOver

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
        player_win_idx = self.names_player.index(winning_card.player)
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
            raise

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

    def getRandomCard(self):
        return random.randrange(len(self.players[self.active_player].hand))

    def getRandomValidOption(self):
        valid_options_idx = self.getValidOptions(self.active_player)# hand index
        rand_idx = random.randrange(len(valid_options_idx))
        return valid_options_idx[rand_idx]

    def getRandomOption_(self):
        incolor = None
        if len(self.on_table_cards)>0:
            incolor = self.on_table_cards[0].color
        options = self.players[self.active_player].getOptions(incolor)#hand index
        if len(options) == 0:
            print("Error has no options left!", options, self.players[self.active_player].hand)
            return None
        rand_card = random.randrange(len(options))
        return rand_card


    def getState(self):
        play_options = self.players[self.active_player].getBinaryOptions(self.getInColor(), self.nu_players, self.nu_cards, self.shifting_phase)# TODO self.shifting_phase
        #play_options = self.convertAvailableActions(play_options)
        on_table, on_hand, played = self.getmyState(self.active_player, self.nu_players, self.nu_cards)
        add_states = [] #(nu_players-1)*5
        for i in range(len(self.players)):
            if i!=self.active_player:
                add_states.extend(self.getAdditionalState(i))
        return np.asarray([on_table+ on_hand+ played+ play_options+ add_states])

    def getLenStates(self):
        len_add_states = self.nu_players *5 #
        len_on_hand    = self.nu_players * self.nu_cards
        return len_add_states, len_on_hand

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
        # return hand index of options
        if self.shifting_phase and self.nu_shift_cards>0:
            options = [x for x in range(len(self.players[player].hand))]
            return options
        else:
            return self.players[player].getOptions(self.getInColor())

    def convertTakeHand(self, player, take_hand):
        converted_cards = []
        for card in take_hand:
            card.player = player.name
            converted_cards.append(card)
        return converted_cards

    def step(self, card_idx, print_=False):
        #Note that card_idx is a Hand Card IDX!
        # it is not card.idx unique number!
        self.shifting_phase = (self.shifted_cards<=self.nu_players*self.nu_shift_cards)
        if self.shifting_phase and self.nu_shift_cards>0:
            shift_round   = int(self.shifted_cards/self.nu_players)
            self.shiftCard(card_idx, self.active_player, self.getShiftPlayer())
            self.shifted_cards +=1

            round_finished = False
            if self.shifted_cards%self.nu_players == 0:
                round_finished = True
            #if print_: print("Shift Round:", shift_round, "Shifted Cards:", self.shifted_cards, "round_finished", round_finished)
            if shift_round == (self.nu_shift_cards)-1 and round_finished:
                if print_: print("\nShifting PHASE FINISHED!!!!!!\n")
                for player in self.players:
                    # convert cards of take hand card.player to correct player!
                    player.take_hand = self.convertTakeHand(player, player.take_hand)
                    player.hand.extend(player.take_hand)
                    if print_: print(player.name, "takes now", player.take_hand, " all cards", player.hand)
                self.shifted_cards  = 100
                self.shifting_phase = False
            self.active_player = self.getNextPlayer()
            return {"state": "shift", "ai_reward": 0}, round_finished, False # rewards, round_finished, gameOver
        else:
            # in case card_idx is a simple int value
            round_finished = False
            # play the card_idx:
            played_card = self.players[self.active_player].hand.pop(card_idx)
            self.on_table_cards.append(played_card)
            # Case round finished:
            trick_rewards    = [0, 0, 0, 0]
            on_table_win_idx = -1
            player_win_idx   = -1
            if len(self.on_table_cards) == self.nu_players:
                winning_card, on_table_win_idx, player_win_idx = self.evaluateWinner()
                trick_rewards[player_win_idx] = self.players[player_win_idx].countResult([self.on_table_cards])
                self.current_round +=1
                self.played_cards.extend(self.on_table_cards)
                self.players[player_win_idx].appendCards(self.on_table_cards)
                self.on_table_cards = []
                self.active_player  = player_win_idx
                round_finished = True

            else:
                self.active_player = self.getNextPlayer()

            if round_finished and len(self.played_cards) == self.nu_cards*self.nu_players:
                self.assignRewards()
    		#yes this is the correct ai reward in case all players are ai players.
            return {"state": "play", "ai_reward": trick_rewards[player_win_idx], "on_table_win_idx": on_table_win_idx, "trick_rewards": trick_rewards, "player_win_idx": player_win_idx, "final_rewards": self.rewards}, round_finished, self.isGameFinished()


    def shiftCard(self, card_idx, current_player, next_player):
        # shift round = 0, 1, ... (for 2 shifted cards)
        #print("I shift now hand idx", card_idx, "from", self.players[current_player].name, "to", self.players[next_player].name)
        card = self.players[current_player].hand.pop(card_idx) # wenn eine Karte weniger index veringern!
        self.players[next_player].take_hand.append(card)

    def assignRewards(self):
        for i, player in enumerate(self.players):
            #print(i, player.offhand)
            self.rewards[i] = player.countResult(player.offhand)

    def getAdditionalState(self, playeridx):
        result = []
        player = self.players[playeridx]

        #extend if this player would win the current cards
        player_win_idx = playeridx
        if len(self.on_table_cards)>0:
            winning_card, on_table_win_idx, player_win_idx = self.evaluateWinner()
        if player_win_idx == playeridx:
            result.extend([1])
        else:
            result.extend([0])
        result.extend(player.colorFree) # 4 per player -> 12 states
        return result


    def getmyState(self, playeridx, players, cards):
        on_table, on_hand, played =[0]*players* cards, [0]*players* cards, [0]*players* cards
        for card in self.on_table_cards:
            on_table[card.idx]= 1

        for card in self.players[playeridx].hand:
            on_hand[card.idx] =1

        for card in self.played_cards:
            played[card.idx] = 1
        return on_table, on_hand, played
