from twisted.internet.protocol import Factory
from twisted.internet import reactor, protocol

import onnxruntime
import numpy as np
from gameClasses import card, deck, player, game
import json
import time

# Copyright
# Author Markus Lamprecht (www.simact.de) 08.12.2019
# A card game named Witches:

class ServerProtocol(protocol.Protocol):
    def __init__(self, factory):
        self.factory = factory

    def connectionMade(self):
        print("Server: New Connection Made")
        self.factory.activeConnections += 1

    def dataReceived(self, data):
        inMsg = str(data, encoding="utf8")
        print ("\n> Received: [%d] %s" % (self.factory.activeConnections, inMsg))

        ##Parse inMsg
        name, command, msg, tmp, outMsg, ende ="", "", "", "", "", ""
        try:
            tmp = inMsg.split(";")
        except Exception as e:
            print(e)
            outMsg = name+";"+"Error;"+"Server could not parse Message split fails"

        # Wenn Nachriten doppelt ankommen. Nimm nur erstere
        if len(tmp)>4:
            test = inMsg.split(";Ende")
            name, command, msg, ende = tmp[0], tmp[1], tmp[2], tmp[3]
        elif len(tmp) == 4:
            name, command, msg, ende = tmp[0], tmp[1], tmp[2], tmp[3]
        else:
            outMsg = name+";"+"Error;"+"Server could not parse Message not enough args found"+str(len(tmp))+" should be 4"

        if "InitClient" in command and self.factory.serverState =="INIT":
            if name not in self.factory.connectedPlayers:
                self.factory.connectedPlayers.append(name)
                outMsg = name+";"+"InitClientSuccess"+";"+"Hello "+name+" wait for "+str(self.factory.nuClients-len(self.factory.connectedPlayers))+" more Clients...."+";"+"Ende"
            else:
                outMsg = name+";"+"Error"+";"+"Hello "+name+" there is already a player connected with this name. Connected Players: "+str(self.factory.nuClients-len(self.factory.connectedPlayers))+" "+";"+"Ende"
            if len(self.factory.connectedPlayers) == self.factory.nuClients:
                self.factory.serverState  = "ALL_CONNECTED"

        elif "GetCards" in command:
            if self.factory.serverState  == "ALL_CONNECTED" or self.factory.serverState  =="ALL_DEALT":
                # Assign new names :
                names  = str(self.factory.getNewNames())
                type   = str(self.factory.getType())
                cards  = self.factory.getAllCards()
                backCard   = str(self.factory.getBack())
                outMsg = name+";"+"GetCardsSuccess"+";"+names+"--"+type+"--"+cards+backCard+";"+"Ende"
                self.factory.serverState  = "ALL_DEALT"
                #self.factory.checkDelete(name)
                #self.factory.boardState = []# geht nicht mit schlechter Verbindung!! (e.g. Handy!)
            else:
                outMsg =  name+";"+"WaitUntilConnected"+";"+"Hello "+name+" we have to wait until all players are connected before I give you your cards"+";"+"Ende"

        elif "WantPlay" in command and self.factory.serverState == "ALL_DEALT":
            self.factory.isReset =  False
            is_turn              = self.factory.checkIsTurn(name)
            is_valid             = self.factory.isValid(msg)
            if is_valid and is_turn and self.factory.deleteCounter>5:
                player_idx, shifting, shifted_cards, on_table_cards, action = self.factory.getPlayedCard(msg)

                # play virtual Card!
                rewards, round_finished, gameOver = self.factory.playVirtualCard(action)
                if not "total_rewards" in rewards:
                    rewards = "[]"
                else:
                    rewards = str(list(rewards["total_rewards"]))

                if gameOver:
                    #self.factory.serverState  = "GAME_OVER"
                    self.factory.LastRewards  = rewards

                self.factory.appendBoardState([player_idx, msg, shifting, shifted_cards, on_table_cards, rewards, str(round_finished), str(gameOver)])

                outMsg = name+";"+"PlayedCard"+";"+player_idx+"--"+msg+"--"+shifting+"--"+shifted_cards+"--"+on_table_cards+"--"+str(rewards)+"--"+str(round_finished)+"--"+str(gameOver)+";Ende"
            else:
                outMsg = name+";"+"WrongCard"+";"+str(self.factory.boardState)+";"+"Ende"
                self.factory.checkDelete(name)
        elif "GameOver" in command:
            # send gameOver Cards....
            print(self.factory.getGameOverState())
            print("Last rewards\n",str(self.factory.LastRewards))
            outMsg = name+";"+"GameOver"+";"+self.factory.getGameOverState()+"--"+self.factory.LastRewards+"--"+str(list(self.factory.my_game.total_rewards))+";Ende"

        elif "Restart" in command:
            if self.factory.gameOverSendToAll(name):
                print("\n\n RESTART SERVER \n\n")
                self.factory.reset()
                outMsg = name+";"+"Restart"+";"+""+";"+"Ende"
            else:
                print("Send GameOver State again by SERVERRRR")
                outMsg = name+";"+"GameOver"+";"+self.factory.getGameOverState()+"--"+self.factory.LastRewards+"--"+str(list(self.factory.my_game.total_rewards))+";Ende"

        print("Server out:",  self.factory.options["names"][self.factory.my_game.active_player], outMsg)
        self.transport.write(outMsg.encode("utf8"))

    def connectionLost(self, reason):
        self.factory.activeConnections -= 1

class ServerFactory(Factory):
    def __init__(self):
        #Setup the Game:
        self.options_file_path =  "server_options.json"
        with open(self.options_file_path) as json_file:
            self.options = json.load(json_file)
        self.my_game = game(self.options)
        print("Server opened the game with:", self.options["type"])

        self.nuClients        = self.countClients()
        self.connectedPlayers = []
        self.activeConnections= 0
        self.serverState      ="INIT"
        self.boardState       = []
        self.LastRewards      = ""
        self.nuVisited        = {}
        self.gOverVisited     = {}
        self.nexPlay          = {}
        self.lastBoardState   = []
        self.deleteCounter    = 0
        self.isReset          = False


    def gameOverSendToAll(self, name, nu=3):
        if name in self.gOverVisited:
            self.gOverVisited[name] +=1
        else:
            self.gOverVisited[name] = 0

        print(self.gOverVisited)
        if self.getMaxVisited(self.gOverVisited)>nu:
            return True
        return False

    def reset(self):
        # reset only once!
        print("Server inside reset")
        if not self.isReset:
            print("Insideeeee hereee")
            self.isReset          = True
            self.my_game.reset_game()
            self.serverState      ="ALL_CONNECTED"
            self.boardState       = []
            self.LastRewards      = ""
            self.nuVisited        = {}
            self.gOverVisited     = {}
            self.lastBoardState   = []
            self.deleteCounter    = 0
            # do not play client here cause then card is not in hand and will not be dealt!

    def countClients(self):
        res = 0
        for i in self.options["type"]:
            if i =="Client":
                res +=1
        return res

    def getType(self):
        return self.options["type"]

    def getBack(self):
        return self.options["back_CardColor"]

    def getNewNames(self):
        j = 0
        for u,i in enumerate(self.options["type"]):
            if i =="Client":
                self.options["names"][u] = self.connectedPlayers[j]
                j +=1
        return self.options["names"]

    def getAllCards(self):
        cards = ""
        for i in range(len(self.options["names"])):
            cards +=str(self.my_game.players[i].hand)+"--"
            if len(self.my_game.players[i].hand)==0:
                print(eee)
        return cards

    def getGameOverState(self):
        cards = []
        for i in range(4):
            cards.append([item for sublist in  self.my_game.players[i].offhand for item in sublist])
        return str(cards)

    def checkIsTurn(self, name):
        a =  self.my_game.active_player
        print("CHeck isTurn", name, self.options["names"][a], self.my_game.active_player, self.options["names"][a] == name)
        if self.options["names"][a] == name:
            return True
        return False

    def convertCardString2Card(self, cardmsg):
        tmp = cardmsg.split("of")
        value = int(tmp[0].replace("'",""))
        color = str(tmp[1].replace("of","").replace(" ","").replace("'",""))
        return card(color, value)

    def isValid(self, name):
        if len(name) == 0:
            return False
        card_msg = ""
        try:
            card_msg = self.convertCardString2Card(name)
        except:
            print("No valid card message:", name)
            return False

        return self.my_game.validMove(card_msg)

    def getPlayedCard(self, name):
        player_idx     = self.my_game.active_player
        shifting       = self.my_game.shifting_phase
        shifted_cards = self.my_game.shifted_cards
        on_table_cards = self.my_game.on_table_cards
        card           = self.convertCardString2Card(name)
        action         = self.my_game.players[player_idx].specificIndexHand(card)
        return str(player_idx), str(shifting), str(shifted_cards), str(len(on_table_cards)), action

    def getPlayedCardPC(self, action):
        player_idx     = self.my_game.active_player
        shifting       = self.my_game.shifting_phase
        shifted_cards  = self.my_game.shifted_cards
        on_table_cards = self.my_game.on_table_cards
        try:
            card           = self.my_game.players[player_idx].hand[action]
        except:
            print("Action was", action, "but does not have:", self.my_game.players[player_idx].hand)
            card           = self.my_game.players[player_idx].hand[0]
        return str(player_idx), str(shifting), str(shifted_cards), str(len(on_table_cards)), str(card)

    def getMaxVisited(self, dict_):
        array = []
        for key in dict_:
            if not "_send" in key:
                array.append(dict_[key])
        if len(array) == 0: return 0
        return min(array)

    def checkDelete(self, name):
        print(self.nuVisited, self.deleteCounter)
        if name in self.nuVisited:
            if len(str(self.lastBoardState)) == len(str(self.boardState)):
                self.nuVisited[name] +=1
                self.deleteCounter   +=1
                self.nuVisited[name+"_send"] = self.boardState
        else:
            self.nuVisited[name] = 0

        #print("VISITED", self.nuVisited, self.deleteCounter)
        if self.getMaxVisited(self.nuVisited)>3 and self.deleteCounter>5:
            for key in self.nuVisited:
                self.nuVisited[key]  = 0
            self.lastBoardState = []
            self.boardState     = []
            self.playVirtualStep()

    def appendBoardState(self, entry):
        self.deleteCounter  = 0
        self.lastBoardState = self.boardState
        self.boardState.append(entry)

    def playVirtualStep(self):
        print(self.my_game.ai_player, self.my_game.names_player, self.my_game.ai_player[self.my_game.active_player], len(self.my_game.players[self.my_game.active_player].hand))
        if not "Client" in self.my_game.ai_player[self.my_game.active_player] and len(self.my_game.players[self.my_game.active_player].hand)>0:
            action = self.selectAction()
            player_idx, shifting, shifted_cards, on_table_cards, card = self.getPlayedCardPC(action)
            print("\tClient:", self.my_game.ai_player[self.my_game.active_player], self.options["names"][self.my_game.active_player], "played", card)
            rewards, round_finished, gameOver = self.playVirtualCard(action)
            if not "total_rewards" in rewards:
                rewards = "[]"
            else:
                rewards = str(list(rewards["total_rewards"]))


            self.appendBoardState([player_idx, card, shifting, shifted_cards, on_table_cards, rewards, str(round_finished), str(gameOver)])
            #self.checkDelete(self.my_game.ai_player[self.my_game.active_player])

            if gameOver:
                #self.serverState  ="GAME_OVER"
                self.LastRewards  = rewards
                print("GAME_OVER", rewards, round_finished)
                return

    def selectAction(self):
        # action is a hand card index or???
        # Version 2.0 shifting active do not use nn, mcts anymore!
        current_player = self.my_game.active_player
        if "RL"  in self.my_game.ai_player[current_player]:
            state_240 = self.my_game.getState_240().flatten()
            state_303 = self.my_game.getState_303().flatten() # used in rl_path11_op
            try:
                rl_type = int(''.join(x for x in self.my_game.ai_player[current_player] if x.isdigit()))
            except:
                print("Error did not find rl_type set it to 1")
                rl_type = 1
            action = self.rl_onnx(state_240, state_303, "../data/"+self.options["onnx_rl_path"][rl_type]+".onnx")
            card   = self.my_game.players[current_player].getIndexOfCard(action)
            action = self.my_game.players[current_player].specificIndexHand(card)
            is_allowed_list_idx = self.my_game.getValidOptions(self.my_game.active_player)
            incolor =self.my_game.getInColor()
            if action not in is_allowed_list_idx and incolor is not None:
                print("RL: ACTION NOT ALLOWED!", card)
                print("I play random possible option instead")
                action = self.my_game.getRandomOption_()
        else:# "RANDOM":
            if self.my_game.shifting_phase:
                action = self.my_game.getRandomCards()[0]
            else:
                action = self.my_game.getRandomOption_()
        return action

    def rl_onnx(self, state_240, state_303, path):
        '''Input:
        x:      180x1 list binary values
        path    *.onnx (with correct model)'''
        ort_session = onnxruntime.InferenceSession(path)
        if ort_session.get_inputs()[0].shape[0] == 240:
            ort_inputs  = {ort_session.get_inputs()[0].name: np.asarray(state_240, dtype=np.float32)}
        elif ort_session.get_inputs()[0].shape[0]==303:
            ort_inputs  = {ort_session.get_inputs()[0].name: np.asarray(state_303, dtype=np.float32)}
        else:
            print("Error wrong inputs!")
        ort_outs    = ort_session.run(None, ort_inputs)
        max_value = (np.amax(ort_outs))
        result = np.where(ort_outs == np.amax(ort_outs))
        return result[1][0]

    def playVirtualCard(self, actionHandIdx):
         return self.my_game.step_idx_with_shift(actionHandIdx)

    def buildProtocol(self, addr):
        return ServerProtocol(self)

reactor.listenTCP(8000, ServerFactory())
#only required if not in main thread see here: https://twistedmatrix.com/trac/wiki/FrequentlyAskedQuestions#Igetexceptions.ValueError:signalonlyworksinmainthreadwhenItrytorunmyTwistedprogramWhatswrong
reactor.run(installSignalHandlers=0)
