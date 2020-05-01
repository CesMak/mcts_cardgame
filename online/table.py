import sys
import os
from PyQt5.QtGui  import QFont, QPainter, QColor, QBrush, QPen
from PyQt5.QtCore import QDataStream, QIODevice, QByteArray, QCoreApplication, QEventLoop, pyqtSignal, pyqtSlot, Qt, QRectF, QPointF, QTimer, QPoint
from PyQt5.QtWidgets import QApplication, QPushButton, QGraphicsView, QWidget, QGraphicsScene, QGridLayout
from PyQt5.QtNetwork import QTcpSocket, QAbstractSocket
from PyQt5 import QtSvg
import time
import easygui
import json
from prettyjson import prettyjson

# Building an exe use onnx
import onnxruntime
import numpy as np

import threading
import pickle

# For server / client:
import socket # required in get IP
import re
import ast
from gameClasses import card, deck, player, game

import urllib.request

import random

# Copyright
# Author Markus Lamprecht (www.simact.de) 08.12.2019
# A card game named Witches:

class QGraphicsViewExtend(QGraphicsView):
    """ extends QGraphicsView for resize event handling  """
    def __init__(self, parent=None):
        super(QGraphicsViewExtend, self).__init__(parent)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def resizeEvent(self, event):
        #self.fitInView(QRectF(0,0,640,480),Qt.KeepAspectRatio)
        self.fitInView(QRectF(self.viewport().rect()),Qt.KeepAspectRatio)

class CardGraphicsItem(QtSvg.QGraphicsSvgItem):
    """ Extends QtSvg.QGraphicsSvgItem for card items graphics """
    def __init__(self, my_card, ind, svgFile, player=0, faceDown=True):
        super(CardGraphicsItem, self).__init__(svgFile)
        self.card     = card(my_card.color, my_card.value)
        self.svgFile  = svgFile # svg file for card graphics
        self.player   = player # which player holds the card
        self.faceDown = faceDown # does the card faceDown
        self.isPlayed = False

    def __repr__(self):
        return '<CardGraphicsItem: %s>' % self.card


class cardTableWidget(QWidget):
    server_receivedSig             = pyqtSignal(str)
    """ main widget for handling the card table """
    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)
        # init is done in start_server.py

    def initUI(self, path_to_options):
        """ initialize the view-scene graphic environment """
        self.scene = QGraphicsScene()
        #self.scene.setSceneRect(0, 0, 640, 480)
        self.view = QGraphicsViewExtend(self.scene)
        self.view.setSceneRect(QRectF(self.view.viewport().rect()))
        self.view.setSceneRect(QRectF(0,0, 850, 900))
        self.view.setRenderHint(QPainter.Antialiasing)
        layout = QGridLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)
        self.setBackgroundColor(QColor('green'))

        # special properties
        self.svgCardsPath = "../cards"
        self.cardsGraphItems = [] #holds all the cards items
        self.defInsertionPos = QPointF(0,0)
        self.defAngle = 0
        self.defScale = 0.5
        self.deckBackSVG = 'back_1'
        self.numOfPlayers = 4
        self.playersHandsPos = [(75, 50, 0), (210, 50, 180), (680, 50, 0), (210, 385, 0)] #(x,y,angle)
        self.defHandSpacing = 24
        self.midCards  = []
        self.options_file_path =  path_to_options

        # Card fields
        pen = QPen()
        brush = QBrush()
        self.scene.addRect(QRectF(200, 230, 100, 80), pen, brush)
        self.scene.addRect(QRectF(200+120, 230, 100, 80), pen, brush)
        self.scene.addRect(QRectF(200+120*2, 230, 100, 80), pen, brush)
        self.scene.addRect(QRectF(200+120*3, 230, 100, 80), pen, brush)

        # Player Names
        self.player1_label = self.addPlayerLabel(425, 350, "Player 1")
        self.player2_label = self.addPlayerLabel(0, 240,   "Player 2")
        self.player3_label = self.addPlayerLabel(425, 20,  "Player 3")
        self.player4_label = self.addPlayerLabel(782, 240, "Player 4")

        self.card1_label   = self.addPlayerLabel(200, 210,       "")
        self.card2_label   = self.addPlayerLabel(200+120, 210,   "")
        self.card3_label   = self.addPlayerLabel(200+120*2, 210, "")
        self.card4_label   = self.addPlayerLabel(200+120*3, 210, "")

        self.card_label_l  = [self.card1_label, self.card2_label, self.card3_label, self.card4_label]
        self.card_label_pla= [self.player1_label, self.player2_label, self.player3_label, self.player4_label]

        self.play_1_state  = self.addPlayerLabel(200, 250, "")
        self.play_2_state  = self.addPlayerLabel(200+120, 250, "")
        self.play_3_state  = self.addPlayerLabel(200+120*2, 250, "")
        self.play_4_state  = self.addPlayerLabel(200+120*3, 250, "")
        self.game_indicator= self.addPlayerLabel(650, 5, "Game: ")
        self.mode_label    = self.addPlayerLabel(150, 5, "Mode: ")

        playbtn = QPushButton('Start', self)
        playbtn.resize(50, 32)
        playbtn.move(10, 10)
        playbtn.clicked.connect(self.start_clicked)

        options = QPushButton('Options', self)
        options.resize(80, 32)
        options.move(65, 10)
        options.clicked.connect(self.options_clicked)

        nextRound = QPushButton('nextRound', self)
        nextRound.resize(80, 32)
        nextRound.move(150, 10)
        nextRound.setVisible(False)
        nextRound.clicked.connect(self.nextRound_clicked)

        self.scene.addWidget(playbtn)
        self.scene.addWidget(nextRound)
        self.scene.addWidget(options)

        self.my_game = None

        # Testing tree:
        self.my_tree = None

        # Storing game_play
        self.game_play = {}

        self.corrString       = ""
        # emit signal:
        self.server_receivedSig.connect(self.parseClient)

        ### Client stuff:
        self.clientTimer      = QTimer(self)
        self.tcpSocket        = None
        self.games_played     = 0
        self.reset_client()


    def reset_client(self):
        # used  also in "Restart"
        self.removeAll()
        self.deckBackSVG      = 'back_1' # in shifting phase soll man karten sehen!
        self.clientCards      = None
        self.ClientName       = ""
        self.dealAgain        = False
        self.GameOver         = False
        self.gotCards         = 0
        self.rounds_played    = 1
        self.nuSend           = 0
        self.nuReceived       = 0
        self.wantPlay         = ""
        self.games_played     += 1


    def options_clicked(self):
        '''
        Read in json, modify, write it read it in as dict again!
        '''
        with open(self.options_file_path) as json_file:
            test = json.load(json_file)
            txt = prettyjson(test)
        text = easygui.textbox("Contents of file:\t"+self.options_file_path, "Adjust your options", txt)
        if text is not None: # Cancel pressed (None case)
            dict = (json.loads(text))
            with open(self.options_file_path, 'w') as outfile:
                json.dump(dict, outfile)

    def nextRound_clicked(self):
        '''
        Reset the game with the same options as before!
        '''
        print("This option is not available!")

#########################CLIENT #################################
#########################CLIENT #################################
#########################CLIENT #################################
    def convertCardsArray(self, stringArray):
        cards = []
        for ele in ast.literal_eval(stringArray):
            cards.append(self.convertCardString2Card(ele))
        return cards

    def convertCardString2Card(self, cardmsg):
        tmp = cardmsg.split("of")
        value = int(tmp[0].replace("'",""))
        color = str(tmp[1].replace("of","").replace(" ","").replace("'",""))
        return card(color, value)

    def send_msgClient(self, msg):
        self.tcpSocket.waitForBytesWritten(100) # waitForBytesWritten  waitForConnected
        self.tcpSocket.write(bytes( str(msg), encoding='ascii'))

    def displayErrorClient(self, socketError):
        self.changePlayerName(self.mode_label,  "Error just wait")
        if socketError == QAbstractSocket.RemoteHostClosedError:
            pass
        else:
            print("Server does not seem to be open or wrong open_ip!")
            if not self.clientTimer.isActive():
                print("The following error occurred: %s." % self.tcpSocket.errorString())
                self.clientTimer.timeout.connect(self.clientReconnectTimer)
                self.clientTimer.start(2000)

    def applyOneState(self, state):
        # only for single message
        res = []
        for i in state:
            res.append(ast.literal_eval(i))

        try:
            player_idx, my_card, shifting, nu_shift_cards, on_table_cards, endround, gameOver = int(res[0]), self.convertCardString2Card(res[1]), str(res[2]), int(res[3]), int(res[4]), res[6], res[7]
        except:
            print("Could not parse ERRROR")
            return

        print("apply Board state:", res, self.gotCards)

        #Do not play if card is already deleted or in the mid
        item = self.findGraphicsCardItem_(my_card)
        if item is not None:
            print(item, item.isPlayed)
        if item is None or item.isPlayed:
            return

        if "False" in shifting:
            shifting = False
        else:
            shifting = True

        if not shifting:
            on_table_cards = len(self.midCards)

        if "True" in str(gameOver):
            self.GameOver = True

        # If cards are dealt again do not apply shifted states again (they still might be contained in board state)
        if shifting and self.gotCards >=2:
            return
        self.playCardClient(item, player_idx, on_table_cards, player_idx, shifting, nu_shift_cards)

    def applyBoardState(self, msg):
        if len(msg) == 0:
            return
        for i in msg:
            self.applyOneState(i)


    @pyqtSlot(str)
    def parseClient(self, inMsg):
        self.nuReceived +=1
        #self.clientTimer.stop()
        print("\nReceived::", inMsg)

        name, command, msg, tmp, outMsg ="", "", "", "", ""
        try:
            tmp = inMsg.split(";")
        except Exception as e:
            print(e)
            outMsg = name+";"+"Error;"+"Server could not parse Message split fails"

        if len(tmp) == 3:
            name, command, msg = tmp[0], tmp[1], tmp[2]
        else:
            print("Not enough ;")
            outMsg = name+";"+"Error;"+"Server could not parse Message not enough args found"+str(len(tmp))+" should be 3"

        #### TODO
        #### Achtung wenn gleiche Nachricht 2 mal kommt soll nichts gemacht werden
        #### Warte bis timer fertig bevor neuen starten!!!

        if command == "InitClientSuccess" or command =="WaitUntilConnected":
            self.changePlayerName(self.mode_label,  "Mode: Shift")
            self.send2Server("GetCards", "Server give me my cards and the game state", once=False)
        elif command == "GetCardsSuccess":
            self.gotCards +=1
            # Wende nur zwei mal an!!!
            if self.gotCards <=2:
                try:
                    ttmp = msg.split("--")
                    names, typee, cards, deck = ttmp[0], ttmp[1], ttmp[2:6], ttmp[6]
                except:
                    print("Could not parse ERRROR")
                    return

                self.options["names"] = ast.literal_eval(names)
                self.options["type"]  = ast.literal_eval(typee)
                self.setNames()

                for i,c in enumerate(cards):
                    if self.options["names"][i] == self.clientName:
                        self.clientCards = self.convertCardsArray(c)
                        self.deal_cards(self.convertCardsArray(c), i, fdown=False)
                    else:
                        self.deal_cards(self.convertCardsArray(c), i, fdown=True)
                self.deckBackSVG = str(ttmp[6])

            self.send2Server("WantPlay", str(self.wantPlay), once=False)
        elif command  == "WrongCard":
            self.applyBoardState(ast.literal_eval(msg))
            if self.dealAgain:
                self.send2Server("GetCards", str(self.wantPlay), once=False)
                self.dealAgain = False
            elif self.GameOver:
                self.send2Server("GameOver", "game is over", once=False)
                self.GameOver = False
            else:
                ########### random card for testing:
                # item = None
                # if len(self.clientCards)>0:
                #     while item is None:
                #         number = random.randrange(len(self.clientCards))
                #         my_card = self.clientCards[number]
                #         item = self.findGraphicsCardItem_(my_card)
                #     self.wantPlay = my_card
                #########
                self.send2Server("WantPlay", str(self.wantPlay), once=False)
        elif command =="PlayedCard":
            self.wantPlay = ""
            try:
                ttmp = msg.split("--")
                player_idx, card, shifting, nu_shift_cards, on_table_cards, endround, gameover = int(ttmp[0]), self.convertCardString2Card(ttmp[1]), ast.literal_eval(ttmp[2]), int(ttmp[3]), int(ttmp[4]), ttmp[6], ttmp[7]
            except:
                print("Could not parse ERRROR")
                return

            #Do not play if card is already deleted or in the mid
            item = self.findGraphicsCardItem_(card)
            if item is None or item.isPlayed:
                print("ERROR CARD NOT FOUND!!!! \n\n")
                self.send2Server("WantPlay", str(self.wantPlay), once=False)
                return

            if not shifting:
                on_table_cards = len(self.midCards)

            try:
                self.playCardClient(item, player_idx, on_table_cards, player_idx, shifting, nu_shift_cards)
            except:
                self.send2Server("WantPlay", str(self.wantPlay), once=False)
                return

            if self.dealAgain:
                self.send2Server("GetCards", str(self.wantPlay), once=False)
                self.dealAgain = False
            elif "True" in gameover:
                self.send2Server("GameOver", "game is over", once=False)
            else:
                self.send2Server("WantPlay", str(self.wantPlay), once=False)
        elif command =="GameOver":
            # do this only once:
            if not self.dealAgain:
                self.removeAll()
                self.dealAgain =  True
                try:
                    ttmp = msg.split("--")
                    offhandCards, rewards, total_rewards = ttmp[0], ttmp[1], ttmp[2]
                except Exception as e:
                    print("Could not parse ERRROR", e)
                    return

                offhandCards = ast.literal_eval(offhandCards)
                rewards =  ast.literal_eval(rewards)
                total_rewards =  ast.literal_eval(total_rewards)

                for i in range(len(offhandCards)):
                    self.showResultClient(i, str(rewards[i]), str(total_rewards[i]), self.convertCardsArray(str(offhandCards[i])))

                print("CLIENT GAME OVERRRRR")
            self.send2Server("Restart", "Server Please Restart me", once=False)
        elif command=="Restart":
            print("Inside client restart and reset now")
            self.reset_client()
            time.sleep(3)# wait some time before starting new!
            self.send2Server("GetCards", "Server give me my cards for the new game", once=False)

    def receivedMsgClient(self):
        inMsg = str(self.tcpSocket.readAll(), encoding='utf8')
        if ";Ende" in inMsg:
            a = (inMsg.split(";Ende"))
            self.corrString +=a[0]

            self.server_receivedSig.emit(self.corrString)
            self.corrString = ""
        else:
            self.corrString +=inMsg

    def sendServer(self, msg):
        # wait for answer here: https://stackoverflow.com/questions/23265609/persistent-connection-in-twisted
        # see also this https://stackoverflow.com/questions/23265609/persistent-connection-in-twisted (send to multiple....)
        print(">>>Client sends:", msg)
        msg = msg.encode("utf8")
        self.tcpSocket.write(msg)
        self.nuSend +=1
        self.tcpSocket.waitForBytesWritten()

    def send2Server(self, cmd, msg, once=True, delimiter=";"):
        msg = self.clientName+delimiter+cmd+delimiter+msg+delimiter+"Ende"
        if once:
            self.sendServer(msg)
        else:
            # Send after a while to minimize the traffic
            QTimer.singleShot(100, lambda: self.sendServer(msg))

    def openClient(self):
        self.tcpSocket = QTcpSocket(self)
        print("I client connect now with:", self.options["open_ip"])
        self.tcpSocket.connectToHost(self.options["open_ip"], 8000, QIODevice.ReadWrite)
        self.tcpSocket.readyRead.connect(self.receivedMsgClient)
        self.tcpSocket.error.connect(self.displayErrorClient)

        # send start message:
        connected = self.tcpSocket.waitForConnected(1000)
        if connected:
            self.clientTimer.stop()
            self.send2Server("InitClient","Server please init me with my name")
        else:
            print("Not connected, Server not open?, open_ip wrong? Try to reconnect in 2sec")

    def is_valid_ipv4(self, ip):
        """Validates IPv4 addresses.
        """
        pattern = re.compile(r"""
            ^
            (?:
              # Dotted variants:
              (?:
                # Decimal 1-255 (no leading 0's)
                [3-9]\d?|2(?:5[0-5]|[0-4]?\d)?|1\d{0,2}
              |
                0x0*[0-9a-f]{1,2}  # Hexadecimal 0x0 - 0xFF (possible leading 0's)
              |
                0+[1-3]?[0-7]{0,2} # Octal 0 - 0377 (possible leading 0's)
              )
              (?:                  # Repeat 0-3 times, separated by a dot
                \.
                (?:
                  [3-9]\d?|2(?:5[0-5]|[0-4]?\d)?|1\d{0,2}
                |
                  0x0*[0-9a-f]{1,2}
                |
                  0+[1-3]?[0-7]{0,2}
                )
              ){0,3}
            |
              0x0*[0-9a-f]{1,8}    # Hexadecimal notation, 0x0 - 0xffffffff
            |
              0+[0-3]?[0-7]{0,10}  # Octal notation, 0 - 037777777777
            |
              # Decimal notation, 1-4294967295:
              429496729[0-5]|42949672[0-8]\d|4294967[01]\d\d|429496[0-6]\d{3}|
              42949[0-5]\d{4}|4294[0-8]\d{5}|429[0-3]\d{6}|42[0-8]\d{7}|
              4[01]\d{8}|[1-3]\d{0,9}|[4-9]\d{0,8}
            )
            $
        """, re.VERBOSE | re.IGNORECASE)
        return pattern.match(ip) is not None

#########################CLIENT #################################
#########################CLIENT #################################
#########################CLIENT #################################


#########################SERVER #################################
#########################SERVER #################################
#########################SERVER #################################



    def start_server(self):
        import server
        time.sleep(5)

    def getIP(self):
        hostname = socket.gethostname()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
        return ip_address, hostname

    def findFirstClient(self, list):
        for j,i in enumerate(list):
            if "Client" in i:
                return j
        return 0

#########################SERVER #################################
#########################SERVER #################################
#########################SERVER #################################
    def start_clicked(self):
        #1. Load Options
        with open(self.options_file_path) as json_file:
            self.options = json.load(json_file)

        #2. Create Game:
        if self.options["automatic_mode"]:
            with open(self.options["game_play_path"], 'rb') as f:
                self.game_play = pickle.load(f)
            self.automatic_mode()
        else:
            print("Online_type:", self.options["online_type"])
            if "Client" in self.options["online_type"]:
                self.changePlayerName(self.mode_label,  "Mode: Client")
                valid_ip = self.is_valid_ipv4(self.options["open_ip"])
                if len(self.options["names"])>1 or len(self.options["type"])>1 or (not valid_ip) or ("Client" not in self.options["type"]):
                    print("Error use only one unique name in options.  names: ['YourName']")
                    print("Error use only one type in options.         type: ['Client']")
                    print("Error use only IPV4 as open_ip in options.  open_ip: 172.20.80.10")
                    return
                self.clientName = self.options["names"][0]
                self.openClient()
            elif "Server" in self.options["online_type"]:
                self.changePlayerName(self.mode_label,  "Mode: Server")
                #1. Open Server in seperate Thread
                page = str(urllib.request.urlopen("http://checkip.dyndns.org/").read())
                print(">>>THIS IS SERVER OPEN IP ADDRESS:",  re.search(r'.*?<body>(.*).*?</body>', page).group(1))
                print(">>>LOCAL IP OF THIS PC IN LAN    :", self.getIP())
                server_thread = threading.Thread(target=self.start_server, )
                server_thread.start()

                #2. Open Client
                self.options["online_type"] = "Client"
                # give it the name of the first found Client
                self.clientName = self.options["names"][self.findFirstClient(self.options["type"])]
                self.openClient()
            else:
                print("ERROR TODO Mode not online")

    def automatic_mode(self):
        print("inside automatic mode")
        for i in range(4):
            self.deal_cards(self.game_play["cards_player_"+str(i)], i, fdown=False)
        j = 0
        for player, action  in self.game_play["moves"]:
            if (j==4):
                j=0
            active_player_cards = self.game_play["cards_player_"+str(player)]
            item = self.findGraphicsCardItem_(active_player_cards[action])
            self.playCard(item, player, j, self.options["names"][player])
            j +=1
            del self.game_play["cards_player_"+str(player)][action]
            self.checkFinished()

    def clientReconnectTimer(self):
        self.openClient()

    def getHighlight(self, playeridx):
        try:
            if playeridx == self.my_game.active_player:
                return 1
            else:
                return 0
        except:
            return 0

    def setNames(self):
        self.changePlayerName(self.player1_label,  self.options["names"][0]+" ("+self.options["type"][0]+")", highlight=self.getHighlight(0))
        self.changePlayerName(self.player2_label,  self.options["names"][1]+" ("+self.options["type"][1]+")", highlight=self.getHighlight(1))
        self.changePlayerName(self.player3_label,  self.options["names"][2]+" ("+self.options["type"][2]+")", highlight=self.getHighlight(2))
        self.changePlayerName(self.player4_label,  self.options["names"][3]+" ("+self.options["type"][3]+")", highlight=self.getHighlight(3))

    def showResultClient(self, i, reward, total_reward, offhandCards):
        labels1 = [self.card1_label, self.card2_label, self.card3_label, self.card4_label]
        labels2 = [self.play_1_state, self.play_2_state, self.play_3_state, self.play_4_state]
        label1 = labels1[i]
        label2 = labels2[i]
        self.changePlayerName(label1,  self.options["names"][i]+" ("+self.options["type"][i]+")", highlight=0)
        self.changePlayerName(label2,  reward+" ["+total_reward+"]", highlight=0)
        self.deal_cards(offhandCards, i)
        self.view.viewport().repaint()


    def getNextPlayer(self, currPlayer):
        if currPlayer < len(self.options["names"])-1:
            return currPlayer+1
        else:
            return 0

    def getPreviousPlay(self, input_number, nuPlayers=4):
        if input_number == 0:
            prev_player = nuPlayers-1
        else:
            prev_player = input_number -1
        return prev_player

    def getInColorOfCards(self, cards):
        # returns the leading color of the on_table_cards
        # if only joker are played None is returned
        for i, card in enumerate(cards):
            if card is not None:
                if card.value <15:
                    return card.color
        return None

    def getWinnerForCards(self, cards, active_player, nu_players=4):
        # Not this function is used at the client side!
        highest_value    = 0
        winning_card     = cards[0]
        incolor          = self.getInColorOfCards(cards)
        on_table_win_idx = 0
        if  incolor is not None:
            for i, card in enumerate(cards):
                # Note 15 is a Jocker
                if card is not None and ( card.value > highest_value and card.color == incolor and card.value<15):
                    highest_value = card.value
                    winning_card = card
                    on_table_win_idx = i

        player_win_idx = active_player
        for i in range(nu_players-on_table_win_idx-1):
            player_win_idx = self.getPreviousPlay(player_win_idx, nuPlayers=nu_players)
        return winning_card, on_table_win_idx, player_win_idx


    def playCardClient(self, graphic_card_item, current_player, label_idx, player_name, shifting, shifted_cards):
        print( graphic_card_item, current_player, label_idx, player_name, shifting, shifted_cards)
        self.setNames()
        if shifting:
            card_label        =  self.card_label_l[graphic_card_item.player]
            self.changePlayerName(card_label, self.options["names"][graphic_card_item.player], highlight=0)
            self.view.viewport().repaint()
            shift_round = int(shifted_cards/4)
            graphic_card_item.setPos(card_label.pos().x(), card_label.pos().y()+20+shift_round*50)
        else:
            card_label        =  self.card_label_l[label_idx]
            self.changePlayerName(card_label,  self.options["names"][graphic_card_item.player], highlight=0)
            self.view.viewport().repaint()
            graphic_card_item = self.changeCard(graphic_card_item, faceDown=False)
            graphic_card_item.setPos(card_label.pos().x(), card_label.pos().y()+20)
        self.midCards.append(graphic_card_item)
        self.view.viewport().repaint()
        graphic_card_item.isPlayed = True

        if len(self.midCards) == 8:
            # remove all Client Cards (are dealt again!)
            print("Remove all cards now")
            self.removeAll()
            self.removeMidNames()
            self.midCards = []
            self.dealAgain = True
            self.changePlayerName(self.mode_label,  "Mode: Play")
        if len(self.midCards) == 4 and shifted_cards>10:
            print("Remove Mid Cards now")
            ttmp = self.midCards
            time.sleep(1)
            for i in self.midCards:
                self.removeCard(i)
            self.midCards = []
            self.removeMidNames()
            # TODO mark next player (may set trick winner etc.)
            lll = []
            for i in ttmp:
                lll.append(i.card)
            winning_card, on_table_win_idx, player_win_idx = self.getWinnerForCards(lll, int(current_player), nu_players=4)
            self.changePlayerName(self.card_label_pla[player_win_idx], self.options["names"][player_win_idx], highlight=1)
            self.changePlayerName(self.game_indicator,  "Game: "+str(self.games_played)+" Round: "+str(self.rounds_played))
            self.rounds_played +=1
            self.view.viewport().repaint()
        else:
            # mark next player:
            self.changePlayerName(self.card_label_pla[self.getNextPlayer(current_player)], self.options["names"][self.getNextPlayer(current_player)], highlight=1)
            self.view.viewport().repaint()

        # TODO Game indicator round etc.

        return 1 # card played!


    def getGraphicCard(self, label_idx, player_name, graphic_card_item):
        self.setNames()
        if self.my_game.shifting_phase:
            card_label        =  self.card_label_l[self.my_game.active_player]
            self.changePlayerName(card_label, player_name, highlight=0)
            self.view.viewport().repaint()
            time.sleep(self.options["sleepTime"])
            shift_round = int(self.my_game.shifted_cards/self.my_game.nu_players)
            graphic_card_item.setPos(card_label.pos().x(), card_label.pos().y()+20+shift_round*50)
        else:
            card_label        =  self.card_label_l[label_idx]
            self.changePlayerName(card_label, player_name, highlight=0)
            self.view.viewport().repaint()
            time.sleep(self.options["sleepTime"])
            graphic_card_item = self.changeCard(graphic_card_item, faceDown=False)
            graphic_card_item.setPos(card_label.pos().x(), card_label.pos().y()+20)
        return graphic_card_item

    def findGraphicsCardItem_(self, my_card):
        for i in self.getCardsList():
            try:
                if (i.card == my_card) or ((i.card.value==my_card.value) and (i.card.color == my_card.color)):
                    return i
            except:
                pass
        return None

    def findGraphicsCardItem(self, action_idx, player_idx):
        try:
            card_to_play = self.my_game.players[player_idx].hand[action_idx]
        except:
            print("Error no card left anymore!")
            return None
        for i in self.getCardsList():
            if i.card == card_to_play:
                return i

    def removeMidNames(self):
        self.card1_label.setPlainText("")
        self.card2_label.setPlainText("")
        self.card3_label.setPlainText("")
        self.card4_label.setPlainText("")

    def addPlayerLabel(self, x_pos, y_pos, name, highlight=0, font=QFont.Bold):
        item = self.scene.addText(name, QFont('Arial Black', 11, font))
        if highlight:
            item.setDefaultTextColor(Qt.yellow)
        item.setPos(x_pos, y_pos)
        return item

    def changePlayerName(self, text_item, name, highlight=0):
        text_item.setPlainText(name)
        if highlight:
            text_item.setDefaultTextColor(Qt.yellow)
        else:
            text_item.setDefaultTextColor(Qt.black)

    def mouseDoubleClickEvent(self, event):
        try:
            # check if item is a CardGraphicsItem
            p = event.pos()
            p -= QPoint(10, 10) #correction to mouse click. not sure why this happen
            itemAt = self.view.itemAt(p)
            if isinstance(itemAt, CardGraphicsItem):
                self.cardPressed(itemAt)
        except Exception as e:
            print(e)

    def checkCard(self, cardStr):
        for i in self.clientCards:
            if str(cardStr) in str(i):
                return True
        return False

    def cardPressed(self, card):
        if "Client" in self.options["online_type"]:
            # check if it is your card!
            if (self.checkCard(card.card)):
                self.wantPlay = str(card.card)
            else:
                print(self.clientName+" This is not your card!")
            if self.gotCards<=1:
                self.changePlayerName(self.mode_label,  "Mode: Shift "+str(self.wantPlay).replace("of",""))
            else:
                self.changePlayerName(self.mode_label,  "Mode: Play "+str(self.wantPlay).replace("of",""))
        else:
            print("other not allowed currently!!!")
            print(eeee)

    def setBackgroundColor(self, color):
        """ add background color """
        brush = QBrush(color)
        self.scene.setBackgroundBrush(brush)
        self.scene.backgroundBrush()


    def cardSvgFile(self, name):
        """ get card svg file from card name
        name = 'c_4','d_Q',...
        for jokers name = 'j_r' or 'j_b'
        for back name = 'back_1', 'back_2', ...
        """
        fn = os.path.join(self.svgCardsPath, name + ".svg") # TODO change to SVG
        return fn


    def addCard(self, my_card, player=0, faceDown=False):
        """ adds CardGraphicsItem graphics to board.
        also updates the total cards list
        """
        # svg file of the card graphics
        if faceDown:
            svgFile = self.cardSvgFile(self.deckBackSVG)
        else:
            svgFile = self.cardSvgFile(str(my_card.color)+str(my_card.value))

        # create CardGraphicsItem instance
        ind = len(self.getCardsList()) + 1
        tmp = CardGraphicsItem(my_card, ind, svgFile, player, faceDown)
        tmp.setScale(self.defScale)
        tmp.setZValue(ind) # set ZValue as index (last in is up)
#        self.cardsGraphItems.append(tmp)
        self.scene.addItem(tmp)
        # sanity check

        #print("num of cards=" + str(len(self.cardsList)))

    def removeAll(self):
        try:
            for i in (self.getCardsList()):
                self.scene.removeItem(i)
        except Exception as e:
            print("Exception:", e)


    def removeCard(self, card):
        """
        removes CardGraphicsItem graphics from board
        """
        self.scene.removeItem(card)

    # TODO - UPDATE THIS FUNCTION
    def changeCard(self, graphicsCardElement, faceDown=False):
        """ replace CardGraphicsItem
        keeps same index and ZValue !
        """
        nameToAdd = str(graphicsCardElement.card.color)+str(graphicsCardElement.card.value)
        zValueTmp = graphicsCardElement.zValue()
        position = graphicsCardElement.pos()
        angle = graphicsCardElement.rotation()
        scale = graphicsCardElement.scale()
        player = graphicsCardElement.player
        self.scene.removeItem(graphicsCardElement)

        # svg file of the card graphics
        if faceDown:
            svgFile = self.cardSvgFile(self.deckBackSVG)
        else:
            svgFile = self.cardSvgFile(nameToAdd)

        ind = int(zValueTmp)
        tmp = CardGraphicsItem(card(graphicsCardElement.card.color, graphicsCardElement.card.value), ind, svgFile, player, faceDown)
        tmp.setScale(self.defScale)
        tmp.setZValue(ind) # set ZValue as index (last in is up)

        self.scene.addItem(tmp)
        return tmp

    def getCardsList(self):
        """ returns and prints all CardGraphicsItem in scene (disregard other graphics items) """
        itemsOut=[]
        #print("Cards List:")
        for item in self.scene.items():
            if isinstance(item, CardGraphicsItem):
                itemsOut.append(item)
                #print("Ind=%3d | Name=%4s | Player=%d | faceDown=%r " % \
                #     (item.ind, item.name, item.player, item.faceDown) )
        #print("Total cards num = " + str(len(itemsOut)))
        return itemsOut

    def deal_cards(self, cards, playerNum, fdown=False):
        n=1
        c2=0
        dx = [0, self.defHandSpacing, 0, self.defHandSpacing]
        dy = [self.defHandSpacing, 0, self.defHandSpacing, 0]
        x, y, ang = self.playersHandsPos[playerNum-1]
        for card in cards:
            self.addCard(card, player=playerNum, faceDown=fdown)#add the item to the scene
            self.getCardsList()[0].setPos(x+dx[playerNum-1]*c2, y+dy[playerNum-1]*c2)#change the position
            n += 1
            c2 += 1

def main():
    # not used if gui.py is started!
    app = QApplication(sys.argv)
    form = cardTableWidget()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()
