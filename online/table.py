from __future__ import print_function
import sys
from PyQt5.QtWidgets import *
import os
from PyQt5.QtCore import *
from PyQt5.QtGui  import *
from PyQt5 import QtSvg
import time
import easygui
import json
from prettyjson import prettyjson

# Building an exe use onnx
import onnxruntime
import numpy as np

import pickle
from copy import deepcopy

# For server / client:
import socket
from PyQt5.QtCore import QDataStream, QIODevice, QByteArray, QCoreApplication, QEventLoop
from PyQt5.QtWidgets import QApplication, QDialog, QPushButton, QLineEdit, QLabel, QVBoxLayout
from PyQt5.QtNetwork import QTcpSocket, QAbstractSocket, QTcpServer, QHostAddress
import re
import ast
from gameClasses import card, deck, player, game

import urllib.request



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
    def __init__(self, card, ind, svgFile, player=0, faceDown=True):
        super(CardGraphicsItem, self).__init__(svgFile)
        # special properties
        self.card = card
        self.svgFile = svgFile # svg file for card graphics
        self.player = player # which player holds the card
        self.faceDown = faceDown # does the card faceDown
        self.anim = QPropertyAnimation() # will use to animate card movement
        #self.isPressed = False

        #default properties
        # IF CORE DUMPED uncomment following line"
        #self.setAcceptHoverEvents(True) #by Qt default it is set to False

    def hoverEnterEvent(self, event):
        """ event when mouse enter a card """
        try:
            effect = QGraphicsDropShadowEffect(self)
            effect.setBlurRadius(15)
            effect.setColor(Qt.red)
            effect.setOffset(QPointF(-5,0))
            self.setGraphicsEffect(effect)
        except:
            print("error")

    def __repr__(self):
        return '<CardGraphicsItem: %s>' % self.card


class cardTableWidget(QWidget):
    """ main widget for handling the card table """
    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)
        self.initUI()

    def initUI(self):
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
        self.svgCardsPath = "cards"
        self.cardsGraphItems = [] #holds all the cards items
        self.defInsertionPos = QPointF(0,0)
        self.defAngle = 0
        self.defScale = 0.5
        self.deckBackSVG = 'back_1'
        self.numOfPlayers = 4
        self.playersHandsPos = [(75, 50, 0), (210, 50, 180), (680, 50, 0), (210, 385, 0)] #(x,y,angle)
        self.defHandSpacing = 24
        self.midCards  = []
        self.options_file_path =  "gui_options.json"

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

        self.play_1_state  = self.addPlayerLabel(200, 250, "")
        self.play_2_state  = self.addPlayerLabel(200+120, 250, "")
        self.play_3_state  = self.addPlayerLabel(200+120*2, 250, "")
        self.play_4_state  = self.addPlayerLabel(200+120*3, 250, "")
        self.game_indicator= self.addPlayerLabel(650, 5, "Game: ")

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

        ### Server stuff:
        self.tcpServer        = None
        self.clientConnections= []
        self.blockSize        = 0
        self.server_state     = "INIT"
        self.timeouttt        = False
        self.timer            = QTimer(self)

        ### Client stuff:
        self.clientTimer      = QTimer(self)
        self.tcpSocket        = None
        self.clientCards      = None
        self.clientNames      = None
        self.clientType       = None
        self.clientId         = -1


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
        if self.my_game is not None:
            self.my_game.reset_game()
            self.runGame()
        else:
            print("Error, click start first!")
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

    def parseClientMessage(self, msg):
        print("Client received:", msg)
        command, message ="", ""
        try:
            command, message = msg.split(";")[0], msg.split(";")[1].replace(";", "")
        except Exception as e:
            print("Client cannot parse incoming message:", msg)
            return

        if "InitMyCards" in command:
            self.clientCards = self.convertCardsArray(message)
            self.deal_cards(self.clientCards, self.clientId, fdown=False)
        elif "InitOtherCards" in command:
            cardsStr, player = message.split("--")[0], message.split("--")[1]
            self.deal_cards(self.convertCardsArray(cardsStr), int(player), fdown=True)
        elif "Names" in command:
            self.clientNames =ast.literal_eval(message)
            for i,j in enumerate(self.clientNames):
                if self.options["names"][0] == j:
                    self.clientId =i
                    break
            self.options["names"] = self.clientNames
            #update board with client names
        elif "Type" in command:
            self.clientType  = ast.literal_eval(message)
            self.options["type"] = self.clientType
            self.setNames()
        elif "NOK" in command:
            print("Error: NOK", message)
        elif "PlayCard" in command:
            player_name, card = message.split(",")[0].replace(" ",""), self.convertCardString2Card(message.split(",")[1])
            shifting, nu_shift_cards, action, on_table_cards = message.split(",")[2], int(message.split(",")[3]), message.split(",")[4], int(message.split(",")[5])
            item = self.findGraphicsCardItem_(card)
            if "False" in shifting:
                shifting = False
            else:
                shfiting = True
            if shifting:
                card_played = self.playCardClient(item, self.clientId, on_table_cards, self.options["names"][self.clientId], shifting, nu_shift_cards)
            else:
                print("Not shifting play card client!!!")
                card_played = self.playCardClient(item, self.clientId, len(self.midCards), self.options["names"][self.clientId], shifting, nu_shift_cards)
            self.send_msgClient(self.options["names"][self.clientId]+";ClientPlayed;"+str(action))
        elif "DeleteCards" in command:
            # do this only once:
            self.removeAll()
        elif "ShiftMyCards" in command:
            self.clientCards = self.convertCardsArray(message)
            self.deal_cards(self.clientCards, self.clientId, fdown=False)
        elif "ShiftOtherCards" in command:
            cardsStr, player = message.split("--")[0], message.split("--")[1]
            self.deal_cards(self.convertCardsArray(cardsStr), int(player), fdown=True)
        elif "PutCard" in command:
            player_name, card = message.split(",")[0].replace(" ",""), self.convertCardString2Card(message.split(",")[1])
            shifting, nu_shift_cards, action, player, on_table_cards = bool(message.split(",")[2]), int(message.split(",")[3]), message.split(",")[4], message.split(",")[5], int(message.split(",")[6])
            if "False" in message.split(",")[2]:
                shifting = False
            item = self.findGraphicsCardItem_(card)
            if shifting:
                card_played = self.playCardClient(item, int(player), on_table_cards, self.options["names"][int(player)], bool(shifting), nu_shift_cards)
            else:
                card_played = self.playCardClient(item, int(player), len(self.midCards), self.options["names"][int(player)], bool(shifting), nu_shift_cards)
        elif "DeleteMid" in command:
            self.removeMidNames()
            for i in self.midCards:
                self.removeCard(i)
            self.midCards = []
        elif "ShowResult" in command:
            player, reward, total_reward, offhandCards  = int(message.split("--")[0]), message.split("--")[1], message.split("--")[2], self.convertCardsArray(message.split("--")[3])
            self.showResultClient(player, reward, total_reward, offhandCards)
        elif "BackHand" in command:
            print("BACK HAND:", message)
            self.deckBackSVG = message
        else:
            print("Sry I(Client) did not understand this command", command)
            return

    def send_msgClient(self, msg):
        self.tcpSocket.waitForConnected(1000)
        # TODO send with name in options[names][0]
        self.tcpSocket.write(bytes( str(msg), encoding='ascii'))

    def displayErrorClient(self, socketError):
        print("Client Error")
        if socketError == QAbstractSocket.RemoteHostClosedError:
            pass
        else:
            print("Server does not seem to be open or wrong open_ip!")
            if not self.clientTimer.isActive():
                print("The following error occurred: %s." % self.tcpSocket.errorString())
                self.clientTimer.timeout.connect(self.clientReconnectTimer)
                self.clientTimer.start(5000)


    def openClient(self, ip):
        self.tcpSocket = QTcpSocket(self)
        print("I client connect now with:", ip)
        self.tcpSocket.connectToHost(ip, 8000, QIODevice.ReadWrite)
        self.tcpSocket.readyRead.connect(self.dealCommunication)
        self.tcpSocket.error.connect(self.displayErrorClient)

        # send start message:
        connected = self.tcpSocket.waitForConnected(1000)
        if connected:
            self.clientTimer.stop()
            self.tcpSocket.write(bytes( self.options["names"][0]+";"+"InitClient;Server please init me with my name", encoding='ascii'))
        else:
            print("Not connected, Server not open?, open_ip wrong? Try to reconnect in 5sec")

    def dealCommunication(self):
        instr = QDataStream(self.tcpSocket)
        instr.setVersion(QDataStream.Qt_5_0)
        if self.blockSize == 0:
            if self.tcpSocket.bytesAvailable() < 2:
                return
            self.blockSize = instr.readUInt16()
        if self.tcpSocket.bytesAvailable() < self.blockSize:
            return
        # Print response to terminal, we could use it anywhere else we wanted.
        in_msg = str(instr.readString(), encoding='ascii')
        self.parseClientMessage(in_msg)
        self.blockSize = 0

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
    def serverInputCommunication(self):
        self.clientConnections.append({"conn":self.tcpServer.nextPendingConnection(), "idx": len(self.clientConnections)})
        self.clientConnections[len(self.clientConnections)-1]["conn"].readyRead.connect(self.receivedMessagesServer)

    def getNuClients(self):
        j,m = 0,0
        idx = []
        for i in self.options["type"]:
            if "Client" in i:
                idx.append(j)
                m +=1
            j+=1
        return m, idx

    def getCardIndex(self, card, card_list):
        for j,c in enumerate(card_list):
            if c.color == card.color and c.value == card.value:
                return j
        return -1

    def parseServerMessage(self, conn, msg):
        try:
            tmp = msg.split(";")
            print(tmp)
            client_name, command, message = tmp[0], tmp[1], tmp[2]
            if not "name" in conn and "InitClient" in command:
                conn["name"] = client_name
            if not "msgs" in conn:
                conn["msgs"] = []
        except:
            # cannot parse this message
            return

        conn["msgs"].append(message)

        if not (conn["name"] == client_name):
            print("wrong name", client_name, conn["name"])
            return

        if "WantPlay" in command:
            card = self.convertCardString2Card(message)
            action = self.getCardIndex(card, self.my_game.players[self.my_game.active_player].hand)
            if action == -1:
                self.send_msgServer(conn["idx"], "WantPlayNOK;"+str(card)+" does not belong to active player!")
                return
            is_allowed_list_idx = self.my_game.getValidOptions(self.my_game.active_player) #caution in shifting phase!
            incolor =self.my_game.getInColor()
            # Caution in shifting phase all is allowed!
            print(is_allowed_list_idx, incolor)
            if action not in is_allowed_list_idx and incolor is not None:
                self.send_msgServer(conn["idx"], "WantPlayNOK;"+"I cannot play"+str(card)+" not allowed!")
                return
            # send all that card is played!
            self.send_msgServer(-1, "PlayCard;"+self.options["names"][self.my_game.active_player]+","+str(card)+","+str(self.my_game.shifting_phase)+","+str(self.my_game.shifted_cards)+","+str(action)+","+str(len(self.my_game.on_table_cards)))
            item = self.findGraphicsCardItem(action, self.my_game.active_player)
            card_played = self.playCard(item, self.my_game.active_player, len(self.my_game.on_table_cards), self.my_game.names_player[self.my_game.active_player])
        elif "ClientPlayed" in command:
            rewards, round_finished = self.playVirtualCard(int(message))
            if len(self.my_game.players[self.my_game.active_player].hand)==0:
                self.checkFinished()
                self.showResult(rewards)
                return
            self.checkFinished()
            if "Server" in self.options["online_type"]:
                self.playUntilClient()
            else:
                #5. Play until human:
                self.playUntilHuman()
        elif "   " in command:
            print("hallo")
        else:
            print("not understood command:", command)

    def receivedMessagesServer(self):
        for conn in self.clientConnections:
            self.parseServerMessage(conn, str(conn["conn"].readAll(), encoding='ascii'))
        #check for state all_connected?
        tmp, _ = self.getNuClients()
        if len(self.clientConnections) == tmp:
            self.server_state =  "ALL_CONNECTED"
            #send signal

    def getConnOfIdx(self, idx):
        for conn in self.clientConnections:
            if conn["idx"] == idx:
                return conn
        return None

    def send_msgServer(self, idx, msg):
        'idx: connection idx player index'
        to =""
        if idx==-1:
            to = "all"
        else:
            conn =self.getConnOfIdx(idx)
            if conn is None:
                print("Sry this idx was not found. Current Connections:", self.clientConnections)
                return
        block = QByteArray()
        # QDataStream class provides serialization of binary data to a QIODevice
        out = QDataStream(block, QIODevice.ReadWrite)
        # We are using PyQt5 so set the QDataStream version accordingly.
        out.setVersion(QDataStream.Qt_5_0)
        out.writeUInt16(0)
        print(">>>>>Server sends:", msg)

        out.writeString(bytes(msg, encoding='ascii'))
        out.device().seek(0)
        out.writeUInt16(block.size() - 2)
        if to=="all":
            for connections in self.clientConnections:
                #actually should wait for emit signal!!!
                # see here: https://doc.qt.io/archives/4.6/qabstractsocket.html#waitForReadyRead
                connections["conn"].waitForReadyRead(100)
                connections["conn"].write(block)
        else:
            co = conn["conn"]
            co.waitForReadyRead(100)
            co.write(block)

    def getIP(self):
        hostname = socket.gethostname()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
        print(ip_address, hostname, "<<localhost")
        return ip_address, hostname

    def openServer(self):
        self.tcpServer = QTcpServer(self)
        PORT = 8000
        ip_, _ = self.getIP()
        address = QHostAddress(ip_) # e.g. use your server ip 192.144.178.26
        if not self.tcpServer.listen(address, PORT):
            print("cant listen!")
            self.close()
            return
        self.tcpServer.newConnection.connect(self.serverInputCommunication)
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
            self.deckBackSVG = self.options["back_CardColor"]
            if "Server" in self.options["online_type"]:
                if not "Server" in self.options["type"][0]:
                    print("Error type at position 0 must be Server in options. type:['Server', 'Client', etc.]", self.options["type"][0])
                    return
                self.my_game     = game(self.options)
                self.openServer()
                self.runGame()
            elif "Client" in self.options["online_type"]:
                valid_ip = self.is_valid_ipv4(self.options["open_ip"])
                if len(self.options["names"])>1 or len(self.options["type"])>1 or (not valid_ip) or ("Client" not in self.options["type"]):
                    print("Error use only one unique name in options.  names: ['YourName']")
                    print("Error use only one type in options.         type: ['Client']")
                    print("Error use only IPV4 as open_ip in options.  open_ip: 172.20.80.10")
                    return
                self.openClient(self.options["open_ip"])
                QCoreApplication.instance().processEvents(QEventLoop.WaitForMoreEvents)
            else:
                self.my_game     = game(self.options)
                self.runGame()

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

    def servertimeout(self):
        print("Wait for all players to be connected!", self.server_state, self.clientConnections)
        self.timeouttt = False

    def clientReconnectTimer(self):
        self.openClient(self.options["open_ip"])
        QCoreApplication.instance().processEvents(QEventLoop.WaitForMoreEvents)

    def runGame(self):
        # remove all cards which were there from last game.
        self.removeAll()
        if "Server" in self.options["online_type"] and self.my_game.nu_games_played<1:
            # get open ip:
            page = str(urllib.request.urlopen("http://checkip.dyndns.org/").read())
            print("THIS IS SERVER OPEN IP ADDRESS:",  re.search(r'.*?<body>(.*).*?</body>', page).group(1))

            # send cards to clients, wait for all clients
            # wait until all players are connected!
            print("Wait for all players to be connected!", self.server_state, self.clientConnections)
            while "ALL_CONNECTED" not in self.server_state:
                if not self.timeouttt:
                    self.timer.timeout.connect(self.servertimeout)
                    self.timer.start(5000)
                    self.timeouttt = True
                QCoreApplication.instance().processEvents(QEventLoop.WaitForMoreEvents)
            self.timer.stop()
            print("All clients are connected now:")
            print(self.clientConnections, "\n")
            #update names if all are connected.
            _, tmp = self.getNuClients()
            for i, conn in enumerate(self.clientConnections):
                self.options["names"][tmp[i]] = conn["name"]
            self.send_msgServer(-1, "BackHand;"+self.deckBackSVG)

        #3. Deal Cards:
        for i in range(len(self.my_game.players)):
            if "Server" in self.options["online_type"]:
                if "Server" in self.options["type"][i]:
                    self.deal_cards(self.my_game.players[i].hand, i, fdown=False)
                else:
                    self.deal_cards(self.my_game.players[i].hand, i, fdown=True)
            else:
                self.deal_cards(self.my_game.players[i].hand, i, fdown=self.options["faceDown"][i])

        if "Server" in self.options["online_type"]:
            if self.my_game.nu_games_played<1:
                self.send_msgServer(-1, "Names;"+str(self.options["names"]))
                self.send_msgServer(-1, "Type;"+str(self.options["type"]))

            # send all cards to all clients:
            self.send_msgServer(-1, "DeleteCards;"+"  ")
            for i in range(len(self.my_game.players)):
                for conn in self.clientConnections:
                    if conn["name"] == self.options["names"][i]:
                        self.send_msgServer(conn["idx"], "InitMyCards;"+str(self.my_game.players[i].hand))
                    else:
                        self.send_msgServer(conn["idx"], "InitOtherCards;"+str(self.my_game.players[i].hand)+"--"+str(i))
            print(">>>Please play first card as server!")

        # 4. Setup Names:
        self.setNames()
        self.changePlayerName(self.game_indicator,  "Game: "+str(self.my_game.nu_games_played+1))

        if "Server" in self.options["online_type"]:
            self.playUntilClient()
        else:
            #5. Play until human:
            self.playUntilHuman()

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

    def showResult(self, rewards):
        i = 0
        for f, b in zip([self.card1_label, self.card2_label, self.card3_label, self.card4_label], [self.play_1_state, self.play_2_state, self.play_3_state, self.play_4_state]):
            self.changePlayerName(f, self.my_game.names_player[i]+" ("+self.my_game.ai_player[i]+")", highlight=0)
            self.changePlayerName(b,  str(int(rewards["total_rewards"][i]))+" ["+str(int(self.my_game.total_rewards[i]))+"]", highlight=0)

            # print offhand cards:
            offhand_cards = [item for sublist in  self.my_game.players[i].offhand for item in sublist]
            self.deal_cards(offhand_cards, i)
            if "Server" in self.options["online_type"] and len(offhand_cards)>0:
                self.send_msgServer(-1, "ShowResult;"+str(i)+"--"+str(int(rewards["total_rewards"][i]))+"--"+str(int(self.my_game.total_rewards[i]))+"--"+str(offhand_cards))
            i +=1
        self.view.viewport().repaint()
        time.sleep(self.options["sleepTime"]*50)
        self.changePlayerName(self.game_indicator,  "Game: "+str(self.my_game.nu_games_played+1))
        self.my_game.current_round = 0
        if self.options["nu_games"] > self.my_game.nu_games_played+1:
            self.nextRound_clicked()

    def to_numpy(self, tensor):
        # used in test_onnx
        return tensor.detach().cpu().numpy()

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

    def test_onnx(self, x, path):
        #print("path:", path)
        ort_session = onnxruntime.InferenceSession(path)

        # compute ONNX Runtime output prediction
        #print(type(x[0]), x[0])
        #print(type(np.asarray(x[0])), np.asarray(x[0]))
        #print("I will now test your model!")
        ort_inputs = {ort_session.get_inputs()[0].name: np.asarray(x[0], dtype=np.float32)}
        ort_outs = ort_session.run(None, ort_inputs)
        #print(ort_outs)
        max_value = (np.amax(ort_outs))
        #print(max_value)
        result = np.where(ort_outs == np.amax(ort_outs))
        #print(result)

        #TODO sort after indices?!
        return result[1][0]

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
            action = self.rl_onnx(state_240, state_303, "data/"+self.options["onnx_rl_path"][rl_type]+".onnx")
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

    def playVirtualCard(self, action):
        current_player = deepcopy(self.my_game.active_player)
        if self.options["save_game_play"] and len(self.my_game.played_cards) == 0:
            self.game_play = {}
            self.game_play["moves"] = []
            for i, player in enumerate(self.my_game.players):
                self.game_play["cards_player_"+str(i)] = deepcopy(player.hand)
        rewards, round_finished, gameOver = self.my_game.step_idx_with_shift(action)
        if round_finished and "Server" in self.options["online_type"] and not self.my_game.shifting_phase:
            self.send_msgServer(-1, "DeleteMid;  ")
        if self.options["save_game_play"]:
            self.game_play["moves"].append([current_player, action])
            if len(self.my_game.played_cards) == 60:
                with open(self.options["game_play_path"], 'wb') as f:
                    pickle.dump(self.game_play, f)
        return rewards, round_finished

    def playUntilClient(self):
        while (not "Client" in self.my_game.ai_player[self.my_game.active_player]) and (not "Server" in self.my_game.ai_player[self.my_game.active_player]):
            action = self.selectAction()
            item = self.findGraphicsCardItem(action, self.my_game.active_player)
            self.playCard(item, self.my_game.active_player, len(self.my_game.on_table_cards), self.my_game.names_player[self.my_game.active_player])
            if "Server" in self.options["online_type"]:
                self.send_msgServer(-1, "PutCard;"+self.options["names"][self.my_game.active_player]+","+str(item.card)+","+str(self.my_game.shifting_phase)+","+str(self.my_game.shifted_cards)+","+str(action)+","+str(self.my_game.active_player)+","+str(len(self.my_game.on_table_cards)))
            rewards, round_finished = self.playVirtualCard(action)
            if len(self.my_game.players[self.my_game.active_player].hand)==0:
                self.checkFinished()
                self.showResult(rewards)
                return
            self.setNames()
            self.checkFinished()
            self.changePlayerName(self.game_indicator,  "Game: "+str(self.my_game.nu_games_played+1)+" Round: "+str(self.my_game.current_round+1))

    def playUntilHuman(self):
        print("inside playUntil Human")
        while not "HUMAN" in self.my_game.ai_player[self.my_game.active_player]:
            action = self.selectAction()
            item = self.findGraphicsCardItem(action, self.my_game.active_player)
            self.playCard(item, self.my_game.active_player, len(self.my_game.on_table_cards), self.my_game.names_player[self.my_game.active_player])
            rewards, round_finished = self.playVirtualCard(action)
            if len(self.my_game.players[self.my_game.active_player].hand)==0:
                self.checkFinished()
                self.showResult(rewards)
                return
            self.setNames()
            self.checkFinished()
            self.changePlayerName(self.game_indicator,  "Game: "+str(self.my_game.nu_games_played+1)+" Round: "+str(self.my_game.current_round+1))

    def playCardClient(self, graphic_card_item, current_player, label_idx, player_name, shifting, shifted_cards):
        if graphic_card_item.player == current_player:
            self.setNames()
            if shifting:
                card_label        =  self.card_label_l[current_player]
                self.changePlayerName(card_label, player_name, highlight=0)
                self.view.viewport().repaint()
                shift_round = int(shifted_cards/4)
                graphic_card_item.setPos(card_label.pos().x(), card_label.pos().y()+20+shift_round*50)
            else:
                card_label        =  self.card_label_l[label_idx]
                self.changePlayerName(card_label, player_name, highlight=0)
                self.view.viewport().repaint()
                graphic_card_item = self.changeCard(graphic_card_item, faceDown=False)
                graphic_card_item.setPos(card_label.pos().x(), card_label.pos().y()+20)
            self.midCards.append(graphic_card_item)
            self.view.viewport().repaint()
            return 1 # card played!
        else:
            print("ERROR I cannot play card", graphic_card_item, "it belongs player", graphic_card_item.player, "current player is", current_player)
            return 0

    def playCard(self, graphic_card_item, current_player, label_idx, player_name):
        try:
            if graphic_card_item.player == current_player:
                graphic_card_item = self.getGraphicCard(label_idx, player_name, graphic_card_item)
                self.midCards.append(graphic_card_item)
                self.view.viewport().repaint()
                return 1 # card played!
            else:
                print("ERROR I cannot play card", graphic_card_item, "it belongs player", graphic_card_item.player, "current player is", current_player)
                return 0
        except Exception as ee:
            print(ee)

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
            except Exception as e:
                print(e, my_card, i.card)

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

    def checkFinished(self):
        if self.my_game.shifting_phase:
            return
        if len(self.midCards)==8:
            # deal cards again (with hand cards of shifting phase)
            self.removeAll()
            for i in range(len(self.my_game.players)):
                if "Server" in self.options["online_type"]:
                    if "Server" in self.options["type"][i]:
                        self.deal_cards(self.my_game.players[i].hand, i, fdown=False)
                    else:
                        self.deal_cards(self.my_game.players[i].hand, i, fdown=True)
                else:
                    self.deal_cards(self.my_game.players[i].hand, i, fdown=self.options["faceDown"][i])
            self.midCards = []

            if "Server" in self.options["online_type"]:
                # send dealt cards to clients
                self.send_msgServer(-1, "DeleteCards;"+"   ")
                for i in range(len(self.my_game.players)):
                    for conn in self.clientConnections:
                        if conn["name"] == self.options["names"][i]:
                            self.send_msgServer(conn["idx"], "ShiftMyCards;"+str(self.my_game.players[i].hand))
                        else:
                            self.send_msgServer(conn["idx"], "ShiftOtherCards;"+str(self.my_game.players[i].hand)+"--"+str(i))

        if len(self.midCards)>=4:
            time.sleep(self.options["sleepTime"])
            for i in self.midCards:
                self.removeCard(i)
            self.midCards = []
            self.removeMidNames()

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

    def cardPressed(self, card):
        if "Client" in self.options["online_type"]:
            print("I client want to play now:", card)
            self.send_msgClient(self.options["names"][self.clientId]+";WantPlay;"+str(card.card))
        else:
            if "Client" in self.my_game.ai_player[self.my_game.active_player]:
                print("We wait for client", self.my_game.players[self.my_game.active_player], "currently.... you as Server cannot play his card!")
                return
            try:
                action = (self.my_game.players[self.my_game.active_player].hand.index(card.card))
            except:
                print("Cannot get action. Card does not belong to this player!")
                return
            is_allowed_list_idx = self.my_game.getValidOptions(self.my_game.active_player)
            incolor =self.my_game.getInColor()
            print(is_allowed_list_idx, incolor)
            if action not in is_allowed_list_idx and incolor is not None:
                print("I cannot play", card, " not allowed!")
                return
            card_played = self.playCard(card, self.my_game.active_player, len(self.my_game.on_table_cards), self.my_game.names_player[self.my_game.active_player])
            if "Server" in self.options["online_type"]:
                self.send_msgServer(-1, "PutCard;"+self.options["names"][self.my_game.active_player]+","+str(card.card)+","+str(self.my_game.shifting_phase)+","+str(self.my_game.shifted_cards)+","+str(action)+","+str(self.my_game.active_player)+","+str(len(self.my_game.on_table_cards)))
            if card_played:
                rewards, round_finished = self.playVirtualCard(action)
                print(rewards, round_finished)
                if len(self.my_game.players[self.my_game.active_player].hand)==0:
                    self.checkFinished()
                    self.showResult(rewards)
                    return
                self.checkFinished()
                #print("Active Player", self.my_game.active_player)
                #print("Human Card Played: ", card)
                if "Server" in self.options["online_type"]:
                    self.playUntilClient()
                else:
                    #5. Play until human:
                    self.playUntilHuman()


    def getCenterPoint(self)        :
        """ finds screen center point """
        rect = self.view.geometry()
        return QPointF(rect.width()/2,rect.height()/2)


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


    def addCard(self, card, player=0, faceDown=False):
        """ adds CardGraphicsItem graphics to board.
        also updates the total cards list
        """
        # svg file of the card graphics
        if faceDown:
            svgFile = self.cardSvgFile(self.deckBackSVG)
        else:
            svgFile = self.cardSvgFile(str(card.color)+str(card.value))

        # create CardGraphicsItem instance
        ind = len(self.getCardsList()) + 1
        tmp = CardGraphicsItem(card, ind, svgFile, player, faceDown)
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
        tmp = CardGraphicsItem(card, ind, svgFile, player, faceDown)
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
