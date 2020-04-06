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
from modules.prettyjson import prettyjson

from modules.gameClasses import card, deck, player, game

#For NN:
# (Optional for testing)
# from train import test_trained_model

# Building an exe use onnx
import onnxruntime
import numpy as np

#For MCTS:
from modules.VanilaMCTS import VanilaMCTS
from modules.stdout import enable, disable  # for silent print
import pickle
from copy import deepcopy

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

    # def mousePressEvent(self, event):
    #     print("mouse Press Event!!!", event)
    #     p = event.pos()
    #     p -= QPoint(10, 10) #correction to mouse click. not sure why this happen
    #     print(p)
    #     self.isPressed = True

    # def hoverLeaveEvent(self, event):
    #     """ event when mouse leave a card """
    #     try:
    #         effect = QGraphicsDropShadowEffect(self)
    #         effect.setBlurRadius(150)
    #         effect.setColor(Qt.red)
    #         effect.setOffset(QPointF(-5,0))
    #         self.setGraphicsEffect(effect)
    #     except:
    #         print("error")

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
        self.options_file_path =  "data/gui_options.json"

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

        nextRound = QPushButton('nextRound', self)
        nextRound.resize(80, 32)
        nextRound.move(65, 10)
        nextRound.clicked.connect(self.nextRound_clicked)

        options = QPushButton('Options', self)
        options.resize(80, 32)
        options.move(150, 10)
        options.clicked.connect(self.options_clicked)

        self.scene.addWidget(playbtn)
        self.scene.addWidget(nextRound)
        self.scene.addWidget(options)

        self.my_game = None

        # Testing tree:
        self.my_tree = None

        # Storing game_play
        self.game_play = {}

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


    def runGame(self):
        # remove all cards which were there from last game.
        self.removeAll()

        #3. Deal Cards:
        for i in range(len(self.my_game.players)):
            self.deal_cards(self.my_game.players[i].hand, i, fdown=self.options["faceDown"][i])

        # 4. Setup Names:
        self.setNames()
        self.changePlayerName(self.game_indicator,  "Game: "+str(self.my_game.nu_games_played+1))

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

    def showResult(self, rewards):
        i = 0
        for f, b in zip([self.card1_label, self.card2_label, self.card3_label, self.card4_label], [self.play_1_state, self.play_2_state, self.play_3_state, self.play_4_state]):
            self.changePlayerName(f, self.my_game.names_player[i]+" ("+self.my_game.ai_player[i]+")", highlight=0)
            self.changePlayerName(b,  str(int(rewards[i]))+" ["+str(int(self.my_game.total_rewards[i]))+"]", highlight=0)

            # print offhand cards:
            offhand_cards = [item for sublist in  self.my_game.players[i].offhand for item in sublist]
            self.deal_cards(offhand_cards, i)
            i +=1
        self.view.viewport().repaint()
        time.sleep(self.options["sleepTime"]*20)
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
        rewards, round_finished = self.my_game.step_idx_with_shift(action)
        if self.options["save_game_play"]:
            self.game_play["moves"].append([current_player, action])
            if len(self.my_game.played_cards) == 60:
                with open(self.options["game_play_path"], 'wb') as f:
                    pickle.dump(self.game_play, f)
        return rewards, round_finished

    def playUntilHuman(self):
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

    def playCard(self, graphic_card_item, current_player, label_idx, player_name):
        if graphic_card_item.player == current_player:
            graphic_card_item = self.getGraphicCard(label_idx, player_name, graphic_card_item)
            self.midCards.append(graphic_card_item)
            self.view.viewport().repaint()
            return 1 # card played!
        else:
            print("ERROR I cannot play card", graphic_card_item, "it belongs player", graphic_card_item.player, "current player is", current_player)
            return 0

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
            if i.card == my_card:
                return i

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
                self.deal_cards(self.my_game.players[i].hand, i, fdown=self.options["faceDown"][i])
            self.midCards = []
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
        # print("All items at pos: ", end="")
        # print(self.view.items(p))
        # print("view.mapToScene: ",end="")
        # print(self.view.mapToScene(p))

    # def mousePressEvent(self, event):
    #     #overriding mousemoveevent SEGFAULTS - Qt Centre Forum
    #     # pure virtual method called
    #     # terminate called without an active exception
    #     # Aborted (core dumped)

    def cardPressed(self, card):
        try:
            action = (self.my_game.players[self.my_game.active_player].hand.index(card.card))
        except:
            print("Cannot get action. Card does not belong to this player!")
            return
        is_allowed_list_idx = self.my_game.getValidOptions(self.my_game.active_player)
        incolor =self.my_game.getInColor()
        if action not in is_allowed_list_idx and incolor is not None:
            print("I cannot play", card, " not allowed!")
            return
        card_played = self.playCard(card, self.my_game.active_player, len(self.my_game.on_table_cards), self.my_game.names_player[self.my_game.active_player])
        if card_played:
            rewards, round_finished = self.playVirtualCard(action)
            if len(self.my_game.players[self.my_game.active_player].hand)==0:
                self.checkFinished()
                self.showResult(rewards)
                return
            self.checkFinished()
            #print("Active Player", self.my_game.active_player)
            #print("Human Card Played: ", card)
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
