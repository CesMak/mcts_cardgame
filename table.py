from __future__ import print_function
import sys
from PyQt5.QtWidgets import *
import os
from PyQt5.QtCore import *
from PyQt5.QtGui  import *
from PyQt5 import QtSvg
import time

from gameClasses import card, deck, player, game

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

        #default properties
        self.setAcceptHoverEvents(True) #by Qt default it is set to False

    def hoverEnterEvent(self, event):
        """ event when mouse enter a card """
        effect = QGraphicsDropShadowEffect(self)
        effect.setBlurRadius(15)
        effect.setColor(Qt.red)
        effect.setOffset(QPointF(-5,0))
        self.setGraphicsEffect(effect)


    def hoverLeaveEvent(self, event):
        """ event when mouse leave a card """
        self.setGraphicsEffect(None)

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

        self.play_1_state  = self.addPlayerLabel(200, 250, "")
        self.play_2_state  = self.addPlayerLabel(200+120, 250, "")
        self.play_3_state  = self.addPlayerLabel(200+120*2, 250, "")
        self.play_4_state  = self.addPlayerLabel(200+120*3, 250, "")

        playbtn = QPushButton('Start', self)
        playbtn.resize(50, 32)
        playbtn.move(10, 10)
        playbtn.clicked.connect(self.start_clicked)

        nextRound = QPushButton('nextRound', self)
        nextRound.resize(80, 32)
        nextRound.move(65, 10)
        #nextRound.clicked.connect(self.start_clicked)

        options = QPushButton('Options', self)
        options.resize(80, 32)
        options.move(150, 10)
        #options.clicked.connect(self.start_clicked)

        self.scene.addWidget(playbtn)
        self.scene.addWidget(nextRound)
        self.scene.addWidget(options)

        self.my_game = None

    def start_clicked(self):
        #1. Load Options
        # options for class game and for this gui handling (e.g. faceDown)
        self.options = {}
        self.options["names"] = ["Tim", "Bob", "Frank", "Lea"]
        self.options["type"]  = ["NN", "NN", "NN", "NN"]
        self.options["expo"]  = [600, 600, 600, 600]
        self.options["depths"]= [300, 300, 300, 300]
        self.options["itera"]     = [100, 100, 100, 100]
        self.options["faceDown"]  = [True, True, True, False]
        self.options["sleepTime"] = 0.1

        # remove all cards which were there from last game.
        self.removeAll()

        #2. Create Game:
        self.my_game     = game(self.options)

        #3. Deal Cards:
        for i in range(len(self.my_game.players)):
            #TODO give them wrong sided!
            self.deal_cards(self.my_game.players[i].hand, i, fdown=self.options["faceDown"][i])

        # 4. Setup Names:
        self.setNames()

        #5. Play until human:
        self.playUntilHuman()

    def getHighlight(self, playeridx):
        if playeridx == self.my_game.active_player:
            return 1
        return 0

    def setNames(self):
        self.changePlayerName(self.player1_label,  self.my_game.names_player[0], highlight=self.getHighlight(0))
        self.changePlayerName(self.player2_label,  self.my_game.names_player[1], highlight=self.getHighlight(1))
        self.changePlayerName(self.player3_label,  self.my_game.names_player[2], highlight=self.getHighlight(2))
        self.changePlayerName(self.player4_label,  self.my_game.names_player[3], highlight=self.getHighlight(3))

    def showResult(self, rewards):
        i = 0
        for f, b in zip([self.card1_label, self.card2_label, self.card3_label, self.card4_label], [self.play_1_state, self.play_2_state, self.play_3_state, self.play_4_state]):
            self.changePlayerName(f, self.my_game.names_player[i], highlight=0)
            self.changePlayerName(b,  str(int(rewards[i])), highlight=0)

            # print offhand cards:
            offhand_cards = [item for sublist in  self.my_game.players[i].offhand for item in sublist]
            self.deal_cards(offhand_cards, i)
            i +=1

    def playUntilHuman(self):
        while not "Human" in self.my_game.ai_player[self.my_game.active_player]:
            action = self.my_game.getRandomOption_()
            item = self.findGraphicsCardItem(action, self.my_game.active_player)
            self.playCard(item, self.my_game.active_player, len(self.my_game.on_table_cards), self.my_game.names_player[self.my_game.active_player])
            rewards, round_finished = self.my_game.step_idx(action, auto_shift=False)
            if rewards is not None:
                self.checkFinished()
                self.showResult(rewards)
                return
            self.setNames()
            self.checkFinished()

    def playCard(self, graphic_card_item, current_player, cards_on_table, player_name):
        if graphic_card_item.player == current_player:
            if cards_on_table == 0:
                self.setNames()
                self.changePlayerName(self.card1_label, player_name, highlight=0)
                self.view.viewport().repaint()
                time.sleep(self.options["sleepTime"])
                graphic_card_item = self.changeCard(graphic_card_item, faceDown=False)
                graphic_card_item.setPos(self.card1_label.pos().x(), self.card1_label.pos().y()+20)
            elif cards_on_table == 1:
                self.setNames()
                self.changePlayerName(self.card2_label, player_name, highlight=0)
                self.view.viewport().repaint()
                time.sleep(self.options["sleepTime"])
                graphic_card_item = self.changeCard(graphic_card_item, faceDown=False)
                graphic_card_item.setPos(self.card2_label.pos().x(), self.card2_label.pos().y()+20)
            elif cards_on_table == 2:
                self.setNames()
                self.changePlayerName(self.card3_label, player_name, highlight=0)
                self.view.viewport().repaint()
                time.sleep(self.options["sleepTime"])
                graphic_card_item = self.changeCard(graphic_card_item, faceDown=False)
                graphic_card_item.setPos(self.card3_label.pos().x(), self.card3_label.pos().y()+20)
            elif cards_on_table == 3:
                self.setNames()
                self.changePlayerName(self.card4_label, player_name, highlight=0)
                self.view.viewport().repaint()
                time.sleep(self.options["sleepTime"])
                graphic_card_item = self.changeCard(graphic_card_item, faceDown=False)
                graphic_card_item.setPos(self.card4_label.pos().x(), self.card4_label.pos().y()+20)
            self.midCards.append(graphic_card_item)
            self.view.viewport().repaint()
            return 1 # card played!
        else:
            print("ERROR I cannot play card", graphic_card_item, "it belongs player", graphic_card_item.player, "current player is", current_player)
            return 0

    def findGraphicsCardItem_(self, my_card):
        for i in self.getCardsList():
            if i.card == my_card:
                return i
    def findGraphicsCardItem(self, action_idx, player_idx):
        try:
            card_to_play = self.my_game.players[player_idx].hand[action_idx]
        except:
            print("Error no card left anymore!")
        for i in self.getCardsList():
            if i.card == card_to_play:
                return i

    def removeMidNames(self):
        self.card1_label.setPlainText("")
        self.card2_label.setPlainText("")
        self.card3_label.setPlainText("")
        self.card4_label.setPlainText("")

    def checkFinished(self):
        if len(self.midCards)==4:
            time.sleep(self.options["sleepTime"])
            for i in self.midCards: self.removeCard(i)
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

    def mousePressEvent(self, event):
        # check if item is a CardGraphicsItem
        p = event.pos()
        p -= QPoint(10, 10) #correction to mouse click. not sure why this happen
        itemAt = self.view.itemAt(p)
        if isinstance(itemAt, CardGraphicsItem):
            self.cardPressed(itemAt)

        # print("All items at pos: ", end="")
        # print(self.view.items(p))
        # print("view.mapToScene: ",end="")
        # print(self.view.mapToScene(p))

    def cardPressed(self, card):
        try:
            action = (self.my_game.players[self.my_game.active_player].hand.index(card.card))
        except:
            print("Cannot get action. Card does not belong to this player!")
            return
        is_allowed_list_idx = self.my_game.getValidOptions(self.my_game.active_player)
        if action not in is_allowed_list_idx:
            print("I cannot play", card, " not allowed!")
            return
        card_played = self.playCard(card, self.my_game.active_player, len(self.my_game.on_table_cards), self.my_game.names_player[self.my_game.active_player])
        if card_played:
            print("Active Player", self.my_game.active_player)
            print("before round finished!")
            rewards, round_finished = self.my_game.step_idx(action, auto_shift=False)
            self.checkFinished()
            print("Active Player", self.my_game.active_player)
            print("Human Card Played: ", card)
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
        for i in self.getCardsList():
            self.scene.removeItem(i)

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
            self.addCard(card, player=playerNum, faceDown=fdown)
            self.getCardsList()[0].setPos(x+dx[playerNum-1]*c2, y+dy[playerNum-1]*c2)
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
