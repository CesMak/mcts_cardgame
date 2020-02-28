from __future__ import print_function
import sys
from PyQt5.QtWidgets import *
import os
from PyQt5.QtCore import *
from PyQt5.QtGui  import *
from PyQt5 import QtSvg
import random
import re


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
    def __init__(self, name, ind, svgFile, player=0, faceDown=True):
        super(CardGraphicsItem, self).__init__(svgFile)
        # special properties
        self.name = name
        self.ind = ind # index
        self.svgFile = svgFile # svg file for card graphics
        self.player = player # which player holds the card
        self.faceDown = faceDown # does the card faceDown
        self.anim = QPropertyAnimation() # will use to animate card movement

        #default properties
        self.setAcceptHoverEvents(True) #by Qt default it is set to False


    def getSuit(self):
        """ get card suit type """
        return int(re.findall(r'\d+', self.name)[0])


    def getRank(self):
        """ get card rank type """
        return  "".join(re.split("[^a-zA-Z]*", self.name))

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
        return '<CardGraphicsItem: %s>' % self.name


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
        self.playersHandsPos = [(75, 50, 0), (190, 50, 180), (680, 25, 0), (190, 385, 0)] #(x,y,angle)
        self.defHandSpacing = 24

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
        self.player3_label = self.addPlayerLabel(782, 240, "Player 4")

        #Highlight card:
        self.changePlayerName(self.player1_label,"Max", highlight=1)

    def addPlayerLabel(self, x_pos, y_pos, name, highlight=0):
        item = self.scene.addText(name, QFont('Arial Black', 11, QFont.Bold))
        if highlight:
            item.setDefaultTextColor(Qt.yellow)
        item.setPos(x_pos, y_pos)
        return item

    def changePlayerName(self, text_item, name, highlight=0):
        text_item.setPlainText(name)
        if highlight:
            text_item.setDefaultTextColor(Qt.yellow)

    def mousePressEvent(self, event):
        # check if item is a CardGraphicsItem
        p = event.pos()
        print(p)
        p -= QPoint(10,10) #correction to mouse click. not sure why this happen
        itemAt = self.view.itemAt(p)
        if isinstance(itemAt, CardGraphicsItem):
            self.cardPressed(itemAt)
        print(p)
        #print("mapFromScene",end="")
        #print(self.view.mapFromScene(event.pos()))
        print("All items at pos: ", end="")
        print(self.view.items(p))
        print("view.mapToScene: ",end="")
        print(self.view.mapToScene(p))

    def cardPressed(self, card, animate=True):
        if animate:
            card.anim.setDuration(150)
            #anim.setStartValue(self.pos())
            card.anim.setEndValue(self.getCenterPoint())
            card.anim.start()
        else:
            card.setPos(self.getCenterPoint())
        print("Card Played: " + card.name)


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


    def addCard(self, name, player=0, faceDown=False):
        """ adds CardGraphicsItem graphics to board.
        also updates the total cards list
        """
        # svg file of the card graphics
        if faceDown:
            svgFile = self.cardSvgFile(self.deckBackSVG)
        else:
            svgFile = self.cardSvgFile(name)

        # create CardGraphicsItem instance
        ind = len(self.getCardsList()) + 1
        tmp = CardGraphicsItem(name, ind, svgFile, player, faceDown)
        tmp.setScale(self.defScale)
        tmp.setZValue(ind) # set ZValue as index (last in is up)
#        self.cardsGraphItems.append(tmp)
        self.scene.addItem(tmp)
        # sanity check

        #print("num of cards=" + str(len(self.cardsList)))


    def removeCard(self, card):
        """ removes CardGraphicsItem graphics from board
        also removes from the total cards list
        """
        if isinstance(card,int):
            allCards = self.getCardsList()
            indices = [c.ind for c in allCards]
            ind = indices.index(card)
            self.scene.removeItem(allCards[ind])
        if isinstance(card,CardGraphicsItem):
            self.scene.removeItem(card)


    # TODO - UPDATE THIS FUNCTION
    def changeCard(self, cardIndRemove, nameToAdd, faceDown=False):
        """ replace CardGraphicsItem
        keeps same index and ZValue !
        """
        zValueTmp = self.cardsGraphItems[cardIndRemove].zValue()
        position = self.cardsGraphItems[cardIndRemove].pos()
        angle = self.cardsGraphItems[cardIndRemove].rotation()
        scale = self.cardsGraphItems[cardIndRemove].scale()
        self.scene.removeItem(self.cardsGraphItems[cardIndRemove])
        self.cardsGraphItems.pop(cardIndRemove)
        player = self.cardsList[cardIndRemove].player
        self.cardsList.pop(cardIndRemove)

        # svg file of the card graphics
        if faceDown:
            svgFile = self.cardSvgFile(self.deckBackSVG)
        else:
            svgFile = self.cardSvgFile(nameToAdd)

        ind = cardIndRemove
        self.cardsList.insert(ind,CardItem(nameToAdd,self.value(nameToAdd),player,faceDown))
        tmp = CardGraphicsItem(nameToAdd, ind, position, svgFile, angle, scale)
        tmp.setZValue(zValueTmp) # set ZValue as previous
        self.cardsGraphItems.insert(ind, tmp)
        self.scene.addItem(self.cardsGraphItems[ind])
        self.checkLists()


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

    def buildDeckList(self):
        suits = ['B','G','R','Y']
        ranks = [str(i) for i in range (1, 16)]
        l = list()
        for suit in suits:
            for rank in ranks:
                l.append(suit + '' + rank)
        return l

    def dealDeck(self):
        d = self.buildDeckList()
        random.shuffle(d)
        #print(d)
        playerNum=1
        n=1
        c2=0
        dx = [0,self.defHandSpacing,0,self.defHandSpacing]
        dy = [self.defHandSpacing,0,self.defHandSpacing,0]
        x, y, ang = self.playersHandsPos[playerNum-1]
        for card in d:
            self.addCard(card,player=playerNum)
            self.getCardsList()[0].setPos(x+dx[playerNum-1]*c2,
                                           y+dy[playerNum-1]*c2)
            self.getCardsList()[0]#.rotate(ang)

            if n % (60 / self.numOfPlayers) == 0:
                playerNum += 1
                if playerNum > self.numOfPlayers:
                    break
                x, y, ang = self.playersHandsPos[playerNum-1]
                c2=0
            n += 1
            c2 += 1

def main():
    app = QApplication(sys.argv)
    form = cardTableWidget()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()
