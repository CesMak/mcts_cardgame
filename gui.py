# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui  import *
import random
from modules.table import cardTableWidget

#Links
# https://github.com/eladj/QtCards

class CardTableWidgetExtend(cardTableWidget):
    """ extension of CardTableWidget """
    def __init__(self):
        super(CardTableWidgetExtend, self).__init__()

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.cardsTable = CardTableWidgetExtend()

        # main layout
        self.mainLayout = QVBoxLayout()

        # add all widgets to the main vLayout
        #self.mainLayout.addWidget(self.label1)
        self.mainLayout.addWidget(self.cardsTable)

        # central widget
        self.centralWidget = QWidget()
        self.centralWidget.setLayout(self.mainLayout)
#
#       # set central widget
        self.setCentralWidget(self.centralWidget)

        #self.cardsTable.addCard('c_K')

        #self.cardsTable.addCard('d_8')
        #self.cardsTable.addCard('j_r')
        #self.cardsTable.changeCard(1,'h_J', faceDown=True) # -> does not work!

        #for i in range (1, 20):
        #    self.cardsTable.removeCard(i)

        # PLAYGROUND
        #self.cardsTable.dealDeck()
        # self.cardsTable.getCardsList()[0].setPos(200, 230)
        # self.cardsTable.getCardsList()[1].setPos(200+120, 230)
        # self.cardsTable.getCardsList()[2].setPos(200+120*2, 230)
        # self.cardsTable.getCardsList()[3].setPos(200+120*3, 230)
        #self.cardsTable.getCardsList()

def create_widget(use_gui=1):
    app    = QApplication(sys.argv)
    widget = MainWindow()
    widget.setWindowTitle("Wichtes AI - developed by Markus Lamprecht @2020")
    widget.setWindowIcon(QIcon('cards/icon.png'))
    widget.show()
    sys.exit(app.exec_()) # blocks whole programm!

if __name__ == "__main__":
    create_widget()
