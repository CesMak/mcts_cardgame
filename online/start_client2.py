# -*- coding: utf-8 -*-
import sys
from PyQt5.QtGui  import QIcon
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtWidgets import QMainWindow, QGraphicsView, QWidget, QGraphicsScene, QGridLayout, QApplication, QVBoxLayout
from table import cardTableWidget

#Links
# See also https://github.com/eladj/QtCards

class CardTableWidgetExtend(cardTableWidget):
    """ extension of CardTableWidget """
    def __init__(self):
        super(CardTableWidgetExtend, self).__init__()

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.cardsTable = CardTableWidgetExtend()
        self.cardsTable.initUI("client_options2.json")

        # main layout
        self.mainLayout = QVBoxLayout()

        # add all widgets to the main vLayout
        #self.mainLayout.addWidget(self.label1)
        self.mainLayout.addWidget(self.cardsTable)

        # central widget
        self.centralWidget = QWidget()
        self.centralWidget.setLayout(self.mainLayout)

        self.setCentralWidget(self.centralWidget)


def create_widget():
    app    = QApplication(sys.argv)
    widget = MainWindow()
    widget.setWindowTitle("Wichtes AI - developed by Markus Lamprecht @2020")
    widget.setWindowIcon(QIcon('cards/icon.png'))
    widget.show()
    sys.exit(app.exec_()) # blocks whole programm!

if __name__ == "__main__":
    create_widget()
