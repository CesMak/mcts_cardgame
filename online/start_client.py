# -*- coding: utf-8 -*-
import sys
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QFont
from PyQt5.QtWidgets import QMainWindow, QApplication, QSplashScreen, QProgressBar
from PyQt5 import QtCore

#Links
# See also https://github.com/eladj/QtCards

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        ## LOAD IMPORTS HERE TO SAVE STARTUP TIME
        print("load imports")
        from PyQt5.QtWidgets import QGraphicsView, QWidget, QGraphicsScene, QGridLayout, QVBoxLayout
        from table import cardTableWidget, CardTableWidgetExtend

        self.cardsTable = CardTableWidgetExtend()
        self.cardsTable.initUI("client_options.json")

        # main layout
        self.mainLayout = QVBoxLayout()

        # add all widgets to the main vLayout
        #self.mainLayout.addWidget(self.label1)
        self.mainLayout.addWidget(self.cardsTable)

        # central widget
        self.centralWidget = QWidget()
        self.centralWidget.setLayout(self.mainLayout)

        self.setCentralWidget(self.centralWidget)
        
    def closeEvent(self, event):
        print("Nasty error close, disable Server and Client!")
        event.accept() # let the window close
        print(eee)


def initStartupProgressBar():
    splash_pix = QPixmap(300,100)
    splash_pix.fill(QtCore.Qt.white)
    painter = QPainter(splash_pix)
    font=QFont("Times", 30)
    painter.setFont(font)
    painter.drawText(20,65,"Loading all libs")
    painter.end()
    splash = QSplashScreen(splash_pix, QtCore.Qt.WindowStaysOnTopHint)
    progressBar = QProgressBar(splash)
    progressBar.setGeometry(splash.width()/10, 8*splash.height()/10,8*splash.width()/10, splash.height()/10)
    splash.setMask(splash_pix.mask())
    splash.show()
    return splash, progressBar

def create_widget():
    app    = QApplication(sys.argv)
    splash, progressBar = initStartupProgressBar()
    widget = MainWindow()
    progressBar.setValue(100)
    app.processEvents()
    widget.setWindowTitle("Witches 0.5 - developed by Markus Lamprecht @2020")
    widget.setWindowIcon(QIcon('../cards/icon.png'))
    widget.show()
    splash.finish(widget)
    sys.exit(app.exec_()) # blocks whole programm!

if __name__ == "__main__":
    create_widget()
