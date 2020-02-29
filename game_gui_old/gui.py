from threading import Thread
from queue import Queue, Empty
import pygame
import time
from enum import Enum, auto
### uncomment this line for testing!
#from witches_game import card, deck, player, game

_screenSize = _screenWidth, _screenHeight = (1300, 700)

_path_to_card_imgs = "game_gui/cards"

BACKGROUND = (15, 105, 25)
WHITE      = (255, 255, 255)
YELLOW     = (240, 240, 0)

class GUI(Thread):

	class MessageType(Enum):
		SURFACE = auto()
		EMPTY  = auto()
		RECT   = auto()

	def __init__(self, human_player_idx=-1):
		#-1 means no human player (show all cards open!)
		Thread.__init__(self)

		pygame.init()
		pygame.font.init()

		self.card_rect_positions = []
		self.queue = Queue()
		self.names = []
		self.list_of_cards = {0: [], 1: [], 2: [], 3: []}
		self.input_cards   = {0: 0, 1: 0, 2: 0, 3: 0}
		self.screen = pygame.display.set_mode(_screenSize)
		pygame.display.set_caption("Witches_by_M.Lamprecht")
		self.screen.fill(BACKGROUND)
		self.human_player_idx = human_player_idx

	def clear(self):
		self.queue.put((GUI.MessageType.EMPTY,))

	def getCardImage(self, input_card, show_back=True):
		if show_back:
			return pygame.image.load(_path_to_card_imgs+"/"+"back"+".png")
		return pygame.image.load(_path_to_card_imgs+"/"+str(input_card.color)+str(input_card.value)+".png")

	def putInputCards(self):
		for key in (self.input_cards):
			card = self.input_cards[key]
			if card!= 0:
				cardImage = self.getCardImage(card, show_back=False)
				imageWidth, imageHeight = cardImage.get_size()
				x = _screenWidth // 4 + imageWidth/2*key
				y = _screenHeight // 4 + imageWidth/2
				self.queue.put((GUI.MessageType.SURFACE, cardImage, (x, y), card))

	def removeInputCards(self, winner_idx=0, results_=[]):
		self.clear()
		# for i in range(4):
		# 			self.dealCards(i, self.list_of_cards[i], appendCard=0)
		self.input_cards   = {0: 0, 1: 0, 2: 0, 3: 0}
		#show stats:
		my_text = ""
		# TODO:
		# for j, name in enumerate(self.names):
		# 	print(j, name)
			#my_text+=str(name[0])+": "+str(results_[j])+"   "
		self.stats_text(text=self.names[winner_idx]+" won this round|  "+my_text)

	def highlight_name(self, player_name = 0):
		# Unhighlight player
		self.nameUp(self.names[0], highlight=0)
		self.nameRight(self.names[1], highlight=0)
		self.nameDown(self.names[2], highlight=0)
		self.nameLeft(self.names[3], highlight=0)

		if player_name == 0:
			self.nameUp(self.names[player_name], highlight=1)
		elif player_name == 1:
			self.nameRight(self.names[player_name], highlight=1)
		elif player_name == 2:
			self.nameDown(self.names[player_name], highlight=1)
		elif player_name == 3:
			self.nameLeft(self.names[player_name], highlight=1)



	def playCard(self, input_card, player=0, round_finished=True):
		# kind of a nasty hack:
		if round_finished:
			self.clear()

		#remove this card from the deck
		print(self.list_of_cards[player])
		print("inputcard:", input_card)
		self.list_of_cards[player].remove(input_card)

		if round_finished:
			for i in range(4):
					self.dealCards(i, self.list_of_cards[i], appendCard=0)
		self.input_cards[player] = input_card
		self.putInputCards()

		#highlight next Player which has to move:
		if player<3:
			self.highlight_name(player_name=player+1)
		if player == 3:
			self.highlight_name(player_name=0)


	# Render cards
	def cardLeft(self, card, nu_cards=2, index=0, show_back=False):
		cardImage = self.getCardImage(card, show_back)
		imageWidth, imageHeight = cardImage.get_size()
		x = _screenWidth // 7 - imageWidth // 2
		y = _screenHeight // nu_cards+index*37
		self.queue.put((GUI.MessageType.SURFACE, cardImage, (x, y), card))
		self.card_rect_positions.append(cardImage)
		self.nameLeft(self.names[3])

	def cardDown(self, card, nu_cards=2, index=0, show_back=False):
		cardImage = self.getCardImage(card, show_back)
		imageWidth, imageHeight = cardImage.get_size()
		x = _screenWidth // nu_cards+index*37+imageWidth  #x = _screenWidth // 2 - imageWidth // 2
		y = (_screenHeight * 4) // 5 - imageHeight // 2
		self.queue.put((GUI.MessageType.SURFACE, cardImage, (x, y), card))
		self.nameDown(self.names[2])

	def cardRight(self, card, nu_cards=2, index=0, show_back=False):
		cardImage = self.getCardImage(card, show_back)
		imageWidth, imageHeight = cardImage.get_size()
		x = (_screenWidth * 4) // 4.6 - imageWidth // 2
		y = _screenHeight // nu_cards+index*37 #y = _screenHeight // 2 - imageHeight // 2
		self.queue.put((GUI.MessageType.SURFACE, cardImage, (x, y), card))
		self.nameRight(self.names[1])

	def cardUp(self, card, nu_cards=2, index=0, show_back=False):
		cardImage = self.getCardImage(card, show_back)
		imageWidth, imageHeight = cardImage.get_size()
		x = _screenWidth // nu_cards+index*37+imageWidth
		y = _screenHeight // 4 - imageHeight // 2
		self.queue.put((GUI.MessageType.SURFACE, cardImage, (x, y), card))
		self.nameUp(self.names[0])

	# Render names
	def nameLeft(self, name,  highlight=0):
		nameFont = pygame.font.SysFont("Comic Sans MS", 30)
		if highlight:
			nameSurface = nameFont.render(name, False, YELLOW)
		else:
			nameSurface = nameFont.render(name, False, WHITE)
		nameSurface = pygame.transform.rotate(nameSurface, 90)
		surface_width, surface_height = nameSurface.get_size()
		x = 10
		y = _screenHeight // 2 - surface_height // 2
		self.queue.put((GUI.MessageType.SURFACE, nameSurface, (x, y)))

	def nameDown(self, name,  highlight=0):
		nameFont = pygame.font.SysFont("Comic Sans MS", 30)
		if highlight:
			nameSurface = nameFont.render(name, False, YELLOW)
		else:
			nameSurface = nameFont.render(name, False, WHITE)
		surface_width, surface_height = nameSurface.get_size()
		x = _screenWidth // 2 - surface_width // 2
		y = _screenHeight - surface_height - 10
		self.queue.put((GUI.MessageType.SURFACE, nameSurface, (x, y)))

	def nameRight(self, name,  highlight=0):
		nameFont = pygame.font.SysFont("Comic Sans MS", 30)
		if highlight:
			nameSurface = nameFont.render(name, False, YELLOW)
		else:
			nameSurface = nameFont.render(name, False, WHITE)
		nameSurface = pygame.transform.rotate(nameSurface, -90)
		surface_width, surface_height = nameSurface.get_size()
		x = _screenWidth - surface_width - 10
		y = _screenHeight // 2 - surface_height // 2
		self.queue.put((GUI.MessageType.SURFACE, nameSurface, (x, y)))

	def nameUp(self, name, highlight=0):
		nameFont = pygame.font.SysFont("Comic Sans MS", 30)
		if highlight:
			nameSurface = nameFont.render(name, False, YELLOW)
		else:
			nameSurface = nameFont.render(name, False, WHITE)
		surface_width, surface_height = nameSurface.get_size()
		x = _screenWidth // 2 - surface_width // 2
		y = 10
		self.queue.put((GUI.MessageType.SURFACE, nameSurface, (x, y)))

	def getWidth_Hight_ofName(self, name, player):
		nameFont = pygame.font.SysFont("Comic Sans MS", 30)
		nameSurface = nameFont.render(name, False, WHITE)
		#player 0 and player 2 do not rotate at all
		if player == 1:
			nameSurface = pygame.transform.rotate(nameSurface, -90)
		if player == 3:
			nameSurface = pygame.transform.rotate(nameSurface, 90)
		surface_width, surface_height = nameSurface.get_size()
		return surface_width, surface_height

	def clearName(self, name, player):
		surface_width, surface_height  = self.getWidth_Hight_ofName(name, player)
		if player == 1:
			x = _screenWidth - surface_width - 10
			y = _screenHeight // 2 - surface_height // 2
		elif player == 0:
			x = _screenWidth // 2 - surface_width // 2
			y = 10
		elif player == 2:
			x = _screenWidth // 2 - surface_width // 2
			y = _screenHeight - surface_height - 10
		elif player == 3:
			x = 10
			y = _screenHeight // 2 - surface_height // 2
		rect = pygame.Rect(x, y, surface_width, surface_height)
		self.queue.put((GUI.MessageType.RECT, rect))

	def stats_text(self, text="Stats here:"):
		nameFont = pygame.font.SysFont("Comic Sans MS", 20)
		nameSurface = nameFont.render(text, False, WHITE)
		surface_width, surface_height = nameSurface.get_size()
		x = _screenWidth/50
		y = _screenHeight/50
		self.queue.put((GUI.MessageType.SURFACE, nameSurface, (x, y)))


	def dealCards(self, player_nu, card_list, nu_players=4, appendCard=1):
		show_back = False
		if self.human_player_idx>=0:
			show_back = True
			if player_nu == self.human_player_idx:
				show_back = False

		if not nu_players==4:
			print("Error currently just 4 Players allowed!")
			return None
		if player_nu == 0:
			for i, card in enumerate(card_list):
				self.cardUp(card, nu_cards=len(card_list), index=i, show_back=show_back)
		elif player_nu == 1:
			for i, card in enumerate(card_list):
				self.cardRight(card, nu_cards=len(card_list), index=i, show_back=show_back)
		elif player_nu == 2:
			for i, card in enumerate(card_list):
				self.cardDown(card, nu_cards=len(card_list), index=i, show_back=show_back)
		elif player_nu == 3:
			for i, card in enumerate(card_list):
				self.cardLeft(card, nu_cards=len(card_list), index=i, show_back=show_back)

		if appendCard:
			self.list_of_cards[player_nu] = card_list

	def run(self):
		# läuft die ganze zeit durch die queue und malt alles was in der queue ist!
		while True:
			# Events
			close = False
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					close = True
					break
				if event.type == pygame.MOUSEBUTTONDOWN:
					x, y = event.pos
					print(x, y)
					if self.card_rect_positions[0].get_rect().collidepoint(x, y):
						print('clicked on image')
			if close:
				break

			# Message queue
			try:
				# following line removes item from queue and if possible adds it to the display
				item = self.queue.get(block = False)
				messageType = item[0]
				# Draw a surface
				if messageType is GUI.MessageType.SURFACE:
					surface = item[1]
					x, y = item[2]
					self.screen.blit(surface, (x, y))
				elif messageType is GUI.MessageType.RECT:#übermale namen mit grün
					my_rect = item[1]
					self.screen.fill(BACKGROUND, rect=my_rect)
				# Clear the screen
				elif messageType is GUI.MessageType.EMPTY:
					self.screen.fill(BACKGROUND)
					print("RECT is:")
					print(self.screen.get_rect())
			except Empty:
				pass

			# updates entire surface all at once! to new queue!
			pygame.display.flip()

		pygame.display.quit()


##### Uncomment following to test just the gui:
# if __name__ == "__main__":
# 	gui = GUI()
# 	gui.start() # start thread!
#
# 	my_game = game(["Tim", "Bob", "Lena", "Anja"])
# 	gui.names = my_game.names_player
# 	for i in range(4):
# 		gui.dealCards(i, my_game.players[i].getHandCardsSorted())
#
# 	for j in range(10):
# 		for i in range(4):
# 			gui.playCard(my_game.players[i].getHandCardsSorted()[j], player=i)
# 		time.sleep(1)
# 		gui.removeInputCards()
