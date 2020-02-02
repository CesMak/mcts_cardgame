In here I test Monte Carlo Tree Search (MCTS) for a multi player card game.
Normally MCTS is used for games like chess to select a set of promising options (instead of testing all possible options as in minimax algorithm).
Now in comparison to a card game chess is played in turn. In a card game the winner of the last round is allowed to play the next card.
In here I tested mcts on the multi player card game named [witches](https://www.amazon.de/Unbekannt-4990-AMIGO-Witches/dp/B00J5Z7APO)(rules below) for 2 players (each know all cards).

In the feature I would like to extend this card came for multi player. In this case the game will become an imperfect information game.
Thus there are many many more options the algorithm needs to test! Thereby the waiting time will be much longer!

# Getting started
	*	python play.py
	* Select Human player by (line 35, 35 in play.py)
	```
	action = int(input("Action index to play\n"))
	#action = my_game.getRandomOption_()
	```
	* choose different deck size by e.g.: (line 48 in gameClasses.py)
		```
				for val in range(10, 16):# choose different deck size here!
		```
	* The smaller the deck the faster the algorithm
	* Choose Hyperparameters to speed up the algorithm (in play.py)
		```
		mcts = VanilaMCTS(n_iterations=1500, depth=15, exploration_constant=300, state=state, player=current_player, game=my_game)
		```
		* n_iterations:
		* depth: depth level of the tree
		* exploration_constant: exploit a leaf node more or search for even better options?

# Procedure of Monte Carlo Tree Search
1. create VanilaMCTS object
2. run SOLVE
	* for n_iterations = 50 do:
		* **Selection :** select leaf node which have maximum uct(exploration vs. exploitation) value, out:node to expand, depth (root=0)
		* **Expansion :** create all possible outcomes from leaf node    out: expanded tree
		* **Simulation:** simulate game from child node's state until it reaches the resulting state of the game. out: rewards
		* **Backprob  :**  assign rewards back to the top node!

Tree consists of state([self.players, self.rewards, self.on_table_cards, self.played_cards]), current_player, n, w, q
* n = number of iterations
* w = summed reward at that depth
* q = w/n

# Achievements
*	currently for two players
* both now all cards
* just 12-14 of each color no jokers (Test1) was done
* 1-15 (15=joker) 60 cards for 2 players (Test2) was done

# Statistic details / Performance
* Stats with 12 cards of 2,13,14 of Blue Green Red Yellow and 2 Players
	+	Tested with 50 Games, mcts player (always starts) and a **random player**
	+	mcts adjustements: n_iterations=1500, depth=15, exploration_constant=300
	+	Result: total_rewards: [-211. -439.] for game 50 total time: 0:06:52.199611
	+	rewards per game: -4.22, 8,8 per game
* Play 10 games with a **human opponent**
	+	rewards: total_rewards: [-64. -66.] for game 10 total time: 0:10:25.083755
	+ human lost a close match
	+ per game: -6.4, -6.6 per game
* Stats with *60* cards 1-14 15=Joker, and 2 Players (mcts and **random**)
	+ (n_iterations=1500, depth=15, exploration_constant=300


# TODO
* extend for multiplayer (you do not know the options of the other players -> imperfect information game!)
* do tests for evaluation on hyperparams (n_iter, exploration_const, depth)
* Change depth level (such that only x turns are calculated in advance not until the game is finished!)

# Links:
*	https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
*	https://github.com/hayoung-kim/mcts-tic-tac-
* https://www.youtube.com/watch?v=UXW2yZndl7U

# Rules of witches:
*	Aim:	Have a minimum of minus Points!
*	60 	   Cards(4xJoker, 1-14 in Yellow, Green, Red, Blue)
*	Red    Cards give -1 Point (except Red 11)
*	Blue   Cards do nothing    (except Blue 11 if you have it in your offhand deletes green 11 and green 12 if you have it in your offhand as well)
*	Green  Cards do nothing	   (except Green 11 -5 and Green 12 -10 Points)
* Yellow Cards do nothing    (except Yellow 11 +5)
*	A joker can be placed anytime (you do not have to give the same color as the first player)
* If you have no joker and you are not the first player, you have to play the same color as the first player.
* The winner of one round (highest card value) has to start with the next round
* If only Jokers are played the first one wins this round.
* Note: Number 15 is a Joker (in the code)

# Further Notes:
Problem: mcts geht nicht fuer imperfect information games oder sprich sobald mehr als 2 Personen dabei sind explodiert alles?! Bspw. wenn jeder nur 3 Karten bekommt.

Für 4 Spieler und 3 Karten
Müsste man 3!=6 MCTS lernen lassen mit:
1. Zug max 81 mögliche Zustände
Erster Spieler hat 3 Möglichkeiten
Zweiter Spieler 9 mögliche Zustände
Dritter Spieler 27 mögliche Zustände
Vierter Spieler 81 mögliche Zustände

2. Zug für jeden der 81 möglichen Zustände
- wenn spieler 1 gewinnt:
	dann hat erster Spieler 2 möglichkeiten 81*2 zustände
	dann hat zweiter Spieler 2 möglichkeiten 81*4 zustände
	dann hat dritter Spieler 2 möglichkeiten 81*8
	dann hat vierter Spieler 2 möglichkeiten 81*16
das ganze nochmal mal 4 wenn ein anderer Spieler gewinnt also 5184 Zustände.

3. Zug:
