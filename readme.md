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
	+ n_iterations:
	+ depth: depth level of the tree
	+ exploration_constant: exploit a leaf node more or search for even better options?

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

# Proposed Procedure:
1. Simulate with Random moves
2. Train a NN (Classifier) with the input of 500 Games
3. Use the trained NN to Simulate and get better actions
4. Repeat Setp 2-3

# Achievements
*	player can be chosen: RANDOM, MCTS, HUMAN, NN(trained NN with MCTS actions)
* 1-15 (15=joker) 60 cards for 2 players (Test2) was done
* included first shifting round (NN plays currently random)

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
	+	total_rewards: [  13. -266.] for game 10 total time: 1:09:16.619273
* 50 Games, Stats with *60* cards, and **4 Players** (["MCTS", "RANDOM", "MCTS", "RANDOM"]))
	+ n_iterations=50, depth=15, exploration_constant=300
	+ total_rewards: [-106. -517.  -71. -504.] for game 50 total time: 0:11:06.290802

* 50 Game Stats with 4 MCTS Players (Hyper Parameter Search)
	```
	total_rewards: [-305. -258. -241. -169.] for game 50 total time: 1:06:36.474031
	I reset the next game:  50
	The game was started with:
	Number Games  : 50
	Expo Constants: [300, 300, 300, 300]
	depth		 : [15, 15, 15, 15]
	Iteraions     : [100, 100, 100, 100]


	total_rewards: [-235. -243. -263. -270.] for game 50 total time: 4:09:13.044767
	I reset the next game:  50
	The game was started with:
	Number Games  : 50
	Expo Constants: [300, 300, 300, 300]
	depth		 : [15, 15, 15, 15]
	Iteraions     : [1000, 100, 1000, 100]


	total_rewards: [-135. -368. -263. -268.] for game 50 total time: 1:06:20.726154
	I reset the next game:  50
	The game was started with:
	Number Games  : 50
	Expo Constants: [3000, 300, 3000, 300]
	depth		 : [15, 15, 15, 15]
	Iteraions     : [100, 100, 100, 100]

	total_rewards: [-240. -252. -368. -211.] for game 50 total time: 4:13:52.280434
	I reset the next game:  50
	The game was started with:
	Number Games  : 50
	Expo Constants: [3000, 300, 3000, 300]
	depth		 : [15, 15, 15, 15]
	Iteraions     : [100, 1000, 100, 1000]

	total_rewards: [-291. -246. -251. -185.] for game 50 total time: 3:39:30.612880
	Number Games  : 50
	Expo Constants: [400, 400, 400, 400]
	depth		 : [15, 30, 15, 30]
	Iteraions     : [500, 500, 500, 500]

	--> 600, 30, 1000 (should be good adjustements)
	```
* **NN Test** commit: nn_working
  + Test it with ```collect_train_data.py```
	+ a NN with 18000 batches was trained.
	+ 10 Games played timing Performance
	```
	total_rewards: [-70. -34. -84. -58.] for game 10 total time: 0:00:00.706998
	I reset the next game:  10
	The game was started with:
	Number Games  : 10
	Players       : ['NN', 'NN', 'NN', 'NN']
	Expo Constants: [600, 600, 600, 600]
	depth         : [300, 300, 300, 300]
	Iteraions     : [100, 100, 100, 100]
	```
	+ NN vs MCTS:
	+ MCTS is much better (cause it knows all cards of the players)
	+ NN does not know the cards of each player!
	```
	total_rewards: [-267.  -42.  -92. -191.] for game 25 total time: 0:07:33.946991
	I reset the next game:  25
	The game was started with:
	Number Games  : 25
	Players       : ['NN', 'MCTS', 'MCTS', 'NN']
	Expo Constants: [600, 600, 600, 600]
	depth         : [300, 300, 300, 300]
	Iteraions     : [100, 100, 100, 100]
	```
	+ **Problem:** NN did not learn constraint!
	+ In case that it suggest to make an impossible move the first possible move is played!
	```
	total_rewards: [-714. -465. -451. -647.] for game 100 total time: 0:00:07.207669
	I reset the next game:  100
	The game was started with:
	Number Games  : 100
	Players       : ['NN', 'NN', 'NN', 'NN']
	Expo Constants: [600, 600, 600, 600]
	depth         : [300, 300, 300, 300]
	Iterations    : [100, 100, 100, 100]
	Invalid moves tried to play: 297
	```

# TODO
* Done: extend for multiplayer [if you know all cards]
* Train a NN:
	* input as 0,1:  played_cards, cards_on_table, card_options
	* output: estimated result value of all players, option to play!
	* Train such a network (to achieve faster moves!) see [here](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

* Do Graphics pygame
* Extend for multiplayer (you do not know the options of the other players -> imperfect information game!)
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


Other Card Games:
+ Hearts: http://fse.studenttheses.ub.rug.nl/15440/1/Bachelor_Thesis_-_Maxiem_Wagen_1.pdf
+ RI Book: https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf
