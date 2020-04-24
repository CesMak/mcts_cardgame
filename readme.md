![mcts_witches](data/imgs/rl_players.png)

To play a card as HUMAN **double click** the card.

# Windows
Download [witches_ai_0.0.zip](https://drive.google.com/file/d/1UZQyMhH46qzoKJoSYkbsGtTatB2liqXQ/view?usp=sharing) and open the **gui.exe**.
MultiPlayer Version [witches_ai_0.1.zip](https://drive.google.com/file/d/1sRG3SbiLQUZq3qt2FQw-o59XjfTHH68V/view?usp=sharing)

# Linux
Commited Versions:
  * commit **witches_0.1** Beta shifting included in learning process
  * commit **witches_0.2** Beta shifting included also in gui.py (fixed core dumped when playing with human)
```
python gui.py
```

# Options
See file **gui_options.json**
```json
{
  "names": ["Laura", "Alfons", "Frank", "Lea"],
  "type": ["NN", "RANDOM", "HUMAN", "RL0"],  "[HUMAN, RANDOM, NN, MCTS, RL0, RL1, ... RL5]"
  "expo": [500, 500, 500, 500],              "-> adjustements only for MCTS"
  "depths": [300, 300, 300, 300],            "-> adjustements only for MCTS"
  "itera": [5000, 5000, 5000, 5000],         " -> adjustements only for MCTS"
  "faceDown": [false, false, false, false],  " [true, false] If Cards are visible or not"
  "sleepTime": 0.001,                        " Time to wait between 2 moves"
  "model_path_for_NN": "data/test.pth",      " Input path for Neuronal Network"
  "nu_games": 100,                           " Number of Games"
  "shifting_phase": 20,                      " TODO"
  "mcts_save_actions": false,                " -> adjustements only for MCTS"
  "mcts_actions_path": "data/actions_strong44__mcts.txt",  " -> adjustements only for MCTS"
  "automatic_mode": false,                   " [true, false]  true: play a pickle game_play"
  "save_game_play": false,                   " [true, false]  true: save a pickle game_play"
  "game_play_path": "data/game_play.pkl",    " *.pkl          path for pickle game_play"
  "onnx_path": "data/model_long_training.pth.onnx"  "[model_long_training.pth.onnx, model.pth.onnx, actions_all.pth.onnx]"
  "onnx_rl_path": ["rl_path3", "rl_path4", "rl_path5", "rl_path6"]   " in data/*.onnx [rl_path3, rl_path4, rl_path5, rl_path6, ... rl_path12_further] path PPO trained"
}
```

# Playing online
Use this configuration:
|Option|Server|Client|
|--|--|--|
|online_type|Server|Client|
|open_ip|any|172.168.11.5 (e.g.)|
|names|['Max', ...] updated|['Hans'] <- unique name|
|type|['Server', 'Client', ...]|['Client']|

# Player Types
  * RANDOM: plays a random possible card
  * HUMAN : you can choose to play (use double click)
  * NN    : Is player that was trained by classifying data generated by MCTS
  * MCTS  : Monte Carlo Tree Search Player
    + Chooses and action based on predictions into the future (similar to minimax)
    + *depth*: At what depth should the tree be spanned
    + *expo* : Trade-off between exploration (the higher this value ) and exploitation
    + *itera*: Max number of iterations (the lower the faster)
  * RL     : Reinforcement Learning
    + Select RL(number)  number in range 0, len(onnx_rl_path), the higher the stronger the RL
    + An actor-critic Proximal Policy Optimization (PPO) Reinforcement Learning algorithm.
    + Generate trained model using the **[modules/ppo_witches](https://github.com/CesMak/mcts_cardgame/blob/master/modules/ppo_witches.py)** and the **[modules/gym-witches](https://github.com/CesMak/mcts_cardgame/tree/master/modules/gym-witches)**
    + *onnx_rl_path*: Path to the trained model
        + rl_path4_82.onnx (is the stronges one), wins 80% of the games (against only RANDOM players)
        + trained for 2h on single cpu, i5, no vectorized environment
        + I still can beat ai (ai is to greedy) Stats 3 Games: [Rand=-32, RL=-9, RL=-35, ME=+1]

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
* Buy the real-card game e.g. here: [buy-witches](https://www.amazon.de/Unbekannt-4990-AMIGO-Witches/dp/B00J5Z7APO)

# Training an AI
## Start with MCTS (similar to minimax)
To train a strong human-competitive player I first tested Monte Carlo Tree Search (MCTS).
Usually, MCTS is used for games like chess to select a set of promising options (instead of testing all possible options as in minimax algorithm).
I also used mcts to predict possible future rewards if a specific card is played and recorded the state and the action that mcts predicted.
In here I assumed, that mcts knows all cards(If not it is very hard to predict possible future actions).
The recorded data could then be used to train a Neuronal Network (Classifying Problem) Inputs and Outputs are known.
The trained NN at the end does not know the cards of the other players.
However the NN only performs a little bit better than RANDOM players:
![mcts_witches](data/imgs/nn_better_than_random.png)

## PYTORCH REINFORCE
I tested the pytorch [REINFORCE](https://pytorch.org/docs/stable/distributions.html) (see e.g. this [example](https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py)) algorithm in [**modules/reinforcement_learning.py**](https://github.com/CesMak/mcts_cardgame/blob/master/modules/reinforcement_learning.py).
* see commit **is_learning**
* run with  `modules$ python reinforcement_learning.py`
* player 0 is the trained player
* 50 games played
* half of them won by trained player!
![is_learning](data/imgs/is_learning.png)

## PYTORCH PPO
### Learning Procedure  (gym interface)
* `python state = env.reset()`
  + with state: 240x1
  + state = on_table, on_hand, played, play_options (each 60x1 one-hot encoded)
  + *get the state right before ai_player has to play!*
* `python action = ppo_test.policy_old.act(state, memory)`
* `python state, reward, done, nu_games_won = env.step(action)`

### PPO with Monte Carlo Reward Estimation
No GAE used. See the file **ppo_witches.py**

* See file  [**modules/ppo_witches.py**](https://github.com/CesMak/mcts_cardgame/blob/master/modules/ppo_witches.py).
* First I learned without discounted rewards:
* ![is_learning-img](data/imgs/nodiscounting_final.png)
* Problem: The learning stopped to early (it was a short sighted learning) see also [here](https://github.com/henrycharlesworth/big2_PPOalgorithm/issues/9)
* Next I included the mc rewards and played around with the hyper-parameters:
![is_learning](data/imgs/discounted_rewards.png)
* Results
* Use beta=0.01 (at the beginning!)
* Using 512 for latent layers does not improve the results (64 are already enough)
* Using update_timestep of 2000 is advised!
* eps clipping = 0.1 is advised!
* 81% is a maximum
  * Can still be better!
  * Plays to greedy (always captures blue 11)
  * Plays Joker to early!

* PPO Hyperparameter Tuning
  + if gets bader again at some point lower the lr (to be adjusted first)
* Training against trained players
  + is computationally expensive (use path instead of .onnx)
  + Has not a big effect (correct hyperparams not found yet?)
  + Tested gamma=0.7  -> almost no effect
  + Tested update timestep = 5 -> almost no effect
  + Example 35% : Game ,0018000, rew ,-5.351, inv_mo ,0.0025, won ,[48. 79. 51. 58.],  Time ,0:07:40.390401

**05.04.2020 - adjustements for learning correct moves**
at commit **best_learning_mc** inv moves is at 0.01 after 270000 episodes. Rewards with -100 in case of wrong move and with trick reward+21 in case of correct move.
+ change input to 180 and see if improves     **NO does not learn faster, use also options at input state**
+ change value layer and see if it improves   **NO does not learn faster, use seperate**
+ include shifting? and see what is changing
  + rewarding with total current rewards seems also to work (changed gameClasses with newest one)
  + So far it seems to work however, correct moves and invalid moves has different meaning inv_moves = 0.0455 correct moves = 16.23
  + **Including some additional states has helped!**
  + Found a new best player **rl_path11_op** win rate of 95% against random player! With shifting:
  ![95_percent_winner_mc_rewarding-img](data/imgs/95_percent_winner_mc_rewarding.png)
  **new player is rl0**
  ![95_percent_winner_mc_rewarding-img](data/imgs/95_percent_rl0.png)
  + See commit **95_precent_mc_rewarding_winner** or nicer version **shift_trained_further**
    + Play 10 rounds against this player as a human!
    + Shifted cards are not always best options!
    + Play against pretrained copys (see path 12 was has not better stats....)
    + Train pretrained copys further.....  
    + -1101  -890  in 200 games
    + -969   -935 in again 200 games
    + Test monte carlo options!!! as add input to Network output?!
  + Note that current rewarding aims to find yellow 11 as fast as possible!

**17.04.2020 Learning multi Train against each other**
* learn against trained players endures for 15h, shift works better see rl_path14_multi
* still worse than me (human) and worse than:    "rl_path12_further"
* see commit  **included_multi**
* **Added iig folder**
* adjustable card number
* see commit **included_multi**
* Not to see also the [collab](https://drive.google.com/drive/folders/1ru_TDEmMKXjiv4ie3Zl4nYPsHngkZExr) version (is not faster however, and trains only when active at pc max. 12h)

* how to reward?
* works also with and without step?
* max. should be 0.001 invalid moves in 2000 games or episodes?
* reset to  best_learning_mc and see how fast it learns correct moves ....

+ anderer Zustand:
  * For each opponent is in shifing phase
  * For each opponent has in offhand [11 and 12, 13, 14, J] -> 15 states
  * Not has... color x
  * Control before

### PPO with gae
See the file **ppo2_witches.py**

### PPO with LSTM and gae
See the file **ppo3_witches.py**

### Test Baselines
  * Tested baselines see **ppo_baselines.py**
  + Need of constructing own model! (Damit auch zuege lernt)
  + Not so easy to use (export as onnx etc.)

## TODO
* Tune Hyperparameters in current ppo_witches.py
* Wie sagen, dass shifting phase ist?
* Test if after shift phase player has new cards (in his options!)
* Monitor value, loss, entropy
* Use vectorenv test baselines custom environment  

* Include shifting
  * First  move: shift 1 card (possible cards are all cards on hand)
  * Second move: shift 1 card (possible cards are all on hand except already shifted one)
  * Third  move: play a card.
  + seems not to learn that now is a shift phase!
  + using 1 in on table cards during shift phase
  + Test if after shift phase player has new cards (in his options!)

  * Test minimal ppo implementation with GAE


# Further Notes
## MCTS
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

The tree looks as follows:
tree = {root_id: {'state': state,
					'player': player,
					'cards_away': [], # this is used for shifting!
					'child': [],
					'parent': None,
					'n': 0,
					'w': 0,
					'q': None}}

## PPO
* Clipping probs = torch.clamp(probs, 0, 1) # did not work for me
* torch.nn.utils.clip_grad_norm_(self.parameters(), 5) in updatePolicy
* working adam see commit **working_adam**
* Problem for SGD and adam (what is the reason?!)
  ```
  ERROR!!!! invalid multinomial distribution (encountering probability entry < 0)
  ```
  * So um die 22 gewinnrate rum lokales minimum erreicht?!
  * use different lr !
  * use clipping
  * use clamping?!
* TEST PPO Pytorch from [here](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py) [expl](https://www.youtube.com/watch?v=5P7I-xPq8u8&t=208s)
  * Return = discounted sum of rewards (gamma = Interest in financial get Money NOW! greedy or not)
  * Value Function tries to estimate final reward in this episode
  * advantage estimate = discounted reward - baseline estimate (by value function)
    * >0 Gradient is positive  Increase action probabilities
  * Running Gradient Descent on a single batch destroys your policy (cause of NOISE) -> Trust Region is required!
* Incooperate Loss from [here](https://github.com/henrycharlesworth/big2_PPOalgorithm/blob/master/PPONetwork.py)
* Test nn.Tanh(), as activation function! see [here](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py)
* Test Discounted rewards (set gamma to not greedy)


## Older Stuff:
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

# Creating an exe
	* convert torch model and params to onnx
	* use pyinstaller (also on ubuntu possible!)
	* in this case exe should be smaller!

# TODO
* Create EXE!
* Have a look at TD-Learning!
* Done: extend for multiplayer [if you know all cards]
* Train a NN:
	* input as 0,1:  played_cards, cards_on_table, card_options
	* output: estimated result value of all players, option to play!
	* Train such a network (to achieve faster moves!) see [here](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
* **Use Google collab to generate data for your NN**
* Read this first UCT: https://hci.iwr.uni-heidelberg.de/system/files/private/downloads/297868474/report_robert-klassert.pdf
* Test this one: multiprocessing.Pool here: https://wiseodd.github.io/techblog/2016/06/13/parallel-monte-carlo/
* See also this good explanation: https://pdfs.semanticscholar.org/fe90/c1f9955ba1f06f5ef26bde100bcc5c7a3327.pdf
* Or use CUDA or parallel mcts!

* Do Graphics pygame
* Extend for multiplayer (you do not know the options of the other players -> imperfect information game!)
* do tests for evaluation on hyperparams (n_iter, exploration_const, depth)
* Change depth level (such that only x turns are calculated in advance not until the game is finished!)

# Links:
*	https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
*	https://github.com/hayoung-kim/mcts-tic-tac-
* https://www.youtube.com/watch?v=UXW2yZndl7U

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


# Example tree
```
{(0,): {'state': [[<gameClasses.player object at 0x7f84940f26d8>, <gameClasses.player object at 0x7f84940fe6d8>, <gameClasses.player object at 0x7f84940fe710>, <gameClasses.player object at 0x7f84940fe748>], array([0., 0., 0., 0.]), [3 of B, 13 of B, 6 of B, J of Y], [3 of B, 13 of B, 6 of B, J of Y, 9 of Y, 1 of Y, 5 of Y], 20, array([0., 0., 0., 0.])], 'player': 0, 'cards_away': [], 'child': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 'parent': None, 'n': 3, 'w': 0.0, 'q': 0.0},

(0, 0): {'state': [[<gameClasses.player object at 0x7f8494104588>, <gameClasses.player object at 0x7f8494104c88>, <gameClasses.player object at 0x7f84941123c8>, <gameClasses.player object at 0x7f8494112ac8>], array([0., 0., 0., 0.]), [3 of B], [3 of B], 20, array([0., 0., 0., 0.])], 'player': 1, 'cards_away': 0, 'child': [0, 1, 2, 3, 4, 10], 'parent': (0,), 'n': 1, 'w': 4.0, 'q': 4.0},

(0, 1): {'state': [[<gameClasses.player object at 0x7f8494115208>, <gameClasses.player object at 0x7f8494115908>, <gameClasses.player object at 0x7f8494119048>, <gameClasses.player object at 0x7f8494119748>], array([0., 0., 0., 0.]), [9 of B], [9 of B], 20, array([0., 0., 0., 0.])], 'player': 1, 'cards_away': 1, 'child': [0, 1, 2, 3, 4, 10], 'parent': (0,), 'n': 1, 'w': 4.0, 'q': 4.0},

(0, 2): {'state': [[<gameClasses.player object at 0x7f8494119e48>, <gameClasses.player object at 0x7f8494113588>, <gameClasses.player object at 0x7f8494113c88>, <gameClasses.player object at 0x7f847844b3c8>], array([0., 0., 0., 0.]), [10 of B], [10 of B], 20, array([0., 0., 0., 0.])], 'player': 1, 'cards_away': 2, 'child': [], 'parent': (0,), 'n': 0, 'w': 0, 'q': 0},

(0, 3): {'state': [[<gameClasses.player object at 0x7f847844bac8>, <gameClasses.player object at 0x7f849411b208>, <gameClasses.player object at 0x7f849411b908>, <gameClasses.player object at 0x7f847844f048>], array([0., 0., 0., 0.]), [>11< of B], [>11< of B], 20, array([0., 0., 0., 0.])], 'player': 1, 'cards_away': 3, 'child': [], 'parent': (0,), 'n': 0, 'w': 0, 'q': 0},

(0, 4): {'state': [[<gameClasses.player object at 0x7f847844f748>, <gameClasses.player object at 0x7f847844fe48>, <gameClasses.player object at 0x7f8478453588>, <gameClasses.player object at 0x7f8478453c88>], array([0., 0., 0., 0.]), [12 of B], [12 of B], 20, array([0., 0., 0., 0.])], 'player': 1, 'cards_away': 4, 'child': [], 'parent': (0,), 'n': 0, 'w': 0, 'q': 0}, (0, 5): {'state': [[<gameClasses.player object at 0x7f84784573c8>, <gameClasses.player object at 0x7f8478457ac8>, <gameClasses.player object at 0x7f8478458208>, <gameClasses.player object at 0x7f8478458908>], array([0., 0., 0., 0.]), [J of B], [J of B], 20, array([0., 0., 0., 0.])], 'player': 1, 'cards_away': 5, 'child': [], 'parent': (0,), 'n': 0, 'w': 0, 'q': 0}, (0, 6): {'state': [[<gameClasses.player object at 0x7f8478452048>, <gameClasses.player object at 0x7f8478452748>, <gameClasses.player object at 0x7f8478452e48>, <gameClasses.player object at 0x7f847845c588>], array([0., 0., 0., 0.]), [8 of G], [8 of G], 20, array([0., 0., 0., 0.])], 'player': 1, 'cards_away': 6, 'child': [], 'parent': (0,), 'n': 1, 'w': -8.0, 'q': -8.0}, (0, 7): {'state': [[<gameClasses.player object at 0x7f847845cc88>, <gameClasses.player object at 0x7f847845b3c8>, <gameClasses.player object at 0x7f847845bac8>, <gameClasses.player object at 0x7f8478461208>], array([0., 0., 0., 0.]), [10 of G], [10 of G], 20, array([0., 0., 0., 0.])], 'player': 1, 'cards_away': 7, 'child': [], 'parent': (0,), 'n': 0, 'w': 0, 'q': 0}, (0, 8): {'state': [[<gameClasses.player object at 0x7f8478461908>, <gameClasses.player object at 0x7f8478463048>, <gameClasses.player object at 0x7f8478463748>, <gameClasses.player object at 0x7f8478463e48>], array([0., 0., 0., 0.]), [>11< of G], [>11< of G], 20, array([0., 0., 0., 0.])], 'player': 1, 'cards_away': 8, 'child': [], 'parent': (0,), 'n': 0, 'w': 0, 'q': 0}, (0, 9): {'state': [[<gameClasses.player object at 0x7f8478465588>, <gameClasses.player object at 0x7f8478465c88>, <gameClasses.player object at 0x7f84784683c8>, <gameClasses.player object at 0x7f8478468ac8>], array([0., 0., 0., 0.]), [14 of G], [14 of G], 20, array([0., 0., 0., 0.])], 'player': 1, 'cards_away': 9, 'child': [], 'parent': (0,), 'n': 0, 'w': 0, 'q': 0}, (0, 10): {'state': [[<gameClasses.player object at 0x7f847846a208>, <gameClasses.player object at 0x7f847846a908>, <gameClasses.player object at 0x7f847846b048>, <gameClasses.player object at 0x7f847846b748>], array([0., 0., 0., 0.]), [1 of R], [1 of R], 20, array([0., 0., 0., 0.])], 'player': 1, 'cards_away': 10, 'child': [], 'parent': (0,), 'n': 0, 'w': 0, 'q': 0}, (0, 11): {'state': [[<gameClasses.player object at 0x7f847846be48>, <gameClasses.player object at 0x7f847845e588>, <gameClasses.player object at 0x7f847845ec88>, <gameClasses.player object at 0x7f84784703c8>], array([0., 0., 0., 0.]), [2 of R], [2 of R], 20, array([0., 0., 0., 0.])], 'player': 1, 'cards_away': 11, 'child': [], 'parent': (0,), 'n': 0, 'w': 0, 'q': 0}, (0, 12): {'state': [[<gameClasses.player object at 0x7f8478470ac8>, <gameClasses.player object at 0x7f8478473208>, <gameClasses.player object at 0x7f8478473908>, <gameClasses.player object at 0x7f8478476048>], array([0., 0., 0., 0.]), [4 of R], [4 of R], 20, array([0., 0., 0., 0.])], 'player': 1, 'cards_away': 12, 'child': [], 'parent': (0,), 'n': 0, 'w': 0, 'q': 0},

(0, 13): {'state': [[<gameClasses.player object at 0x7f8478476748>, <gameClasses.player object at 0x7f8478476e48>, <gameClasses.player object at 0x7f8478477588>, <gameClasses.player object at 0x7f8478477c88>], array([0., 0., 0., 0.]), [13 of Y], [13 of Y], 20, array([0., 0., 0., 0.])], 'player': 1, 'cards_away': 13, 'child': [], 'parent': (0,), 'n': 0, 'w': 0, 'q': 0}, (0, 14): {'state': [[<gameClasses.player object at 0x7f84784793c8>, <gameClasses.player object at 0x7f8478479ac8>,

<gameClasses.player object at 0x7f847847b208>, <gameClasses.player object at 0x7f847847b908>], array([0., 0., 0., 0.]), [14 of Y], [14 of Y], 20, array([0., 0., 0., 0.])], 'player': 1, 'cards_away': 14, 'child': [], 'parent': (0,), 'n': 0, 'w': 0, 'q': 0},

(0, 0, 0): {'state': [[<gameClasses.player object at 0x7f847847ef98>, <gameClasses.player object at 0x7f847847e6a0>, <gameClasses.player object at 0x7f8478480cf8>, <gameClasses.player object at 0x7f8478481358>], array([0., 0., 0., 0.]), [3 of B, 1 of B], [3 of B, 1 of B], 20, array([0., 0., 0., 0.])], 'player': 2, 'cards_away': 0, 'child': [], 'parent': (0, 0), 'n': 0, 'w': 0, 'q': 0},
(0, 0, 1): {'state': [[<gameClasses.player object at 0x7f8478481ac8>, <gameClasses.player object at 0x7f8478486198>, <gameClasses.player object at 0x7f8478486898>, <gameClasses.player object at 0x7f8478486f98>], array([0., 0., 0., 0.]), [3 of B, 2 of B], [3 of B, 2 of B], 20, array([0., 0., 0., 0.])], 'player': 2, 'cards_away': 1, 'child': [], 'parent': (0, 0), 'n': 1, 'w': 4.0, 'q': 4.0},
(0, 0, 2): {'state': [[<gameClasses.player object at 0x7f8478485748>, <gameClasses.player object at 0x7f8478485dd8>, <gameClasses.player object at 0x7f847840a518>, <gameClasses.player object at 0x7f847840ac18>], array([0., 0., 0., 0.]), [3 of B, 7 of B], [3 of B, 7 of B], 20, array([0., 0., 0., 0.])], 'player': 2, 'cards_away': 2, 'child': [], 'parent': (0, 0), 'n': 0, 'w': 0, 'q': 0},
(0, 0, 3): {'state': [[<gameClasses.player object at 0x7f847840c3c8>, <gameClasses.player object at 0x7f847840ca58>, <gameClasses.player object at 0x7f847840e198>, <gameClasses.player object at 0x7f847840e898>], array([0., 0., 0., 0.]), [3 of B, 8 of B], [3 of B, 8 of B], 20, array([0., 0., 0., 0.])], 'player': 2, 'cards_away': 3, 'child': [], 'parent': (0, 0), 'n': 0, 'w': 0, 'q': 0},
(0, 0, 4): {'state': [[<gameClasses.player object at 0x7f8478412048>, <gameClasses.player object at 0x7f84784126d8>, <gameClasses.player object at 0x7f8478412dd8>, <gameClasses.player object at 0x7f8478410518>], array([0., 0., 0., 0.]), [3 of B, 13 of B], [3 of B, 13 of B], 20, array([0., 0., 0., 0.])], 'player': 2, 'cards_away': 4, 'child': [], 'parent': (0, 0), 'n': 0, 'w': 0, 'q': 0},
(0, 0, 10): {'state': [[<gameClasses.player object at 0x7f8478410c88>, <gameClasses.player object at 0x7f8478416358>, <gameClasses.player object at 0x7f8478416a58>, <gameClasses.player object at 0x7f8478419198>], array([0., 0., 0., 0.]), [3 of B, J of R], [3 of B, J of R], 20, array([0., 0., 0., 0.])], 'player': 2, 'cards_away': 10, 'child': [], 'parent': (0, 0), 'n': 0, 'w': 0, 'q': 0},

(0, 1, 0): {'state': [[<gameClasses.player object at 0x7f8478419eb8>, <gameClasses.player object at 0x7f847841b748>, <gameClasses.player object at 0x7f847841c550>, <gameClasses.player object at 0x7f847841cc18>], array([0., 0., 0., 0.]), [9 of B, 1 of B], [9 of B, 1 of B], 20, array([0., 0., 0., 0.])], 'player': 2, 'cards_away': 0, 'child': [], 'parent': (0, 1), 'n': 0, 'w': 0, 'q': 0},
(0, 1, 1): {'state': [[<gameClasses.player object at 0x7f847841d3c8>, <gameClasses.player object at 0x7f847841da58>, <gameClasses.player object at 0x7f8478422198>, <gameClasses.player object at 0x7f8478422898>], array([0., 0., 0., 0.]), [9 of B, 2 of B], [9 of B, 2 of B], 20, array([0., 0., 0., 0.])], 'player': 2, 'cards_away': 1, 'child': [], 'parent': (0, 1), 'n': 0, 'w': 0, 'q': 0},
(0, 1, 2): {'state': [[<gameClasses.player object at 0x7f8478424048>, <gameClasses.player object at 0x7f84784246d8>, <gameClasses.player object at 0x7f8478424dd8>, <gameClasses.player object at 0x7f8478423518>], array([0., 0., 0., 0.]), [9 of B, 7 of B], [9 of B, 7 of B], 20, array([0., 0., 0., 0.])], 'player': 2, 'cards_away': 2, 'child': [], 'parent': (0, 1), 'n': 0, 'w': 0, 'q': 0},
(0, 1, 3): {'state': [[<gameClasses.player object at 0x7f8478423c88>, <gameClasses.player object at 0x7f8478427358>, <gameClasses.player object at 0x7f8478427a58>, <gameClasses.player object at 0x7f8478429198>], array([0., 0., 0., 0.]), [9 of B, 8 of B], [9 of B, 8 of B], 20, array([0., 0., 0., 0.])], 'player': 2, 'cards_away': 3, 'child': [], 'parent': (0, 1), 'n': 0, 'w': 0, 'q': 0},
(0, 1, 4): {'state': [[<gameClasses.player object at 0x7f8478429908>, <gameClasses.player object at 0x7f8478429f98>, <gameClasses.player object at 0x7f847842b6d8>, <gameClasses.player object at 0x7f847842bdd8>], array([0., 0., 0., 0.]), [9 of B, 13 of B], [9 of B, 13 of B], 20, array([0., 0., 0., 0.])], 'player': 2, 'cards_away': 4, 'child': [], 'parent': (0, 1), 'n': 0, 'w': 0, 'q': 0},
(0, 1, 10): {'state': [[<gameClasses.player object at 0x7f847842d588>, <gameClasses.player object at 0x7f847842dc18>, <gameClasses.player object at 0x7f8478430358>, <gameClasses.player object at 0x7f8478430a58>], array([0., 0., 0., 0.]), [9 of B, J of R], [9 of B, J of R], 20, array([0., 0., 0., 0.])], 'player': 2, 'cards_away': 10, 'child': [], 'parent': (0, 1), 'n': 1, 'w': 4.0, 'q': 4.0}}
```


# Output of **train.py** commit: *onnx_working*:
```
Read in samples:	18859
One sample:
[[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 57]
[1,   100] loss: 387.769
[1,   200] loss: 337.980
[1,   300] loss: 315.572
[1,   400] loss: 294.084
[1,   500] loss: 275.887
[1,   600] loss: 270.683
[1,   700] loss: 260.207
[1,   800] loss: 253.414
[1,   900] loss: 249.442
[1,  1000] loss: 238.414
[1,  1100] loss: 232.762
[1,  1200] loss: 229.282
[1,  1300] loss: 230.973
[1,  1400] loss: 226.616
[1,  1500] loss: 220.025
[1,  1600] loss: 211.258
[1,  1700] loss: 216.079
[1,  1800] loss: 206.454
tensor([1., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,
        0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 1., 1., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 1.,
        1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 0., 0., 1., 1., 1., 1., 1., 1.])
Finished Training in:	0:00:02.312103
I saved your model to:	data/model.pth
I now save your onnx model with parameters!
I will now check your onnx model using onnx
<onnxruntime.capi.session.InferenceSession object at 0x7f9873930da0>
[<onnxruntime.capi.onnxruntime_pybind11_state.NodeArg object at 0x7f98691a5bc8>]
input.1
I will now test your model!
[array([ -3.2607808,  -3.457503 ,  -7.639492 ,  -7.3476543,  -5.508501 ,
        -6.5215015,  -0.8601465,  -6.4172306,  -7.440096 ,  -2.6843185,
        -8.942799 ,  -5.1809483,  -4.9209766,  -7.719907 ,  -9.005304 ,
        -8.302288 ,  -7.1568375,  -3.3195906,  -8.95669  ,  -7.119462 ,
        -7.390052 ,  -7.781508 ,  -6.76344  ,  -3.5796459,  -6.1413436,
        -2.9167445,  -4.15365  ,  -8.423178 ,  -6.104857 ,  -5.539221 ,
        -4.0965657,  -6.5832458, -11.25923  ,  -5.9366293,  -8.30854  ,
        -8.426708 ,  -2.1296508,  -8.785681 ,  -9.701228 ,  -2.1758657,
        -7.4398003,  -7.3487854,  -7.1235924,  -6.95055  ,  -6.9943233,
        -6.5696526, -11.682266 , -11.028063 , -10.226396 , -10.814424 ,
       -10.284258 ,  -5.9688644,  -5.6975207,  -6.2563257,  -9.259622 ,
        -6.365409 ,  -7.2721806,  -8.853003 , -10.176397 , -10.104825 ],
      dtype=float32)]
train.py:137: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  input_vector = torch.tensor(input_vector[0]).float()
Outputs: using pytorch:
tensor([ -3.2608,  -3.4575,  -7.6395,  -7.3477,  -5.5085,  -6.5215,  -0.8601,
         -6.4172,  -7.4401,  -2.6843,  -8.9428,  -5.1809,  -4.9210,  -7.7199,
         -9.0053,  -8.3023,  -7.1568,  -3.3196,  -8.9567,  -7.1195,  -7.3901,
         -7.7815,  -6.7634,  -3.5796,  -6.1413,  -2.9167,  -4.1536,  -8.4232,
         -6.1049,  -5.5392,  -4.0966,  -6.5832, -11.2592,  -5.9366,  -8.3085,
         -8.4267,  -2.1297,  -8.7857,  -9.7012,  -2.1759,  -7.4398,  -7.3488,
         -7.1236,  -6.9506,  -6.9943,  -6.5697, -11.6823, -11.0281, -10.2264,
        -10.8144, -10.2843,  -5.9689,  -5.6975,  -6.2563,  -9.2596,  -6.3654,
         -7.2722,  -8.8530, -10.1764, -10.1048], grad_fn=<LogSoftmaxBackward>)
```





## Started **modules.reinforcement_learning.py** commit "beginning_reinforce"
* why not learning anything??
* tested with self.rewards, and discountedRewards
* change lr, momentum, gamma (for rewards)
* tested with -1*loss (no effect)
* how does the learning work in general?, when initializing the network params lost?
* see other reinforce examples:
  * Read [here](https://towardsdatascience.com/learning-reinforcement-learning-reinforce-with-pytorch-5e8ad7fc7da0) for a basic understanding!
    * The output of a DQN is going to be a vector of value estimates while the output of the policy gradient is going to be a probability distribution over actions.
  * https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
* Using a different discount function does not work as well!
* use only positive rewards! klappt auch nicht!!!
* Try using **another network!**
* wie ...
[0.8076923076923077, 0.8076923076923077, 0.8076923076923077, 0.8076923076923077, 0.8076923076923077, 0.8076923076923077, 0.6923076923076923, 0.8076923076923077, 0.8076923076923077, 0.8076923076923077, 0.8076923076923077, 0.8076923076923077, 0.8076923076923077, 0.8076923076923077, 0.8076923076923077]
[tensor(-1.4820, grad_fn=<SqueezeBackward1>), tensor(-2.7397, grad_fn=<SqueezeBackward1>), tensor(-1.5664, grad_fn=<SqueezeBackward1>), tensor(-1.0297, grad_fn=<SqueezeBackward1>), tensor(-2.2039, grad_fn=<SqueezeBackward1>), tensor(-2.0374, grad_fn=<SqueezeBackward1>), tensor(-0.8260, grad_fn=<SqueezeBackward1>), tensor(-1.8811, grad_fn=<SqueezeBackward1>), tensor(-0.8654, grad_fn=<SqueezeBackward1>), tensor(-0.4187, grad_fn=<SqueezeBackward1>), tensor(-0.6031, grad_fn=<SqueezeBackward1>), tensor(-1.4244, grad_fn=<SqueezeBackward1>), tensor(-0.8793, grad_fn=<SqueezeBackward1>), tensor(-0.8207, grad_fn=<SqueezeBackward1>), tensor(-1.1921e-07, grad_fn=<SqueezeBackward1>)]


[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.14285714285714285, -0.09523809523809523, -0.047619047619047616]
vorher:
[tensor([-2.6326], grad_fn=<SqueezeBackward1>), tensor([-1.0892], grad_fn=<SqueezeBackward1>), tensor([-2.5067], grad_fn=<SqueezeBackward1>), tensor([-2.5980], grad_fn=<SqueezeBackward1>), tensor([-2.2930], grad_fn=<SqueezeBackward1>), tensor([-2.2207], grad_fn=<SqueezeBackward1>), tensor([-2.1485], grad_fn=<SqueezeBackward1>), tensor([-2.0980], grad_fn=<SqueezeBackward1>), tensor([-0.6237], grad_fn=<SqueezeBackward1>), tensor([-1.1372], grad_fn=<SqueezeBackward1>), tensor([-0.7629], grad_fn=<SqueezeBackward1>), tensor([-1.1921e-07], grad_fn=<SqueezeBackward1>), tensor([-1.1921e-07], grad_fn=<SqueezeBackward1>), tensor([-0.8331], grad_fn=<SqueezeBackward1>), tensor([-1.1921e-07], grad_fn=<SqueezeBackward1>)]
nachher:
tensor([-2.6326e+00, -1.0892e+00, -2.5067e+00, -2.5980e+00, -2.2930e+00,
        -2.2207e+00, -2.1485e+00, -2.0980e+00, -6.2368e-01, -1.1372e+00,
        -7.6289e-01, -1.1921e-07, -1.1921e-07, -8.3311e-01, -1.1921e-07],
       grad_fn=<CatBackward>)

* Does not work as well!!!

* Question on forum
  * How do I know that my algorithm learns something?
  * How to setup the network?
  * What am I missing?
  * What shape should losses have (15x15 matrix?)
  * see [here](https://discuss.pytorch.org/t/reinforce-for-a-multiplayer-game/73207)
  * geht hiermit noch aktuell am besten:         self.optimizer = optim.SGD(self.parameters(), lr=0.1)
  * bestes ergebnis: game finished with::: [-295. -663. -729. -716.]
  * **Should I collect batches????!!!**
  * Problem ist dass  
  * invalid multinomial distribution (encountering probability entry < 0)
  * # clipping to prevent nans:
    # see https://discuss.pytorch.org/t/proper-way-to-do-gradient-clipping/191/6
    torch.nn.utils.clip_grad_norm_(self.parameters(), 5)

* Check Game Logic:
  * ai player plays valid cards!
