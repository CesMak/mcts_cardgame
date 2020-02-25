import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

class policy(object):
	def __init__(self):
		self.tree = {}
		pass

class VanilaMCTS(object):
	def __init__(self, n_iterations=50, depth=15, exploration_constant=5.0, tree = None, win_mark=3, state=None, player=None, game=None):
		#print("MCTS, iterations:", n_iterations, "depth:", depth, "exploration:", exploration_constant)
		self.n_iterations 		  = n_iterations
		self.depth 				  = depth
		self.exploration_constant = exploration_constant
		self.total_n 			  = 0
		self.game 				  = game
		self.player 			  = player # the ai player

		self.leaf_node_id = None

		if tree == None:
			self.tree = self._set_witches(state, player)
		else:
			self.tree = tree

	def _set_witches(self, state, player):
		root_id = (0,)
		#print("\nSet state:\n", state, "\nplayer:", player)
		tree = {root_id: {'state': state,
						  'player': player,
						  'cards_away': [],
						  'child': [],
						  'parent': None,
						  'n': 0,
						  'w': 0,
						  'q': None}}
		return tree

	def selection(self):
		'''
		select leaf node which have maximum uct value
		in:
		- tree
		out:
		- leaf node id (node to expand)
		- depth (depth of node root=0)
		'''
		leaf_node_found = False
		leaf_node_id = (0,) # root node id
		print('\n-------- selection ----------')

		while not leaf_node_found:
			node_id = leaf_node_id
			n_child = len(self.tree[node_id]['child'])
			if n_child == 0:
				leaf_node_id = node_id
				leaf_node_found = True
			else:
				maximum_uct_value = -100.0
				for i in range(n_child):
					action = self.tree[node_id]['child'][i]

					child_id = node_id + (action,)
					w = self.tree[child_id]['w']
					n = self.tree[child_id]['n'] #number of visits
					total_n = self.total_n # total number of rollouts
					if n == 0:
						n = 1e-4
					exploitation_value = w / n
					exploration_value  = np.sqrt(np.log(total_n)/n)# before np.sqrt(np.log(total_n)/n)
					uct_value = exploitation_value + self.exploration_constant * exploration_value

					if uct_value > maximum_uct_value:
						maximum_uct_value = uct_value
						leaf_node_id = child_id

		depth = len(leaf_node_id) # as node_id records selected action set
		# print('leaf node found: ', leaf_node_found)
		# print('n_child: ', n_child)
		# print('selected leaf node: ')
		# print(self.tree[leaf_node_id])
		print("selected:", leaf_node_id, "to expand at depth:", depth)
		return leaf_node_id, depth

	def expansion(self, leaf_node_id):
		'''
		create all possible outcomes from leaf node
		in: tree, leaf_node
		out: expanded tree (self.tree),
			 randomly selected child node id (child_node_id)
		'''
		print('\n-------- expansion ----------')
		leaf_state     = (self.tree[leaf_node_id]['state'])
		current_player = (self.tree[leaf_node_id]['player'])
		self.game.setState(leaf_state+[current_player])

		rewards = self.game.isGameFinished()

		shifting_phase = True if self.game.shifting_phase<self.game.nu_players else False
		possible_actions = []
		if shifting_phase:
			print("Now in shifting phase!!!")
			possible_actions = (self.game.getShiftOptions())
		else:
			possible_actions = self.game.getValidOptions(current_player)

		print(possible_actions, len(self.game.players[current_player].hand))
		child_node_id = leaf_node_id # default value
		if rewards is None:
			'''
			when leaf state is not terminal state
			'''
			childs = []
			for uuu, action_set in enumerate(possible_actions):
				self.game.setState(deepcopy(leaf_state)+[current_player])
				self.game.step_idx(action_set)
				tmp = action_set
				if isinstance(action_set, list):
					action_set = uuu#int(str(action_set[0])+str(action_set[1]))
				state = self.game.getGameState()
				child_id = leaf_node_id + (action_set, )
				childs.append(child_id)
				self.tree[child_id] = {'state': state,
									   'player': self.game.active_player,
									   'cards_away': tmp,
									   'child': [],
									   'parent': leaf_node_id,
									   'n': 0, 'w': 0, 'q':0}
				self.tree[leaf_node_id]['child'].append(action_set)
			rand_idx = np.random.randint(low=0, high=len(childs), size=1)
			print('childs: ', childs)
			print("lenght childs", len(childs))
			print("state", state)
			print("expand now random index:", rand_idx)
			child_node_id = childs[rand_idx[0]]
		return child_node_id

	def simulation(self, child_node_id):
		'''
		simulate game from child node's state until it reaches the resulting state of the game.
		in:
		- child node id (randomly selected child node id from `expansion`)
		out:
		- rewards (after game finished)
		'''
		print('\n-------- simulation ----------')
		self.total_n += 1
		state = deepcopy(self.tree[child_node_id]['state'])
		previous_player = deepcopy(self.tree[child_node_id]['player'])
		anybody_win = False
		self.game.setState(state+[previous_player])

		while not anybody_win:
			#TODO do not random steps here! but steps with NN
			#rewards = self.game.NNStep()
			rewards = self.game.randomStep()
			if rewards is not None:
				anybody_win = True
		print("Simulation Result:", rewards)
		return rewards

	def backprop(self, child_node_id, rewards):
		print('\n-------- backprob ----------')
		player = deepcopy(self.tree[(0,)]['player'])
		reward = rewards[player]

		finish_backprob = False
		node_id = child_node_id
		while not finish_backprob:
			self.tree[node_id]['n'] += 1
			self.tree[node_id]['w'] += reward
			print("Added reward", reward, "to w of", node_id)
			self.tree[node_id]['q'] = self.tree[node_id]['w'] / self.tree[node_id]['n']
			parent_id = self.tree[node_id]['parent']
			if parent_id == (0,):
				self.tree[parent_id]['n'] += 1
				self.tree[parent_id]['w'] += reward
				print("Added reward", reward, "to w of", parent_id)
				self.tree[parent_id]['q'] = self.tree[parent_id]['w'] / self.tree[parent_id]['n']
				finish_backprob = True
			else:
				node_id = parent_id

	def solve(self):
		for i in range(self.n_iterations):
			leaf_node_id, depth_searched = self.selection()
			child_node_id = self.expansion(leaf_node_id)
			rewards = self.simulation(child_node_id)
			self.backprop(child_node_id, rewards)

			print('\n-------- solve ----------')
			print('iter: %d, depth: %d' % (i, depth_searched))
			print('leaf_node_id: ', leaf_node_id)
			print('child_node_id: ', child_node_id)
			print('child node: ')
			print(self.tree[child_node_id])
			if depth_searched > self.depth:
				break

		# SELECT BEST ACTION
		current_state_node_id = (0,)
		action_candidates = self.tree[current_state_node_id]['child']
		# qs = [self.tree[(0,)+(a,)]['q'] for a in action_candidates]
		best_q = -100
		for a in action_candidates:
			q = self.tree[(0,)+(a,)]['q']
			if q > best_q:
				best_q = q
				best_action = a

		# FOR DEBUGGING
		state = self.tree[(0,)]['state']
		print('\n-------Finished Solve---------------')
		print(state)
		print('person to play: ', self.tree[(0,)]['player'])
		# Case1: Shifting result (return 2 card idx)


		if isinstance(self.tree[(0, best_action)]["cards_away"], list):
			print("you are in round 0 --> give back the cards to give away!")
			print("Best action:", best_action)
			cards_to_give_away = self.tree[(0, best_action)]["cards_away"]
			print("I would give away these cards:", cards_to_give_away)
			hand_before = state[0][self.tree[(0,)]['player']].hand
			print("Of these cards\n", hand_before)
			print("I would give away:")
			for card in cards_to_give_away:
				print(hand_before[card])
			return cards_to_give_away, best_q, depth_searched
		else: # Case 2: Play a card just one index to give back:
			print('\nbest_action : %d' % best_action, "which is card:", state[0][self.tree[(0,)]['player']].hand[best_action])
			print('best_q = %.2f' % (best_q))
			print('searching depth = %d' % (depth_searched))
			#print(self.tree)
			return best_action, best_q, depth_searched


'''
for test
'''
# if __name__ == '__main__':
#     mcts = VanilaMCTS(n_iterations=100, depth=10, exploration_constant=1.4, tree = None, n_rows=3, win_mark=3)
#     # leaf_node_id, depth = mcts.selection()
#     # child_node_id = mcts.expansion(leaf_node_id)
#     #
#     # print('child node id = ', child_node_id)
#     # print(' [*] simulation ...')
#     # winner = mcts.simulation(child_node_id)
#     # print(' winner', winner)
#     # mcts.backprop(child_node_id, winner)
#     best_action, max_q = mcts.solve()
#     print('best action= ', best_action, ' max_q= ', max_q)
