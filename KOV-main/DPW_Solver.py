import math
import random
import time

def estimate_value_fun(mdp, state, depth):
    """
    Estimates the value at a leaf node during MCTS.

    Parameters:
        mdp: The Markov Decision Process (e.g., your WhiteBoxMDP instance).
        state: The current state (e.g., WhiteBoxState instance).
        depth (int): The current depth in the tree.

    Returns:
        A float representing the estimated value of the state.
    """
    loss = mdp.loss(state)
    # Adjust the estimated value based on depth (optional)
    adjusted_value = -loss / (depth + 1)
    return adjusted_value



class DPWSolver:
    """
    MCTS solver with Double Progressive Widening (DPW).

    Parameters:
        depth (int): Maximum rollout horizon and tree depth.
        exploration_constant (float): Exploration constant for the UCB equation.
        n_iterations (int): Number of iterations during each solve() call.
        max_time (float): Maximum amount of time (in seconds) to spend iterating.
        k_action (float): Controls action progressive widening.
        alpha_action (float): Controls action progressive widening.
        k_state (float): Controls state progressive widening.
        alpha_state (float): Controls state progressive widening.
        keep_tree (bool): If True, keeps the tree between calls to solve().
        enable_action_pw (bool): If True, enables action progressive widening.
        check_repeat_state (bool): If True, checks for repeated states.
        check_repeat_action (bool): If True, checks for repeated actions.
        tree_in_info (bool): If True, returns the tree in the info dict when action_info is called.
        rng (random.Random): Random number generator.
        estimate_value (callable): Function to estimate the value at leaf nodes.
        init_Q (float): Initial Q(s,a) value at new nodes.
        init_N (int): Initial N(s,a) value at new nodes.
        next_action (callable): Function to choose the next action for progressive widening.
        default_action (callable): Function to determine the action if MCTS fails.
    """

    def __init__(self, depth=10, exploration_constant=1.0, n_iterations=100, max_time=float('inf'),
             k_action=10.0, alpha_action=0.5, k_state=10.0, alpha_state=0.5,
             keep_tree=False, enable_action_pw=True, check_repeat_state=True,
             check_repeat_action=True, tree_in_info=False, rng=None,
             estimate_value=None, init_Q=0.0, init_N=0, next_action=None,
             default_action=None):
        # Initialize fields
        self.depth = depth
        self.exploration_constant = exploration_constant
        self.n_iterations = n_iterations
        self.max_time = max_time
        self.k_action = k_action
        self.alpha_action = alpha_action
        self.k_state = k_state
        self.alpha_state = alpha_state
        self.keep_tree = keep_tree
        self.enable_action_pw = enable_action_pw
        self.check_repeat_state = check_repeat_state
        self.check_repeat_action = check_repeat_action
        self.tree_in_info = tree_in_info
        self.rng = rng if rng is not None else random.Random()
        self.estimate_value = estimate_value if estimate_value is not None else estimate_value_fun
        self.init_Q = init_Q
        self.init_N = init_N
        self.next_action = next_action if next_action is not None else self.random_action_generator
        self.default_action = default_action if default_action is not None else self.exception_rethrow
        # Initialize tree
        self.tree = None

    def search(self, mdp, state):
        """
        Performs MCTS with DPW starting from the given state in the MDP.

        Parameters:
            mdp: The Markov Decision Process.
            state: The current state from which to start planning.

        Returns:
            The best action found by the planner.
        """
        if not self.keep_tree or self.tree is None:
            self.tree = DPWTree()
        root_node = self.get_or_create_state_node(state)
        start_time = time.time()

        for _ in range(self.n_iterations):
            if time.time() - start_time > self.max_time:
                break
            self.simulate(mdp, root_node, state, depth=0)

        best_action = self.best_action(root_node)
        return best_action

    def simulate(self, mdp, node, state, depth):
        """
        Simulates a trajectory from the given node.

        Parameters:
            mdp: The Markov Decision Process.
            node (DPWStateNode): The current node in the tree.
            state: The current state in the MDP.
            depth (int): The current depth in the tree.

        Returns:
            The estimated value of the node.
        """
        if depth >= self.depth:
            # Return value estimate at the leaf node
            return self.estimate_value(mdp, state, depth)

        # Correctly access total_n
        node.tree.total_n[node.index] += 1

        # Progressive widening on actions
        node_visit_count = node.tree.total_n[node.index]
        if self.enable_action_pw and len(node.children()) <= self.k_action * node_visit_count ** self.alpha_action:
            # Add a new action
            action = self.next_action(mdp, state, node)
            if action is None:
                return 0  # No valid action
            action_node = self.get_or_create_action_node(node, action)
        else:
            # Select action using UCB
            action_node = self.select_action_node(node)
            if action_node is None:
                return 0  # No valid action
            action = action_node.action_label()

        # Simulate the action
        next_state, reward = mdp.gen(state, action)
        action_node.tree.n[action_node.index] += 1
        action_node.tree.q[action_node.index] += reward

        # Progressive widening on states
        action_visit_count = action_node.tree.n[action_node.index]
        if len(action_node.transitions()) <= self.k_state * action_visit_count ** self.alpha_state:
            # Add new state node
            next_node_index = self.insert_state_node(self.tree, next_state)
            action_node.tree.transitions[action_node.index].append((next_node_index, reward))
            next_node = DPWStateNode(self.tree, next_node_index)
        else:
            # Select a next state from existing transitions
            next_node_index = self.rng.choice([t[0] for t in action_node.transitions()])
            next_node = DPWStateNode(self.tree, next_node_index)

        # Recursively simulate from the next node
        value = reward + mdp.discount() * self.simulate(mdp, next_node, next_state, depth + 1)

        # Backpropagate
        action_node.tree.q[action_node.index] += value

        return value

    def get_or_create_state_node(self, state):
        """
        Retrieves or creates a state node for the given state.

        Parameters:
            state: The state to retrieve or create a node for.

        Returns:
            A DPWStateNode corresponding to the state.
        """
        if state in self.tree.s_lookup:
            index = self.tree.s_lookup[state]
        else:
            index = self.insert_state_node(self.tree, state)
        return DPWStateNode(self.tree, index)

    def get_or_create_action_node(self, state_node, action):
        """
        Retrieves or creates an action node for the given action.

        Parameters:
            state_node (DPWStateNode): The parent state node.
            action: The action to retrieve or create a node for.

        Returns:
            A DPWActionNode corresponding to the action.
        """
        key = (state_node.index, action)
        if key in self.tree.a_lookup:
            index = self.tree.a_lookup[key]
        else:
            index = self.insert_action_node(self.tree, state_node.index, action, self.init_N, self.init_Q)
        return DPWActionNode(self.tree, index)

    def insert_state_node(self, tree, state):
        tree.total_n.append(0)
        tree.children.append([])
        tree.s_labels.append(state)
        index = len(tree.total_n) - 1
        if self.check_repeat_state:
            tree.s_lookup[state] = index
        return index

    def insert_action_node(self, tree, snode_index, action, n0, q0):
        tree.n.append(n0)
        tree.q.append(q0)
        tree.a_labels.append(action)
        tree.transitions.append([])
        index = len(tree.n) - 1
        tree.children[snode_index].append(index)
        tree.n_a_children.append(0)
        if self.check_repeat_action:
            tree.a_lookup[(snode_index, action)] = index
        return index

    def select_action_node(self, node):
        """
        Selects an action node using the UCB formula.

        Parameters:
            node (DPWStateNode): The parent state node.

        Returns:
            A DPWActionNode representing the selected action.
        """
        best_value = float('-inf')
        best_node = None
        for child_index in node.children():
            n = node.tree.n[child_index]
            q = node.tree.q[child_index]
            if n == 0:
                continue  # Avoid division by zero
            ucb_value = q / n + self.exploration_constant * math.sqrt(
                math.log(node.tree.total_n[node.index]) / n
            )
            if ucb_value > best_value:
                best_value = ucb_value
                best_node = DPWActionNode(node.tree, child_index)
        return best_node

    def best_action(self, root_node):
        """
        Returns the action with the highest average reward from the root node.

        Parameters:
            root_node (DPWStateNode): The root node of the tree.

        Returns:
            The best action found.
        """
        best_value = float('-inf')
        best_action = None
        for child_index in root_node.children():
            n = self.tree.n[child_index]
            q = self.tree.q[child_index]
            if n == 0:
                continue
            avg_reward = q / n
            if avg_reward > best_value:
                best_value = avg_reward
                best_action = self.tree.a_labels[child_index]
        return best_action

    # Placeholder methods
    def default_estimate_value(self, mdp, state, depth):
        """
        Default estimate value function.

        Parameters:
            mdp: The Markov Decision Process.
            state: The current state.
            depth (int): The depth in the tree.

        Returns:
            A default estimated value (0.0).
        """
        return 0.0

    def random_action_generator(self, mdp, state, node):
        """
        Generates a random action.

        Parameters:
            mdp: The Markov Decision Process.
            state: The current state.
            node (DPWStateNode): The current node.

        Returns:
            A randomly selected action.
        """
        actions = mdp.actions(state)
        if not actions:
            return None
        return self.rng.choice(actions)

    def exception_rethrow(self):
        """
        Default action when MCTS fails.

        Raises:
            Exception indicating MCTS failed to find a valid action.
        """
        raise Exception("MCTS failed to find a valid action")

class DPWTree:
    """
    Data structure for the MCTS tree with DPW.
    """

    def __init__(self, sz=1000):
        sz = min(sz, 100_000)
        # For each state node
        self.total_n = []  # Total visit count for each state node
        self.children = []  # Children of each state node (indices of action nodes)
        self.s_labels = []  # State labels
        self.s_lookup = {}  # Lookup from state to index

        # For each action node
        self.n = []  # Visit count for each action node
        self.q = []  # Total value for each action node
        self.transitions = []  # Transitions from each action node (list of (state_index, reward))
        self.a_labels = []  # Action labels
        self.a_lookup = {}  # Lookup from (state_index, action) to action node index

        # For tracking transitions
        self.n_a_children = []  # Number of child states for each action node
        self.unique_transitions = set()  # Set of (action_node_index, state_node_index)

class DPWStateNode:
    """
    Represents a state node in the DPW tree.
    """

    def __init__(self, tree, index):
        self.tree = tree
        self.index = index

    def children(self):
        return self.tree.children[self.index]

    def n_children(self):
        return len(self.children())

    def is_root(self):
        return self.index == 0  # Assuming root is at index 0

    def increment_visit(self):
        self.tree.total_n[self.index] += 1

    def visit_count(self):
        return self.tree.total_n[self.index]
class DPWActionNode:
    """
    Represents an action node in the DPW tree.
    """

    def __init__(self, tree, index):
        self.tree = tree
        self.index = index

    def transitions(self):
        return self.tree.transitions[self.index]

    def n_transitions(self):
        return len(self.transitions())

    def action_label(self):
        return self.tree.a_labels[self.index]
