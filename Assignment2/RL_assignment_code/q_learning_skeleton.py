import numpy as np
import random
NUM_EPISODES = 1000
MAX_EPISODE_LENGTH = 500


DEFAULT_DISCOUNT = 0.9
EPSILON = 0.05
LEARNINGRATE = 0.1




class QLearner():
    """
    Q-learning agent
    """
    def __init__(self, num_states, num_actions, discount=DEFAULT_DISCOUNT, learning_rate=LEARNINGRATE):
        self.name = "agent1"
        self.discount = discount
        self.learning_rate = learning_rate
        self.num_states = num_states
        self.num_actions = num_actions

        self.q = np.zeros((num_states,num_actions))
        
    # TODO: Select some interesting statistics to keep track of
    # Maybe total reward of the episode
    def reset_episode(self):
        """
        Here you can update some of the statistics that could be helpful to maintain
        """
        pass


    def process_experience(self, state, action, next_state, reward, done): 
        """
        Update the Q-value based on the state, action, next state and reward.
        """

        update = (1-self.learning_rate)*self.q[state,action] + self.learning_rate*reward

        if not done:
            update += self.learning_rate*self.discount*np.max(self.q[next_state][:])
        
        self.q[state,action] = update



    # NOTE: At the beginning, when all values are 0, the max value selected is always the first
    # Therefore, the agent almost always tries to go left, and it does not learn properly
    # Possible Solutions:
    # - Set the SLEEP_DEPRIVATION_PENALTY in simple_grid.py to a value less than 0 (originally 0, currently set to -0.1)
    # - Select randomly if all values are the same
    # - Avoid actions that takes us out of the border (for example, not go up or left in the first cell) I don't think this is the intended solution
    # This should probably be taken into account when answering the first questions, and they probably
    # want us to fix this with one of those solutions (probably the first)
    # However I'm not sure, maybe we should ask on Answers EWI
    def select_action(self, state): 
        selection = random.random()
        if selection <= EPSILON:
            return random.randrange(self.num_actions)
        else:
            return np.argmax(self.q[state][:])

    # I cannot come up with something to print
    def report(self):
        """
        Function to print useful information, printed during the main loop
        """
        print("---")

