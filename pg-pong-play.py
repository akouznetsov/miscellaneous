""" Play Pong with saved weights, no learning. Uses OpenAI Gym.
    Adapted from Andre Karpathy's pg-pong.py mentioned in his blog
    - http://karpathy.github.io/2016/05/31/rl/
    - https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
"""
import numpy as np
import cPickle as pickle
import gym
import time

# hyperparameters
H = 200 # number of hidden layer neurons
gamma = 0.99 # discount factor for reward

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
model = pickle.load(open('saveM.p', 'rb'))
print('Loaded saved model')
  
def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
reward_sum = 0
episode_number = 0
last = time.time()
while True:
  try:
    env.render()

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

    y = 1 if action == 2 else 0 # a "fake label"

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    if done: # an episode finished
      episode_number += 1

      end = time.time()
      print('episode: %d, wins: %f. time: %d' %
            (episode_number, 21+reward_sum, (end-last)))
      last = end
      if episode_number % 5 == 0:
        model = pickle.load(open('saveM.p', 'rb'))
        print('reloaded model')
      reward_sum = 0
      observation = env.reset() # reset env
      prev_x = None

  except KeyboardInterrupt:
    break

print('Done.')
