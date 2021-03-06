""" Trains an agent with (stochastic) Policy Gradients on Pong using multiple
    games in parallel. Uses OpenAI Gym. 
    Adapted from Andre Karpathy's pg-pong.py mentioned in his blog
    - http://karpathy.github.io/2016/05/31/rl/
    - https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
    """
import numpy as np
import cPickle as pickle
import gym, time, copy, argparse, threading, sys

# hyperparameters
H = 200 # number of hidden layer neurons
D = 80 * 80 # input dimensionality: 80x80 grid
learning_rate = 1e-3
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
total_wins = 0.0
episodes = 0
grad_buffer = {}
rmsprop_cache = {}
model = {}

# Parameter class
class Args(object):
    threads = 8
    batch_size = 20
    paramsFile = 'saveM.p'

args = Args()
env = gym.make("Pong-v0")
lock = threading.RLock()

def init_model():
    global model, grad_buffer, rmsprop_cache, H, D
    # model initialization
    try:
      model = pickle.load(open(args.paramsFile, 'rb'))
      print('Loaded saved model')
    except:
      model = {}
      model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
      model['W2'] = np.random.randn(H) / np.sqrt(H)
      print('Created new model')      
      
    grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
    rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory

def save_model():
    pickle.dump(model, open(args.paramsFile, 'wb'))
    print('saved model parameters')

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
  global model
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(epx, eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

# For accelerating learning, we want to run the game in multiple processes
# Then, take the back prop and add them up from all processes and apply to model 
def one_run(env, idx):
  """ Do one game set of play and learn from it """
  global grad_buffer, total_wins, model, episodes, lock
  while True:
        observation = env.reset()
        prev_x = None # used in computing the difference frame
        xs,hs,dlogps,drs = [],[],[],[]
        reward_sum = 0
        done = False
        #print('Started thread %d' % idx)
        while not done:
          # preprocess the observation, set input to network to be difference image
          cur_x = prepro(observation)
          x = cur_x - prev_x if prev_x is not None else np.zeros(D)
          prev_x = cur_x

          # forward the policy network and sample an action from the returned probability
          aprob, h = policy_forward(x)
          action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

          # record various intermediates (needed later for backprop)
          xs.append(x) # observation
          hs.append(h) # hidden state
          y = 1 if action == 2 else 0 # a "fake label"
          dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

          # step the environment and get new measurements
          observation, reward, done, info = env.step(action)
          reward_sum += reward

          drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
          
        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(epx, eph, epdlogp)
        lock.acquire(True)
        for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch
        total_wins += 21.0 + reward_sum
        episodes += 1
        lock.release()
        #print('Done with thread %d' % idx)
  
def sum_and_back_prop():
    global last, episodes, lock, model, grad_buffer, rmsprop_cache, total_wins
    last_episode_processed = 0
    while True:
        lock.acquire(True)
        if episodes > last_episode_processed and (episodes % args.batch_size == 0):
            last_episode_processed = episodes
            for k,v in model.iteritems():
              g = grad_buffer[k] # gradient
              rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
              model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
              grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

            save_model()
            # boring book-keeping
            end = time.time()
            print('episodes: %d, Av wins: %.2f, Ave time: %f ******' %
                  (episodes, total_wins/args.batch_size, (end-last)/args.batch_size))
            total_wins = 0.0
            last = end
            
        lock.release()        

def train_agent():
    global env, last
    init_model()
    last = time.time()
    thread_list = []
    try:
        #print('*************Start %d threads' % args.threads)
        for i in range(args.threads):
            t = threading.Thread(target=one_run, args=(copy.deepcopy(env),i))
            t.setDaemon(True)
            thread_list.append(t)
        
        # Run this thread to sum the values
        t = threading.Thread(target=sum_and_back_prop)
        t.setDaemon(True)
        thread_list.append(t)
        for t in thread_list:
          t.start()
          
        while True:
          time.sleep(1)
          
    except KeyboardInterrupt:
        lock.release()
        save_model()

    print('Done.')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an agent to play pong')
    parser.add_argument('--threads', '-t', default=8, type=int,
                        help='Threads to run in parallel')
    parser.add_argument('--batch_size', '-b', default=20, type=int,
                        help='How many games to play per thread')
    parser.add_argument('--paramsFile', '-p', default="saveM.p", 
                        help='Parameter file to read/write from')
    parser.parse_args(sys.argv[1:], args)
    print('Param File is %s' % args.paramsFile)
    train_agent()


