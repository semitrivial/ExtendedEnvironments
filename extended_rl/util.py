def run_environment(env, A, num_steps, logfile=None):
    """
    Given an environment-class 'env' and an agent-class 'A',
    run an instance of A in an instance of env for the
    indicated number of steps.

    The instance of 'env' will be passed A itself (not the
    instance of A). This allows the environment to generate
    independent clones of the agent, in order to inspect the
    agent's hypothetical behavior without calling the agent's
    own functions (as calling the agent's own functions might
    inadvertently alter the true agent).

    Currently this function just returns a dictionary with the
    total reward the agent gets from the environment.
    """
    if logfile:
        env, A = add_log_messages(env, A, logfile)

    step = 0
    results = {'total_reward': 0.0}

    # Actually, to be precise, A itself is not passed to env,
    # but rather, a copy of A with appropriate metadata set.
    # The purpose of this is to allow environment sourcecode to
    # just directly call the agent-class without repetitively
    # specifying the metadata every time.
    A_with_meta = copy_with_meta(A, meta_src=env)

    env = env(A_with_meta)

    A_instance = A_with_meta()

    o = env.start()
    while step < num_steps:
        action = A_instance.act(obs=o)
        reward, o_next = env.step(action)
        A_instance.train(o_prev=o, a=action, r=reward, o_next=o_next)
        o = o_next
        results['total_reward'] += reward
        step += 1

    return results

def copy_with_meta(class_to_copy, meta_src):
    """
    Return a copy of "class_to_copy" but with metadata
    (n_actions and n_obs) copied from meta_src. Usually
    class_to_copy is an agent-class and meta_src is an
    environment-class in whose instances instances of
    class_to_copy are to be run.
    """
    class result(class_to_copy):
        n_actions, n_obs = meta_src.n_actions, meta_src.n_obs

    result.__name__ = class_to_copy.__name__
    result.__qualname__ = class_to_copy.__qualname__
    return result

def args_to_agent(A, **kwargs_outer):
    """
    Given an agent-class A and some keyword arguments, create
    a new agent-class identical to A except that said keyword
    arguments are always passed to it (in addition to any other
    keyword arguments which are manually passed to it).
    """
    class A_with_args:
        def __init__(self, **kwargs_inner):
            self.kwargs = dict(kwargs_outer, **kwargs_inner)
        def act(self, obs):
            A_with_meta = copy_with_meta(A, self)
            self.underlying = A_with_meta(**self.kwargs)
            self.act = self.underlying.act
            self.train = self.underlying.train
            return self.act(obs)

    A_with_args.__name__ = A.__name__
    A_with_args.__qualname__ = A.__qualname__
    return A_with_args

def add_log_messages(env, A, logfile):
    if logfile.tell() == 0:
        logfile.write("agent,environment,message\n")

    A_sim_counter = [1]

    log_prefix = f"{A.__name__},{env.__name__}"

    def log(msg):
        logfile.write(f"{log_prefix},{msg}\n")

    class e:
        n_actions, n_obs = env.n_actions, env.n_obs
        def __init__(self, A0):
            sim_A = copy_with_meta(a_sim, A0)
            self.underlying = env(sim_A)
        def start(self):
            obs = self.underlying.start()
            log(f"Initial obs {obs}")
            return obs
        def step(self, action):
            reward, obs = self.underlying.step(action)
            log(f"Reward {reward}")
            log(f"Obs {obs}")
            return (reward, obs)

    class a_true:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def act(self, obs):
            A_with_meta = copy_with_meta(A, self)
            self.underlying = A_with_meta(**self.kwargs)
            self.act = self._act
            return self.act(obs)
        def _act(self, obs):
            action = self.underlying.act(obs)
            log(f"Action {action}")
            return action
        def train(self, o_prev, a, r, o_next):
            training_data = f"{(o_prev, a, r, o_next)}"
            log(f"Agent trained on {training_data}")
            self.underlying.train(o_prev=o_prev, a=a, r=r, o_next=o_next)

    class a_sim:
        def __init__(self, **kwargs):
            self.serial_number = A_sim_counter[0]
            A_sim_counter[0] += 1
            self.name = f"Sim_{self.serial_number}"
            self.kwargs = kwargs
        def act(self, obs):
            A_with_meta = copy_with_meta(A, self)
            self.underlying = A_with_meta(**self.kwargs)
            self.act = self._act
            return self.act(obs)
        def _act(self, obs):
            action = self.underlying.act(obs)
            log(f"Env queried {self.name} with obs {obs}")
            log(f"{self.name} replied with action {action}")
            return action
        def train(self, o_prev, a, r, o_next):
            training_data = f"{(o_prev, a, r, o_next)}"
            log(f"Env fed {self.name} training-data {training_data}")
            self.underlying.train(o_prev=o_prev, a=a, r=r, o_next=o_next)

    a_true.__name__ = A.__name__
    a_true.__qualname__ = A.__qualname__
    a_sim.__name__ = A.__name__
    a_sim.__qualname__ = A.__qualname__
    e.__name__ = env.__name__
    e.__qualname__ = env.__qualname__

    return (e, a_true)

def eval_and_count_steps(str, local_vars):
    # Count how many steps a string of code takes to execute, as measured
    # by the python debugger, pdb. This function works by hijacking pdb.
    # Returns both the result of the underlying code being executed, and
    # the number of steps the execution required.
    stepcount = [0]

    # import pdb here instead of at the top of util.py, so that users who
    # do not use the RuntimeInspector environment will not depend on pdb
    from pdb import Pdb

    # Mock a pdb interface in which the "user" blindly always chooses to
    # "take 1 step" and all outputs from pdb are ignored.
    class consolemock:
        def readline(self):
            stepcount[0] += 1  # Keep track of how many steps go by
            return "s"  # "take 1 step"
        def write(self, *args):
            return
        def flush(self):
            return

    # Execute the given code using the above-mocked interface.
    runner = Pdb(stdin=consolemock(), stdout=consolemock())
    result = runner.runeval(str, locals = local_vars)

    return result, stepcount[0]