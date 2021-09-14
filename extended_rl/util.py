def run_environment(env, A, num_steps, logfile=None):
    if logfile:
        env, A = add_log_messages(env, A, logfile)

    step = 0
    results = {'total_reward': 0.0}

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

def annotate(
    num_legal_actions,
    num_possible_obs,
    slow=False
):
    def apply_annotations(env_class):
        env_class.num_legal_actions = num_legal_actions
        env_class.num_possible_obs = num_possible_obs
        env_class.slow = slow
        return env_class

    return apply_annotations

def copy_with_meta(class_to_copy, meta_src):
    @annotate(
        num_legal_actions=meta_src.num_legal_actions,
        num_possible_obs=meta_src.num_possible_obs
    )
    class result(class_to_copy):
        pass

    result.__name__ = class_to_copy.__name__
    result.__qualname__ = class_to_copy.__qualname__
    return result

def args_to_agent(A, **kwargs_outer):
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

    @annotate(
        num_legal_actions=env.num_legal_actions,
        num_possible_obs=env.num_possible_obs
    )
    class e:
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