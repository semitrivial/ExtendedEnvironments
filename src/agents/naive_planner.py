from util import memoize

def naive_planner(A):
    @memoize
    def hallucinate(prompt, *meta, **kwargs):
        n_legal_actions, n_possible_obs = meta

        if len(prompt) < 3:
            true_reward, true_obs = prompt
            h = (reward,)
            for obs in range(n_possible_obs):
                h += (obs,)
                h += (A(h, *meta, **kwargs), 0)
            h += (true_obs,)
            return h

        prev = prompt[:-3]
        true_action, true_reward, true_obs = prompt[-3:]

        h = hallucinate(prev, *meta, **kwargs)
        h += (A(h, *meta, **kwargs))
        h += (true_reward,)

        for obs in range(n_possible_obs):
            h += (obs,)
            h += (A(h, *meta, **kwargs), 0)

        h += (true_obs,)
        return h

    def A_modified(prompt, *meta, **kwargs):
        n_legal_actions, n_possible_obs = meta

        h = hallucinate(prompt, *meta, **kwargs)
        obs = prompt[-1]
        planned_action = h[3*(obs-n_possible_obs)]
        return planned_action

    return A_modified