def reality_check(T0):
    """
    Given an agent T0, output an agent T which is the "reality check" of T0.
    What this means is, on any given prompt, T will first inspect the prompt
    and see whether any actions taken in the prompt are NOT the actions which
    T0 would take in those situations. If so, then T freezes up from there
    on, always repeating the first action it took. But if all actions in the
    prompt are actions which T0 would have taken in those situations, then T
    acts as T0 would act.
    """
    def T(prompt, num_legal_actions, num_possible_obs):
        for i in range(len(prompt)//3):
            subprompt = prompt[:3*i+2]
            subaction = prompt[3*i+2]
            expected = T0(subprompt, num_legal_actions, num_possible_obs)

            if expected != subaction:
                return T(prompt[:2], num_legal_actions, num_possible_obs)

        return T0(prompt, num_legal_actions, num_possible_obs)

    return T