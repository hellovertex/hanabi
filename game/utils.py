def parse_timestep(ts):
    # parse timestep, returns vectors for observation, rewards, legal_moves, dones
    dones = ts[0] == 2
    rewards = ts[1]
    legal_moves = ts[3]['mask']
    for i in range(len(legal_moves)):
        lm = legal_moves[i]
        lm = [-1e10 if lmi < 0 else 0 for lmi in lm]
        legal_moves[i] = lm
    obs = ts[3]['state']
    beliefs_prob_dict = ts[3]['beliefs_prob_dict']
    score = ts[3]['score']
    custom_rewards = ts[3]['custom_rewards']
    return obs, rewards, legal_moves, dones, score, custom_rewards, beliefs_prob_dict


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]