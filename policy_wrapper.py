class LegalMovesSampler:
    """ legal_moves_sampler accounts for two things:
    the first being, that the observation contains vectorized + legal_moves_as_int
    and secondly, that we are only allowed to sample legal moves

    usage will be:

    legal_moves_sampler = legal_moves_sampler.LegalMovesSampler(QPolicy)
    action_step = legal_moves_sampler.action(time_steps) """

    pass