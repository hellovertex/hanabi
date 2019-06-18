from tf_agents.policies import py_policy


class LegalMovesSampler(py_policy.Base):
    """ Policy wrapper that ensures we only sample legal pyhanabi.HanabiMove actions """

    def __init__(self, policy, env):
        super(LegalMovesSampler, self).__init__(
            time_step_spec=policy.time_step_spec,
            action_spec=policy.action_spec,
            policy_state_spec=policy.policy_state_spec,
            info_spec=policy.info_spec
        )

        self._policy = policy
        self._env = env

    def _action(self, time_step, policy_state):
        """ Will be called by super class upon calling super.action() method"""
        # get int values of legal moves
        obss = self._env._make_observation_all_players()
        legal_moves_as_int = obss['player_observations'][obss['current_player']]['legal_moves_as_int']

        # loop sampling until legal_move_is_obtained
        move_is_illegal = True

        a = None
        print(legal_moves_as_int)
        print(self._policy)
        while move_is_illegal:
            a = self._policy.action(time_step)
            print(a.action.numpy()[0])
            if a.action.numpy()[0] in legal_moves_as_int:
                move_is_illegal = False

        return a