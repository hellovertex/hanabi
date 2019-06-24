import commandsWebSocket as cmd


def test_parse_action_to_message():
    # 2 player setup after cards been dealt
    num_players = 2
    hand_size = 5
    abs_card_nums = [[9, 8, 7, 6, 5], [4, 3, 2, 1, 0]]
    # calling agent sits at index 0, has hand [9,8,7,6,5]
    agent_pos = 0

    # action examples
    play = {'action_type': 'PLAY', 'card_index': 1}
    discard = {'action_type': 'DISCARD', 'card_index': 0}
    reveal_color = {'action_type': 'REVEAL_COLOR', 'color': 'W', 'target_offset': 1}
    reveal_rank = {'action_type': 'REVEAL_RANK', 'rank': 0, 'target_offset': 1}

    # corresponding jsons
    play_json = 'action {"type":1,"target":6}'
    discard_json = 'action {"type":2,"target":5}'
    reveal_color_json = 'action {"type":0,"target":1,"clue":{"type":1,"value":4}}'
    reveal_rank_json = 'action {"type":0,"target":1,"clue":{"type":0,"value":1}}'
    for i, action in enumerate([play, discard, reveal_color, reveal_rank]):
        comp_arr = [play_json, discard_json, reveal_color_json, reveal_rank_json]
        action_message = cmd.get_server_msg_for_pyhanabi_action(
            action=action,
            abs_card_nums=abs_card_nums,
            agent_pos=agent_pos,
            num_players=num_players,
            hand_size=hand_size
        )
        print(action_message == comp_arr[i])
        assert action_message == [play_json, discard_json, reveal_color_json, reveal_rank_json][i]


# success
# test_parse_action_to_message()

