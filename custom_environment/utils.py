from hanabi_learning_environment.pyhanabi import COLOR_CHAR

REVEAL_COLOR = 3  # matches HanabiMoveType.REVEAL_COLOR
REVEAL_RANK = 4  # matches HanabiMoveType.REVEAL_RANK
PLAY = 1  # matches HanabiMoveType.REVEAL_RANK
DISCARD = 2  # matches HanabiMoveType.REVEAL_RANK
COPIES_PER_CARD = {'0': 3, '1': 2, '2': 2, '3': 2, '4': 1}


def color_char_to_idx(color_char):
  r"""Helper function for converting color character to index.

  Args:
    color_char: str, Character representing a color.

  Returns:
    color_idx: int, Index into a color array \in [0, num_colors -1]

  Raises:
    ValueError: If color_char is not a valid color.
  """
  assert isinstance(color_char, str)
  try:
    return next(idx for (idx, c) in enumerate(COLOR_CHAR) if c == color_char)
  except StopIteration:
    raise ValueError("Invalid color: {}. Should be one of {}.".format(
        color_char, COLOR_CHAR))


def abs_position_player_target(action, cur_player, num_players):
    """
    Utility function. Computes the player ID, i.e. absolute position on table, of the target of the action.
    Args:
        action: pyhanabi.HanabiMove object containing the target_offset for REVEAL_XYZ moves
        cur_player: int, player ID of player that computed the action
        num_players: number of total players in the game
    Returns:
        target pid (player ID)
    """
    # For play moves, the target player ID is equal to relative player ID
    if action.type() in [PLAY, DISCARD]:
        return cur_player
    # For reveal moves, it is computed using the target offset and total num of players
    elif action.type() in [REVEAL_RANK, REVEAL_COLOR]:
        return (cur_player + action.target_offset()) % num_players

    return None


def get_cards_touched_by_hint(hint, target_hand, return_indices=False):
    """
    Computes cards in target_hand, that are touched by hint.
    A card is touched by a hint, if one of the following hold:
     - the cards color is equal to the color hinted
     - the cards rank is equals to the rank hinted
     Args:
         hint: pyhanabi.HanabiMove object
         target_hand: list of pyhanabi.HanabiCard objects
         return_indices: if True, this will return integer indices instead of pyhanabi.HanabiCard objects
    Returns:
        cards_touched: list of pyhanabi.HanabiCard objects containing hinted (touched) cards.
            or if return_indices == True
        list of integers, containing indices of touched cards
    """
    cards_touched = list()
    if hint.type() == REVEAL_COLOR:
        color_hinted = hint.color()
        for i, card in enumerate(target_hand):
            if card.color() == color_hinted:
                if return_indices:
                    cards_touched.append(i)
                else:
                    cards_touched.append(card)
    elif hint.type() == REVEAL_RANK:
        rank_hinted = hint.rank()
        for i, card in enumerate(target_hand):
            if card.rank() == rank_hinted:
                if return_indices:
                    cards_touched.append(i)
                else:
                    cards_touched.append(card)
    else:
        raise ValueError
    return cards_touched


def card_is_last_copy(card, discard_pile):
    """
    Returns true, if for given card, all other of its copies are on the discard_pile (none left in the deck)
    Args:
         card: a pyhanabi.HanabiCard object
         discard_pile: a list of pyhanabi.HanabiCard objects containing discarded cards
    Returns:
         True, if all other copies of card are in discard_pile, False otherwise.
    """
    card_copies_total = COPIES_PER_CARD[str(card.rank())]
    card_copies_discarded = 0
    for discarded in discard_pile:
        if discarded.color() == card.color() and discarded.rank() == card.rank():
            card_copies_discarded += 1
    if card_copies_total - card_copies_discarded == 1:
        return True
    return False


def get_card_played_or_discarded(action, player_hand):
    """
    Returns the card that has been played or discarded from player_hand, according to action.
    Args:
         action: pyhanabi.HanabiMove object
         player_hand: list of pyhanabi.HanabiCard objects constituting the hand of the acting player
    Returns:
        a pyhanabi.HanabiCard object
    """
    return player_hand[action.card_index()]
