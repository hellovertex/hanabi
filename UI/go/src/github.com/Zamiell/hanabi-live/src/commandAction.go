/*
	Sent when the user performs an in-game action
	"data" example:
	{
		clue: { // Not present if the type is 1 or 2
			type: 0, // 0 is a rank clue, 1 is a color clue
			value: 1, // If a rank clue, corresponds to the number
			// If a color clue:
			// 0 is blue
			// 1 is green
			// 2 is yellow
			// 3 is red
			// 4 is purple
			// (these mappings change in the mixed variants)
		},
		target: 1,
		// Either the player index of the recipient of the clue, or the card ID
		// (e.g. the first card of the deck drawn is card #1, etc.)
		type: 0,
		// 0 is a clue
		// 1 is a play
		// 2 is a discard
		// 3 is a deck blind play
		// 4 is a time limit reached (only used by the server)
		// 5 is a idle limit reached (only used by the server)
	}
*/

package main

import (
	"strconv"
	"strings"
	"time"
	"os"
	//"runtime"
	//"path/filepath"
)

func commandAction(s *Session, d *CommandData) {
	/*
		Validate
	*/
	// log.Info("COMMAAAND DAAATAA")
	// log.Info(d)
	// Validate that the game exists
	gameID := s.CurrentGame()
	var g *Game
	if v, ok := games[gameID]; !ok {
		s.Warning("Game " + strconv.Itoa(gameID) + " does not exist.")
		return
	} else {
		g = v
	}

	// Validate that the game has started
	if !g.Running {
		s.Warning("Game " + strconv.Itoa(gameID) + " has not started yet.")
		return
	}

	// Validate that they are in the game
	i := g.GetPlayerIndex(s.UserID())
	if i == -1 {
		s.Warning("You are in not game " + strconv.Itoa(gameID) + ", so you cannot send an action.")
		return
	}

	// Validate that it is this player's turn
	if g.ActivePlayer != i && d.Type != actionTypeIdleLimitReached {
		s.Warning("It is not your turn, so you cannot perform an action.")
		return
	}

	// Validate that it is not a replay
	if g.Replay {
		s.Warning("You cannot perform a game action in a shared replay.")
		return
	}

	// Validate that the game is not paused
	if g.Paused {
		s.Warning("You cannot perform a game action when the game is paused.")
		return
	}

	// Local variables
	p := g.Players[i]

	// Validate that a player is not doing an illegal action for their character
	if characterValidateAction(s, d, g, p) {
		return
	}
	if characterValidateSecondAction(s, d, g, p) {
		return
	}

	/*
		Action
	*/

	// Remove the "fail#" and "blind#" states
	g.Sound = ""

	// Start the idle timeout
	// (but don't update the idle variable if we are ending the game due to idleness)
	if d.Type != actionTypeIdleLimitReached {
		go g.CheckIdle()
	}

	// Do different tasks depending on the action
	doubleDiscard := false
	if d.Type == actionTypeClue {
		// Validate that the target of the clue is sane
		if d.Target < 0 || d.Target > len(g.Players)-1 {
			s.Warning("That is an invalid clue target.")
			return
		}

		// Validate that the player is not giving a clue to themselves
		if g.ActivePlayer == d.Target {
			s.Warning("You cannot give a clue to yourself.")
			return
		}

		// Validate that there are clues available to use
		if g.Clues == 0 {
			s.Warning("You cannot give a clue when the team has 0 clues left.")
			return
		}
		if strings.HasPrefix(g.Options.Variant, "Clue Starved") && g.Clues == 1 {
			s.Warning("You cannot give a clue when the team only has 0.5 clues.")
			return
		}

		// Validate that the clue type is sane
		if d.Clue.Type < clueTypeRank || d.Clue.Type > clueTypeColor {
			s.Warning("That is an invalid clue type.")
			return
		}

		// Validate that rank clues are valid
		if d.Clue.Type == clueTypeRank {
			valid := false
			for _, rank := range variants[g.Options.Variant].ClueRanks {
				if rank == d.Clue.Value {
					valid = true
					break
				}
			}
			if !valid {
				s.Warning("That is an invalid rank clue.")
				return
			}
		}

		// Validate that the color clues are valid
		if d.Clue.Type == clueTypeColor &&
			(d.Clue.Value < 0 || d.Clue.Value > len(variants[g.Options.Variant].ClueColors)-1) {

			s.Warning("That is an invalid color clue.")
			return
		}

		// Validate "Detrimental Character Assignment" restrictions
		if characterCheckClue(s, d, g, p) {
			return
		}

		// Validate that the clue touches at least one card
		p2 := g.Players[d.Target] // The target of the clue
		touchedAtLeastOneCard := false
		for _, c := range p2.Hand {
			if variantIsCardTouched(g.Options.Variant, d.Clue, c) {
				touchedAtLeastOneCard = true
				break
			}
		}
		if !touchedAtLeastOneCard &&
			// Make an exception if they have the optional setting for "Empty Clues" turned on
			!g.Options.EmptyClues &&
			// Make an exception for the "Color Blind" variants (color clues touch no cards),
			// "Number Blind" variants (rank clues touch no cards),
			// and "Totally Blind" variants (all clues touch no cards)
			(!strings.HasPrefix(g.Options.Variant, "Color Blind") || d.Clue.Type != clueTypeColor) &&
			(!strings.HasPrefix(g.Options.Variant, "Number Blind") || d.Clue.Type != clueTypeRank) &&
			!strings.HasPrefix(g.Options.Variant, "Totally Blind") &&
			// Make an exception for certain characters
			!characterEmptyClueAllowed(d, g, p) {

			s.Warning("You cannot give a clue that touches 0 cards in the hand.")
			return
		}

		p.GiveClue(d, g)

		// Mark that the blind-play streak has ended
		g.BlindPlays = 0

		// Mark that the misplay streak has ended
		g.Misplays = 0

	} else if d.Type == actionTypePlay {
		// Validate that the card is in their hand
		if !p.InHand(d.Target) {
			s.Warning("You cannot play a card that is not in your hand.")
			return
		}

		// Validate "Detrimental Character Assignment" restrictions
		if characterCheckPlay(s, d, g, p) {
			return
		}

		c := p.RemoveCard(d.Target, g)
		doubleDiscard = p.PlayCard(g, c)
		p.DrawCard(g)

	} else if d.Type == actionTypeDiscard {
		// Validate that the card is in their hand
		if !p.InHand(d.Target) {
			s.Warning("You cannot play a card that is not in your hand.")
			return
		}

		// Validate that the team is not at the maximum amount of clues
		// (the client should enforce this, but do a check just in case)
		clueLimit := maxClues
		if strings.HasPrefix(g.Options.Variant, "Clue Starved") {
			clueLimit *= 2
		}
		if g.Clues == clueLimit {
			s.Warning("You cannot discard while the team has " + strconv.Itoa(maxClues) + " clues.")
			return
		}

		// Validate "Detrimental Character Assignment" restrictions
		if characterCheckDiscard(s, g, p) {
			return
		}

		g.Clues++
		c := p.RemoveCard(d.Target, g)
		doubleDiscard = p.DiscardCard(g, c)
		characterShuffle(g, p)
		p.DrawCard(g)

		// Mark that the blind-play streak has ended
		g.BlindPlays = 0

		// Mark that the misplay streak has ended
		g.Misplays = 0

	} else if d.Type == actionTypeDeckPlay {
		// Validate that the game type allows deck plays
		if !g.Options.DeckPlays {
			s.Warning("Deck plays are disabled for this game.")
			return
		}

		// Validate that there is only 1 card left
		// (the client should enforce this, but do a check just in case)
		if g.DeckIndex != len(g.Deck)-1 {
			s.Warning("You cannot blind play the deck until there is only 1 card left.")
			return
		}

		p.PlayDeck(g)

	} else if d.Type == actionTypeTimeLimitReached {
		// This is a special action type sent by the server to itself when a player runs out of time
		g.Strikes = 3
		g.EndCondition = actionTypeTimeLimitReached
		g.Actions = append(g.Actions, ActionText{
			Type: "text",
			Text: p.Name + " ran out of time!",
		})
		g.NotifyAction()

	} else if d.Type == actionTypeIdleLimitReached {
		// This is a special action type sent by the server to itself when the game has been idle for too long
		g.Strikes = 3
		g.EndCondition = actionTypeIdleLimitReached
		g.Actions = append(g.Actions, ActionText{
			Type: "text",
			Text: "Players were idle for too long.",
		})
		g.NotifyAction()

	} else {
		s.Warning("That is not a valid action type.")
		return
	}

	// Do post-action tasks
	characterPostAction(d, g, p)

	// Send a message about the current status
	g.NotifyStatus(doubleDiscard)

	// Adjust the timer for the player that just took their turn
	// (if the game is over now due to a player running out of time, we don't
	// need to adjust the timer because we already set it to 0 in the
	// "checkTimer" function)
	if d.Type != actionTypeTimeLimitReached {
		p.Time -= time.Since(g.TurnBeginTime)
		// (in non-timed games, "Time" will decrement into negative numbers to show how much time they are taking)

		// In timed games, a player gains additional time after performing an action
		if g.Options.Timed {
			p.Time += time.Duration(g.Options.TimePerTurn) * time.Second
		}

		g.TurnBeginTime = time.Now()
	}

	// If a player has just taken their final turn,
	// mark all of the cards in their hand as not able to be played
	if g.EndTurn != -1 && g.EndTurn != g.Turn+len(g.Players)+1 {
		log.Info(g.GetName() + "Player \"" + p.Name + "\" just took their final turn; " +
			"marking the rest of the cards in their hand as not playable.")
		for _, c := range p.Hand {
			c.CannotBePlayed = true
		}
	}

	// Increment the turn
	// (but don't increment it if we are on a characters that takes two turns in a row)
	if !characterTakingSecondTurn(d, g, p) {
		g.Turn++
		if g.TurnsInverted {
			// In Golang, "%" will give the remainder and not the modulus,
			// so we need to ensure that the result is not negative or we will get a "index out of range" error below
			g.ActivePlayer += len(g.Players)
			g.ActivePlayer = (g.ActivePlayer - 1) % len(g.Players)
		} else {
			g.ActivePlayer = (g.ActivePlayer + 1) % len(g.Players)
		}
	}
	np := g.Players[g.ActivePlayer] // The next player

	// Check for character-related softlocks
	// (we will set the strikes to 3 if there is a softlock)
	characterCheckSoftlock(g, np)

	// Check for end game states
	if g.CheckEnd() {
		var text string
		if g.EndCondition > endConditionNormal {
			text = "Players lose!"
		} else {
			text = "Players score " + strconv.Itoa(g.Score) + " points."
		}
		g.Actions = append(g.Actions, ActionText{
			Type: "text",
			Text: text,
		})
		g.NotifyAction()
		log.Info(g.GetName() + " " + text)
	}

	// Send the new turn
	// This must be below the end-game text (e.g. "Players lose!"),
	// so that it is combined with the final action
	g.NotifyTurn()

	if g.EndCondition == endConditionInProgress {
		// write the current players name to tmpfile
		//pwd, _ := os.Getwd()
		//_, filename, _, _ := runtime.Caller(1)
		//absPath := filepath.Join(filepath.Dir(filename), "../tmp/cur_player.txt")
		//f, _ := os.Create(absPath)
		f, _ := os.Create("/home/cawa/go/src/github.com/Zamiell/hanabi-live/src/tmp/cur_player.txt")
		defer f.Close()
		f.WriteString(np.Name)
		f.Sync() // end of custom code
		log.Info(g.GetName() + " It is now " + np.Name + "'s turn.")
	} else if g.EndCondition == endConditionNormal {
		if g.Score == g.GetPerfectScore() {
			g.Sound = "finished_perfect"
		} else {
			// The players did got get a perfect score, but they did not strike out either
			g.Sound = "finished_success"
		}
	} else if g.EndCondition > endConditionNormal {
		g.Sound = "finished_fail"
	}

	// Tell every client to play a sound as a notification for the action taken
	g.NotifySound()

	if g.EndCondition > endConditionInProgress {
		g.End()
		return
	}

	// Send the "action" message to the next player
	np.Session.NotifyAction(g)

	// Send every user connected an update about this table
	// (this is sort of wasteful but is necessary for users to see if it is
	// their turn from the lobby and also to see the progress of other games)
	if !g.NoDatabase {
		// Don't send table updates if we are in the process of emulating JSON actions
		notifyAllTable(g)
	}

	// Send everyone new clock values
	g.NotifyTime()

	if g.Options.Timed {
		// Start the function that will check to see if the current player has run out of time
		// (since it just got to be their turn)
		go g.CheckTimer(g.Turn, g.PauseCount, np)

		// If the player queued a pause command, then pause the game
		if np.RequestedPause {
			np.RequestedPause = false
			commandPause(np.Session, &CommandData{
				Value: "pause",
			})
		}
	}
}
