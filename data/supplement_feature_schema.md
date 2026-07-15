# selected_cards = PendingCards zone

# set only one of these to true (check game logic)
selected_cards_make_high_card
selected_cards_make_pair
selected_cards_make_two_pair
selected_cards_make_three_kind
selected_cards_make_straight
selected_cards_make_flush
selected_cards_make_full_house
selected_cards_make_four_kind
selected_cards_make_straight_flush
selected_cards_make_five_kind
selected_cards_make_flush_house
selected_cards_make_flush_five

# set one or many of these to true
selected_cards_have_high_card
selected_cards_have_pair
selected_cards_have_two_pair
selected_cards_have_three_kind
selected_cards_have_straight
selected_cards_have_flush
selected_cards_have_full_house
selected_cards_have_four_kind
selected_cards_have_straight_flush
selected_cards_have_royal_flush
selected_cards_have_five_kind
selected_cards_have_flush_house
selected_cards_have_flush_five

selected_contains_exactly_1_card
selected_contains_exactly_4_cards
selected_contains_exactly_5_cards

selected_hand_type_already_played_this_round
selected_hand_type_is_most_played_hand

# is the first scored card a face card etc...
selected_position_0_scored_is_a_face_card
selected_position_0_scored_is_debuffed

# count of cards that would be scored if PendingHand was played 
selected_face_card_scored_count
selected_ace_scored_count
selected_2_scored_count
selected_6_scored_count
selected_8_scored_count
selected_ace_2_3_5_scored_and_8_count
selected_2_3_4_or_5_scored_count
selected_even_rank_scored_count
selected_odd_rank_scored_count
selected_spade_scored_count
selected_heart_scored_count
selected_diamond_scored_count
selected_club_scored_count
selected_scored_debuff_count # count the number of cards in PendingCards that are debuffed

# check if hands_left == 1
is_final_hand_of_round

# check if dollars <= 4
money_less_than_or_equal_to_4

# is ante == 1
is_ante_1

# For each of these, set the flag to true if an object in CurrentJokers contains the class_id associated with these flags and the object is not debuffed. 
has_four_fingers # 133 j_four_fingers: All Flushes and Straights can be made with 4 cards 
has_shortcut # 195 j_shortcut: Allows Straights to be made with gaps of 1 rank (ex: 10 8 6 5 3)
has_smeared_joker # 198 j_smeared:  Hearts and  Diamonds count as the same suit,  Spades and  Clubs count as the same suit 
has_pareidolia # 174 j_pareidolia: All cards are considered face cards 
has_splash # 202 j_splash: Every played card counts in scoring 

# For each of these, count the number of objects with matching descriptions in the CurrentHand zone.
held_king_count                
held_queen_count               
held_face_card_count          
held_spade_count
held_club_count                 
held_spade_or_club_count       
held_lowest_rank_value          
held_lowest_rank_is_debuffed          


GAME LOGIC/INFORMATION AND CONTEXT:

Higher tier hands take precedence over lower tier hands regardless of their level or scoring, for example, if your hand is K K K K 2, and all of them are diamonds, the hand will always be a Four of a Kind and never a Flush.

Generally, only the cards relevant to the hand are scored. All others are unscored. For example, if an Ace is played high with 4 other cards, only the High card base amount and the Ace's values are used for the hand's score. The other cards (up to 4) are discarded and have no effect. The two main exceptions to this rule are Stone Cards, which always score, and 202	j_splash, which allows all played cards to score.

Hand types:

High Card
	No other hand is possible.
Pair
	Two cards with a matching rank.
Two Pair
	Two separate pairs.
Three of a Kind
	Three cards with a matching rank.
Straight
	Five cards in consecutive order.
Flush
	Five cards of the same suit.
Full House
	Three cards with one rank and two cards with another rank.
Four of a Kind
	Four cards with a matching rank.
Straight Flush
	Five cards in consecutive order and the same suit.
Royal Flush
	Ace-high Straight Flush.
Five of a Kind
	Five cards with the same rank.
Flush House
	Full House where all cards are the same suit.
Flush Five
	Five cards with the same rank and same suit.


    The exact judgment criteria of a Straight Flush is "any hand which is both a Straight and a Flush". With 133	j_four_fingers, it's possible to make a Straight Flush with such cards as 9♠️ 8♠️ 7♥️ 6♠️ 3♠️. Additionally, a Royal Flush is "a Straight Flush with all cards of rank 10 or higher".
        Likewise, the exact judgment criteria of a Flush House is "any hand which is both a Full House and a Flush". With 133	j_four_fingers, it's possible to make a Flush House with such cards as 8♠️ 8♠️ 8♥️ 6♠️ 6♠️.
        Similarly, the exact judgement criteria of Flush Five is "any hand which is both Five of a Kind and a Flush." With 133	j_four_fingers, it's possible to make a Flush Five with 5 cards of the same rank, of which only 4 are of the same suit.
    Two Pair and Full House require that the cards are of different ranks. Thus, a Four of a Kind is not considered as including a Two Pair. The same is true for Five of a Kind and Flush House.
    Unlike most Poker Hands that level up through the use of a Planet Card, the "Royal Flush" instead levels up alongside the "Straight Flush".


# 79 m_wild
Wild Cards are playing cards that are Enhanced to count as all Suits simultaneously. 
When a Wild Card is debuffed, it reverts back to its base suit.




