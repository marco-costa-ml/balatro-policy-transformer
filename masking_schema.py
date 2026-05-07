StartNewRun (check)
SelectBlind (need to make it work with skipblind)
SkipBlind ()
RerollBossBlind
DiscardHand (check)
PlayHand (check)
UseConsumable
SelectCard(i) (half check)
CashOut (check)
SelectPackItem(i)
BuyAndUseShopConsumable(i) (check)
BuyShopItem(i) (check)
LeaveShop (check)
SkipPack
SellItem(i) 
RerollShop (check)
NOOP
SWAP(i, j)


(flatten everything. You turn every action into a single index, even SWAP.
NOOP
SelectBlind_0
SelectBlind_1
...
SWAP_0_1
SWAP_0_2
...
SWAP_10_11)



Pages = Blind_Select, In_Blind, In_Shop, In_TarotSpectral_Pack, In_JokerStandardPlanet_Pack 

PERSISTENT INFORMATION: THIS IS INFORMATION THAT IS FED TO THE MODEL BEFORE EVERY STEP/ACTION

last_tarot_planet: int, default: null - this field should be set equal to the class_id of the target object of any UseConsumable or BuyAndUseConsumable actions that have a target object with a class_id between 236 - 247 or 298 - 319
	
stake: int, default: 268, this field should be set equal to the class_id of the first object in Zone "CurrentStake" in the very first action "StartNewRun"  

ecto_minus: int, default: 0 this field should increment everytime UseConsumable or BuyAndUseShopConsumable action is completed when an object of class_id 253 is the target of the action 
skips: int, default: 0, this field should increment everytime a SkipBlind action is completed
hands_played: int, default: 0 this field should increment everytime a PlayHand action is completed
unused_discards: int, default: 0 this field should increment by the amount in the discards_left ocr field everytime a CashOut action is completed 
vouchers_redeemed: int[], default: [] a class_id should be added to this array anytime a BuyShopItem action is completed with an object of class_id 320-351 as the target of the action

ante_boss_blind: int, default: null, this value should be set to the last object of BlindOfferings that has a class_id between 370 - 399 with the exception of 371 and 394
small_status: int, default: 1, this value should be set to 0 when SelectBlind occurs with a target object of class_id 394 and reset to 1 when a SelectBlind action completes with a target object of class_id between 370 - 399 with the exception of 371 and 394. This value should be set to 2 when a SkipBlind action completes with a target object of class_id equal to 394
big_statusint, default: 1, this value should be set to 0 when SelectBlind occurs with a target object of class_id 371 and reset to 1 when a SelectBlind action completes with a target object of class_id between 370 - 399 with the exception of 371 and 394. This value should be set to 2 when a SkipBlind action completes with a target object of class_id equal to 371
bosses_used: int[], default: [] a class_id should be added to this array anytime a SelectBlind occurs with a class_id between 370 - 399 with the exception of 371 and 394

is_boss_blind_rerolled: bool, default: false, This field should become true if a RerollBossBlind action completes. this field should be reset to false everytime a SelectBlind action completes with a target object of class_id between 370 - 399 with the exception of 371 and 394
hands = {
    ["Flush Five"] =        {level = 1, played = 0, played_this_round = 0},
    ["Flush House"] =       {level = 1, played = 0, played_this_round = 0},
    ["Five of a Kind"] =    {level = 1, played = 0, played_this_round = 0},
    ["Straight Flush"] =    {level = 1, played = 0, played_this_round = 0},
    ["Four of a Kind"] =    {level = 1, played = 0, played_this_round = 0},
    ["Full House"] =        {level = 1, played = 0, played_this_round = 0},
    ["Flush"] =             {level = 1, played = 0, played_this_round = 0},
    ["Straight"] =          {level = 1, played = 0, played_this_round = 0},
    ["Three of a Kind"] =   {level = 1, played = 0, played_this_round = 0},
    ["Two Pair"] =          {level = 1, played = 0, played_this_round = 0},
    ["Pair"] =              {level = 1, played = 0, played_this_round = 0},
    ["High Card"] =         {level = 1, played = 0, played_this_round = 0},
}

cards_in_deck: {} default = {class_ids 0-51} (generate card objects for each class id), this field should be populated with objects of the appropriate class id, metadata of each object should be default for its class_id


INFORMATION THAT SHOULD NOT BE FED TO THE MODEL:
swap_count = 0 This field should increment everytime a SWAP action completes and reset to 0 anytime a SelectBlind, PlayHand or DiscardHand action completes
Boolean deck_detected = false, dummy value for now



CREATING METADATA FOR EACH OBJECT:
Metadata should be created with defaults from metadata_map.csv, note that only

	- Cards (0 - 51)
	- Jokers (80 - 229)
	- Planets (236 - 247)
	- Spectral (248 - 265)
	- Tarots (298 - 319)

	OBJECTS WITH CLASS_ID 231 should be treated like a playing card with an unknown suit, rank, and all other metadata

cost should be calculated as such:

	if the object contains an edition, (class_id 68 - 71), add the cost of the edition to the cost of the parent
	if vouchers_redeemed contains 322 a 25% discount is applied to the cost and sell price of all items
	or if vouchers_redeemed contains 330, a 50% discount is applied to the buy cost and sell price of all items
	Round half down (for example, 8.5 is rounded to 8).
	If the buy price after rounding is $0 or lower, set it to $1.
finally:
	if an object with class_id 85 exists in CurrentJokers, the COST of any objects with class ids 358-360 and 236-247 is set to 0	
	if an object has a sticker of 367, its COST and SELL price becomes 1


Calculating sell value
    Divide the total buy cost by 2.
    Floor the result of the division (for example, 3.75 is rounded to 3).
    If the sell value after flooring is $0 or lower, set it to $1.



ACTION LIST
StartNewRun - not an action, defines the start of a run. contains important persistent state information


SelectBlind - only in blind_select page
SkipBlind - only in blind_select page
	MASK if OfferedTag.length = 0 
RerollBossBlind - only in blind_select page
	if vouchers_redeemed contains 346 OR (vouchers_redeemed contains 324 & is_boss_blind_rerolled == false)
		ALLOW
	else:
		MASK

DiscardHand - only in in_blind page
	MASK unless selected_cards.length > 0
PlayHand - only in in_blind page
	MASK unless selected_cards.length > 0



UseConsumable
	if(CurrentConsumablesSelected.length <= 0)
			MASK
	#-- LOGIC HERE --


BuyAndUseShopConsumable(i) - only in in_shop page, generate BuyAndUseShopConsumable(i) for each in ShopOfferings where isConsumable=true
	SAME LOGIC AS UseConsumable (note selected card will always be 0)

SelectPackItem(i) - only in in_JokerStandardPlanet_Pack page, generate SelectPackItem(i) for each in CurrentPack
	if Zone PackShopOfferingsSelected[0] is class id between [80 - 229] and ocr jokers_current>=jokers_total
		MASK
	LOGIC IS THE SAME AS USE CONSUMABLE, EXCEPT INSTEAD OF (CurrentConsumablesCardsSelected) we will use CurrentPackConsumablesCardsSelected


SelectCard(i) only in In_Blind or In_TarotSpectral_Pack, generate SelectCard(i) for each in CurrentHand

CashOut - only in cash_out page




BuyShopItem(i) - only in in_pack page, generate BuyShopItem(i) for each in ShopOfferings
	if(dollars <= shop_offerings(i).cost)
		MASK
	if(isJoker(shop_offerings(i).class_id) & jokers_current >= jokers_total & shop_offerings(i).edition != negative)
		MASK
	if(isConsumable(shop_offerings(i).class_id) & consumables_current >= consumables_total)
		MASK


LeaveShop - only in in_pack page
SkipPack - only in in_pack page

SellItem(i) - always available. for each in current_jokers & current_consumables generate SellItem action option
	if(item(i).sticker == class_id 369)
		MASK


RerollShop - only in in_shop page
	if(reroll_price >= dollars)
		MASK


NOOP
SWAP(i, j) 
	if swap_count >= (Math.max(0, jokers_current - 1)) * 3
		MASK 


global_state = {
    # page / phase
    "page": page_id,  # Blind_Select, In_Blind, In_Shop, etc.

    # run state
    "stake": int_or_null,
    "ante": int,
    "round": int,
    "dollars": int,
    "hands_left": int,
    "discards_left": int,
    "hand_size": int,
    "joker_slots_used": int,
    "joker_slots_total": int,
    "consumable_slots_used": int,
    "consumable_slots_total": int,

    # persistent counters
    "last_tarot_planet": int_or_null,
    "ecto_minus": int,
    "skips": int,
    "hands_played": int,
    "unused_discards": int,

    # blind state
    "ante_boss_blind": int_or_null,
    "small_status": int,  # 0 selected, 1 available, 2 skipped
    "big_status": int,
    "is_boss_blind_rerolled": bool,

    # shop state
    "reroll_price": int,

    # compact set-like states
    "vouchers_redeemed_mask": [0/1 for each voucher class_id],
    "bosses_used_mask": [0/1 for each boss class_id],

    # poker hand state
    "poker_hands": {
        "Flush Five": {
            "level": int,
            "played": int,
            "played_this_round": int,
        },
        ...
    }
}


zones = {
    "current_hand": [object, object, ...],
    "selected_cards": [object, object, ...],
    "current_jokers": [object, object, ...],
    "current_consumables": [object, object, ...],
    "shop_offerings": [object, object, ...],
    "current_pack": [object, object, ...],
    "blind_offerings": [object, object, ...],
    "offered_tag": [object, object, ...],
    "cards_in_deck": [object, object, ...],
}

obj = {
    # identity
    "class_id": int,
    "object_type": enum_id,   # card, joker, tarot, planet, spectral, voucher, pack, blind, tag, unknown
    "zone": enum_id,
    "position_in_zone": int,

    # economy
    "cost": int,        # 0 if not applicable
    "sell_value": int,  # 0 if not applicable

    # playing-card fields (0 when not a card)
    "rank_index": int,
    "suit_index": int,
    "is_ace": 0/1,
    "is_face": 0/1,

    # modifiers (0 = none)
    "edition": enum_id_or_0,
    "modifier": enum_id_or_0,
    "seal": enum_id_or_0,
    "sticker": enum_id_or_0,

    "is_debuffed": 0/1,

    # joker / ownership info (0 when not applicable)
    "joker_position": int,
    "turns_owned": int,
    "hands_owned": int,
    "rounds_owned": int,

    # visible dynamic values (0 when not present)
    "chips": float,
    "mult": float,
    "xmult": float,
    "dollars": float,
}