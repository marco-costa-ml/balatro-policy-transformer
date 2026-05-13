EVENTS

StartNewRun 
SelectBlind
SkipBlind
RerollBossBlind
DiscardHand
PlayHand
UseConsumable
SelectCard
CashOut
SelectPackItem
BuyAndUseShopConsumable
BuyShopItem
LeaveShop
SkipPack
SellItem
RerollShop
SWAP(i, j)


BuyShopItem
    buyvoucher
    buyandopenplanetstandardbuffoonpack
    buytopshelfjoker
    buyandopentarotspectralpack
    buytopshelfconsumable
SelectPackItem
    selectpackitemtarot
    selectpackitemcard
    selectpackitemjoker
    selectpackitemplanet
SellItem
    selljoker
    sellconsumable
UseConsumable

SkipPack - does not have a target
    skipplanetstandardbuffoonpack
    skiptarotspectralpack
    skipplanetstandardbuffoonpackblind
    skiptarotspectralpackblind




REQUIRES_AT_LEAST_ONE_CARD = {
    249,  # c_aura
    251,  # c_cryptid
    252,  # c_deja_vu
    259,  # c_medium
    263,  # c_talisman
    264,  # c_trance

    298,  # c_chariot
    299,  # c_death
    300,  # c_devil
    302,  # c_empress
    304,  # c_hanged_man
    305,  # c_heirophant
    309,  # c_justice
    310,  # c_lovers
    311,  # c_magician
    312,  # c_moon
    313,  # c_star
    314,  # c_strength
    315,  # c_sun
    317,  # c_tower
    319,  # c_world
}


EVENTS
 
SelectBlind
SkipBlind
RerollBossBlind
DiscardHand
PlayHand
UseConsumable
SelectCard
CashOut
SelectPackItem
BuyAndUseShopConsumable
BuyShopItem
LeaveShop
SkipPack
SellItem 
RerollShop
SWAP(i, j)

ZONES
CurrentHandSelected
CurrentHandAll

PackOfferingsSelected
PackOfferingsAll

TarotSpectralHandSelected
TarotSpectralHandAll

CurrentConsumablesSelected
CurrentConsumablesAll

VoucherShopOfferingsSelected
VoucherShopOfferingsAll

PackShopOfferingsSelected
PackShopOfferingsAll

TopShelfShopOfferingsSelected
TopShelfShopOfferingsAll

CurrentJokersSelected
CurrentJokersAll

BlindToken
BlindOffering
BlindOfferingsNext

OfferedTag
BigBlindTag

CurrentStake
CurrentDeck
CurrentTags


I will now define the role of granularize.py:

First, we define, within just the scope of the script, a synthetic zone called "PendingCards" which will contain only selected playing cards. A copy of "PendingCards" is to be recorded as a parameter in every event.
Secondly, we make a distinction between "Selected" zones and "All" zones. These zones come in the format of ZoneAAll and ZoneASelected. "All" zones always contain copies of all the elements in "Selected" zones. 
Thirdly, we define a list outside the loop called "last_jokers". 

Fourthly, for each relevant event or subevent we define a target object:

SelectBlind, SkipBlind, RerollBossBlind, RerollShop, DiscardHand, PlayHand, CashOut, LeaveShop, and SkipPack do not have target objects.

BuyShopItem
    The target object for "buyvoucher" is always the first element in VoucherShopOfferingsSelected 
    The target object for "buyandopenplanetstandardbuffoonpack" is always the first element in PackShopOfferingsSelected
    The target object for "buyandopentarotspectralpack" is always the first element in PackShopOfferingsSelected
    The target object for "buytopshelfjoker" is always the first element in TopShelfShopOfferingsSelected 
    The target object for "buytopshelfconsumable" is always the first element in TopShelfShopOfferingsSelected 

SelectPackItem
    The target object for "selectpackitemtarot", "selectpackitemcard", "selectpackitemjoker" and "selectpackitemplanet" is always the first element in PackOfferingsSelected 

SellItem
    The target object for "selljoker" is always the first element in CurrentJokersSelected 
    The target object for "sellconsumable" is always the first element in CurrentConsumablesSelected 

BuyAndUseShopConsumable
    The target object for BuyAndUseShopConsumable is always the first element in TopShelfShopOfferingsSelected 

UseConsumable
    The target object for UseConsumable is always the first element in CurrentConsumablesSelected 


Then for relevant events, we determine what the selected playing cards are:

    SelectBlind, SkipBlind, RerollBossBlind, CashOut, BuyAndUseShopConsumable, BuyShopItem, LeaveShop, SkipPack, SellItem, and RerollShop do not require selected playing cards.

    SelectPackItem
        The selected playing cards for the subevent "selectpackitemtarot" are all the elements in TarotSpectralHandSelected 
        "selectpackitemcard", "selectpackitemjoker", "selectpackitemplanet" do not require selected playing cards.
        The pool of playing cards for SelectPackItem is TarotSpectralHandAll

    UseConsumable
        UseConsumable does not always require playing cards, see REQUIRES_AT_LEAST_ONE_CARD for information on which consumables require selected playing cards.
        When in In_TarotSpectral_Pack, the selected playing cards for UseConsumable are all the elements in TarotSpectralHandSelected 
        When in In_TarotSpectral_Pack, the pool of playing cards for UseConsumable is TarotSpectralHandAll
        When in In_Blind, the selected playing cards for UseConsumable are all the elements in CurrentHandSelected
        When in In_Blind, the pool of playing cards for UseConsumable is CurrentHandAll
        UseConsumable never requires playing cards on any of the other pages.

    PlayHand
        The selected playing cards for PlayHand are all the elements in CurrentHandSelected
        The pool of playing cards for PlayHand is CurrentHandAll

    DiscardHand
        The selected playing cards for DiscardHand are all the elements in CurrentHandSelected
        The pool of playing cards for DiscardHand is CurrentHandAll

    If they have selected playing cards, the events SelectPackItem, UseConsumable, PlayHand and DiscardHand become parent events.
    For each parent event and for each selected playing card, we store the playing card's zone id. Then for each playing card in order of zone id, we create a new, synthetically generated event "SelectCard" with the playing card as the target object.
    After a SelectCard event is created, we add its target playing card object to PendingCards, and remove that object from the pool of playing cards.

    Finally, we write the parent event. The parent event receives a copy of PendingCards as one of its populated zones, and all objects in PendingCards are removed from the pool of playing cards. After the parent event has been recorded, PendingCards is emptied.

    Note: I said that all events should have a target object. In reality, we simplify and use target_zone and target_position variables. Target zone is the zone the target object occupies and target_position is the target object's position within that zone. It is critical we accurately determine these values by determining what zone our target object exists in and what position in the zone it occupies.

Before recording any event of the following event types:

DiscardHand
PlayHand
UseConsumable
CashOut
SelectPackItem
BuyAndUseShopConsumable
BuyShopItem
LeaveShop
SellItem

we treat that event as a parent event for the purposes of generating SWAP events.

Before the parent event is recorded, compare last_jokers to the parent event's CurrentJokersAll zone.

If last_jokers and CurrentJokersAll contain the same joker objects but in a different order, generate one or more SWAP events to transform last_jokers into the CurrentJokersAll order, use a greedy algorithm that a model could mimick.
If new jokers were added or old jokers were removed, first reconcile the set difference. SWAP events should only be generated over the shared jokers that exist in both last_jokers and CurrentJokersAll.
For each SWAP event generated, we record the index of the jokers swap_pair	i.e: [ 1, 2 ]

Each generated SWAP event should contain all OCR values from the parent event.
Each generated SWAP event should also contain all populated zones from the parent event, except CurrentJokersAll.
For CurrentJokersAll, each SWAP event should use the current local value of last_jokers at the moment immediately before that SWAP is applied.

After each SWAP event is generated, locally update last_jokers to reflect the joker order after that SWAP.
Once all required SWAP events have been generated, record the parent event.

After the parent event is recorded, set last_jokers equal to the contents of the parent event's CurrentJokersAll zone.

the objects and their order in the CurrentJokersAll zone should be saved.

When one of those events occurs again, the order of the objects in the new event should be compared to the order of the objects in the previous tracked event.

If the orders do not match, several SWAP events should be generated to transform the previous CurrentJokersAll order into the new CurrentJokersAll order.
After the needed SWAP events are generated, the current event should be recorded with the updated CurrentJokersAll order.

For all events, we record all ocr values and we include all "All" zones and their objects as well as target_zone and target_position if the event has a target object. We omit "Selected" zones in order to avoid target leakage. 
The result should be a list of events containing plenty of game state information from zones and ocr values and each event containing either a single target object, a swap pair or just the event type. 




New Action Map design:

We have made a change to how we store information about target objects. Before, we used target objects, we have since switched to a zone index + position in zone index. These two variables give us enough information to narrow down our target to a single object within a zone.

This change gives us an opportunity to create a more detailed action map.

I'd like all actions that have a target object to be in the form of
Action_Zone_Index

    where index ranges from 0 to n, n being the maximum number of observed objects in that zone for the entire corpus.  

    UseConsumable_CurrentConsumables_0
    UseConsumable_CurrentConsumables_n
    
    BuyShopItem_VoucherShopOfferings_0
    BuyShopItem_VoucherShopOfferings_n

    BuyShopItem_PackShopOfferings_0
    BuyShopItem_PackShopOfferings_n

    BuyShopItem_TopShelfShopOfferings_0
    BuyShopItem_TopShelfShopOfferings_n

    SelectPackItem_PackOfferings_0
    SelectPackItem_PackOfferings_n

    SellItem_CurrentJokers_0
    SellItem_CurrentJokers_n

    SellItem_CurrentConsumables_0
    SellItem_CurrentConsumables_n

    BuyAndUseShopConsumable_TopShelfShopOfferings_0
    BuyAndUseShopConsumable_TopShelfShopOfferings_n

    SelectCard_CurrentHand_0
    SelectCard_CurrentHand_n

    SelectCard_TarotSpectralHand_0
    SelectCard_TarotSpectralHand_n

I'd like each all swap actions to be in the form of
    SWAP_i_j where i and j are equal to the values in the swap pair variables, respectively. 

The rest of the actions should be in the form:
    "StartNewRun",
    "SelectBlind",
    "SkipBlind",
    "RerollBossBlind",
    "DiscardHand",
    "PlayHand",
    "CashOut",
    "LeaveShop",
    "SkipPack",
    "RerollShop",


Tensorize design:

Due to the simplification of zomes (removal of "ZoneName" in favor of "ZoneNameAll") 





Pages = Blind_Select, In_Blind, Cash_Out, In_Shop, In_TarotSpectral_Pack, In_JokerStandardPlanet_Pack 

PERSISTENT INFORMATION: THIS IS INFORMATION THAT IS FED TO THE MODEL BEFORE EVERY STEP/ACTION

last_tarot_planet: int, default: null - this field should be set equal to the class_id of the target object of any UseConsumable or BuyAndUseConsumable actions that have a target object with a class_id between 236 - 247 or 298 - 319 the event is processed.
	
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



If a playhand event occurs and the ocr value of hand_and_level evaluates to ... logic here ... set the value of level to ... logic here ... and also increment played for that hand.

cards_in_deck: {} default = {class_ids 0-51} (generate card objects for each class id), this field should be populated with objects of the appropriate class id, metadata of each object should be default for its class_id


INFORMATION THAT SHOULD NOT BE FED TO THE MODEL:
swap_count = 0 This field should increment everytime a SWAP action completes and reset to 0 anytime a SelectBlind, PlayHand or DiscardHand action completes
Boolean deck_detected = false, dummy value for now
last_swap = SWAP_1_5 (default: null)



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
tarots and spectrals that need exactly one card selected:
263	c_talisman
249	c_aura
252	c_deja_vu
264	c_trance
259	c_medium
251	c_cryptid
298	c_chariot
310	c_lovers
317	c_tower
300	c_devil
309 c_justice

tarots and spectrals that need up to 2 cards selected:
302	c_empress
304	c_hanged_man
305	c_heirophant
311	c_magician
314	c_strength

tarots and spectrals that need up to 3 cards selected:
312	c_moon
313	c_star
315	c_sun
319	c_world


tarots that need exactly 2 cards selected:
299	c_death

spectrals and tarots that need jokers_current >= 1:
248	c_ankh
256	c_hex
318	c_wheel_of_fortune


spectrals and tarots that need jokers_current < jokers_total:
265	c_wraith
262	c_soul
308	c_judgement


tarots that need last_tarot_planet != 303
303	c_fool


tarots that need consumables_current < consumables_total:
307	c_high_priestess
301	c_emperor


TAROTS THAT CAN BE BOUGHT AND USED IN THE SHOP:
	303	c_fool IF last_tarot_planet != 303
	THE H

Tarots that dont need anything:
316	c_temperance
306	c_hermit

Tarots and spectrals that can be used in shop:
248	c_ankh
256	c_hex
265	c_wraith
301	c_emperor
303	c_fool
306	c_hermit
307	c_high_priestess
308	c_judgement
316	c_temperance
318	c_wheel_of_fortune



BuyAndUseShopConsumable(i) - only in in_shop page, generate BuyAndUseShopConsumable(i) for each in ShopOfferings where isConsumable=true
	SAME LOGIC AS UseConsumable (note selected card will always be 0)

SelectPackItem(i) - only in in_JokerStandardPlanet_Pack page, generate SelectPackItem(i) for each in CurrentHandOrPackOfferings
	if Zone CurrentHandOrPackOfferingsSelected[0] is class id between [80 - 229] and ocr jokers_current>=jokers_total
		MASK
	-- depricated: LOGIC IS THE SAME AS USE CONSUMABLE, EXCEPT INSTEAD OF CurrentConsumablesSelected we will use CurrentHandOrPackOfferingsSelected. This is confusing because CurrentHandOrPackOfferingsSelected for this event does not contain the " and instead of looking at CurrentHandOrPackOfferingsSelected for selected cards, we are using it for pack offerings. For looking at selected cards, we will use TarotSpectralHandSelected and TarotSpectralHand.


SelectCard(i) only in In_Blind or In_TarotSpectral_Pack, generate SelectCard(i) for each in CurrentHand

CashOut - only in cash_out page




BuyShopItem(i) - only in in_shop page, generate BuyShopItem(i) for each in ShopOfferings
	if(dollars < shop_offerings(i).cost)
		MASK
	if(isJoker(shop_offerings(i).class_id) & jokers_current >= jokers_total & shop_offerings(i).edition != negative)
		MASK
	if(isConsumable(shop_offerings(i).class_id) & consumables_current >= consumables_total)
		MASK


LeaveShop - only in in_shop page
SkipPack - only in in_tarotspectral_pack and In_JokerStandardPlanet_Pack page

SellItem(i) - always available. for each in current_jokers & current_consumables generate SellItem action option
	if(item(i).sticker == class_id 369)
		MASK


RerollShop - only in in_shop page
	if(reroll_price > dollars)
		MASK


SWAP(i, j) - 

Let there be N joker slots. (each joker slot = 1 object in CurrentJokersAll), indexed

0, 1, 2, ..., N - 1

A swap action is defined by choosing two distinct joker slots:

SWAP_i_j

where:

0 ≤ i < j < N

We require i < j because swapping slot i with slot j is the same as swapping slot j with slot i:

SWAP_i_j = SWAP_j_i

So each unordered pair of distinct slots corresponds to exactly one unique swap action.

Therefore, the set of all joker swap actions is:

{ SWAP_i_j | 0 ≤ i < j < N }

The number of such actions is the number of ways to choose 2 slots from N:

C(N, 2) = N(N - 1) / 2

and 

if swap_count >= (Math.max(0, jokers_current - 1)) * 2.5
	MASK 

if last_swap == SWAP_I_J
	MASK

MASK if i >= jokers_current
MASK if j >= jokers_current
MASK if jokers_current < 2

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