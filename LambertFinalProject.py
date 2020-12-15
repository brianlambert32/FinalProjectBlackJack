import random
import numpy as np
from collections import defaultdict
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Dropout
import tensorflow
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
import csv

players = 1
nod = input("How many decks: 1-5 ")
num_decks = int(nod)
tc = num_decks*52

card_types = ['A',2,3,4,5,6,7,8,9,10,10,10,10]

bust_input = input("Bust number: 21 or 25")
bust_amount = int(bust_input)

model_type = input("Which method: User, Baseline, AI")
if model_type =="User":
    #INITIALIZING DECK OF CARDS FROM USER INPUT
    def make_decks(num_decks, card_types):
        new_deck = []
        for i in range(num_decks):
            for j in range(4):
                new_deck.extend(card_types)
        random.shuffle(new_deck)
        return new_deck

    #deck = make_decks(num_decks, card_types)
    #print(deck)

    #DEFINING ACE VALUES BASED ON CARD TOTAL

    def get_ace_values(temp_list):
        sum_array = np.zeros((2**len(temp_list), len(temp_list)))
        # This loop gets the permutations
        for i in range(len(temp_list)):
            n = len(temp_list) - i
            half_len = int(2**n * 0.5)
            for rep in range(int(sum_array.shape[0]/half_len/2)):
                sum_array[rep*2**n : rep*2**n+half_len, i] = 1
                sum_array[rep*2**n+half_len : rep*2**n+half_len*2, i] = 11
        # Only return values that are valid (<=21)
        return [int(s) for s in np.sum(sum_array, axis=1)]

    def ace_values(num_aces):
        temp_list = []
        for i in range(num_aces):
            temp_list.append([1,11])
        return get_ace_values(temp_list)



    def total_up(hand):
        aces = 0
        total = 0
        for card in hand:
            if card != 'A':
                total += card
            else:
                aces += 1

        ace_value_list = ace_values(aces)
        final_totals = [i+total for i in ace_value_list if i+total<=bust_amount]
        
        if final_totals == []:
            return min(ace_value_list) + total
        else:
            return max(final_totals)


    dealer_card_feature = []
    player_card_feature = []
    player_results = []


    #HUMAN VS DEALER GAME:

    blackjack = set(['A',10])
    dealer_cards = make_decks(num_decks, card_types)
    while len(dealer_cards) > tc - 4:
        
        curr_player_results = np.zeros((1,players))
        dealer_hand = []
        player_hands = [[] for player in range(players)]
        
        # Deal FIRST card
        for player, hand in enumerate(player_hands):
            player_hands[player].append(dealer_cards.pop(0))
        dealer_hand.append(dealer_cards.pop(0))
        print("Player's First Card:", *player_hands, "Dealer's First Card:", *dealer_hand)
        # Deal SECOND card
        for player, hand in enumerate(player_hands):
            player_hands[player].append(dealer_cards.pop(0))
        dealer_hand.append(dealer_cards.pop(0))
        print("Player's Cards:", *player_hands)

    # Dealer checks for 21
    if set(dealer_hand) == blackjack:
        for player in range(players):
            if set(player_hands[player]) != blackjack:
                curr_player_results[0,player] = -1
                print("Dealer has Blackjack. You lose!")
            else:
                curr_player_results[0,player] = 0
    else:
        for player in range(players):
            # Players check for 21
            if set(player_hands[player]) == blackjack:
                curr_player_results[0,player] = 1
                print("Player has Blackjack. You win!")
            else:
                    
                    while (total_up(player_hands[player]) != bust_amount):
                        h_s = input("Would you like to hit or stay?: h or s ")
                        if h_s == 'h':
                            player_hands[player].append(dealer_cards.pop(0))
                            print("Player's Cards: ", *player_hands)
                            print("Player Total: ", total_up(player_hands[player]))
                        elif h_s == 's':
                            break


    # Dealer hits based on the rules
    while total_up(dealer_hand) < 17:
        dealer_hand.append(dealer_cards.pop(0))
        print("Final Dealer Hand", *dealer_hand)
        # Compare dealer hand to players hand but first check if dealer busted
        if total_up(dealer_hand) > bust_amount:
            for player in range(players):
                if curr_player_results[0,player] != -1:
                    curr_player_results[0,player] = 1
    else:
        for player in range(players):
            if total_up(player_hands[player]) > total_up(dealer_hand):
                if total_up(player_hands[player]) <= bust_amount:
                    curr_player_results[0,player] = 1
                    print("Player Wins!")
                elif total_up(player_hands[player]) == total_up(dealer_hand):
                    curr_player_results[0,player] = 0
                    print("Player and Dealer Push")
                else:
                    curr_player_results[0,player] = -1
                    print("Player Loses.")
    print('player: ' + str(total_up(player_hands[player])),
          'dealer: ' + str(total_up(dealer_hand)),
          'result: ' + str(curr_player_results)
          )


if model_type =="Baseline":

    valid_actions = [0,1]
    #INITIALIZING DECK OF CARDS FROM USER INPUT
    def make_decks(num_decks, card_types):
        new_deck = []
        for i in range(num_decks):
            for j in range(4):
                new_deck.extend(card_types)
        random.shuffle(new_deck)
        return new_deck

    #deck = make_decks(num_decks, card_types)
    #print(deck)

    #DEFINING ACE VALUES BASED ON CARD TOTAL

    def get_ace_values(temp_list):
        sum_array = np.zeros((2**len(temp_list), len(temp_list)))
        # This loop gets the permutations
        for i in range(len(temp_list)):
            n = len(temp_list) - i
            half_len = int(2**n * 0.5)
            for rep in range(int(sum_array.shape[0]/half_len/2)):
                sum_array[rep*2**n : rep*2**n+half_len, i] = 1
                sum_array[rep*2**n+half_len : rep*2**n+half_len*2, i] = 11
        # Only return values that are valid (<=21)
        return list(set([int(s) for s in np.sum(sum_array, axis=1) if s<=bust_amount]))

    def ace_values(num_aces):
        temp_list = []
        for i in range(num_aces):
            temp_list.append([1,11])
        return get_ace_values(temp_list)



    def total_up(hand):
        aces = 0
        total = 0
        for card in hand:
            if card != 'A':
                total += card
            else:
                aces += 1

        ace_value_list = ace_values(aces)
        final_totals = [i+total for i in ace_value_list if i+total<=bust_amount]
        
        if final_totals == []:
            return min(ace_value_list) + total
        else:
            return max(final_totals)


    dealer_card_feature = []
    player_card_feature = []
    player_results = []


    #Baseline VS DEALER GAME:

    blackjack = set(['A',10])
    dealer_cards = make_decks(num_decks, card_types)
    while len(dealer_cards) > tc - 4:
        
        curr_player_results = np.zeros((1,players))
        dealer_hand = []
        player_hands = [[] for player in range(players)]
        
        # Deal FIRST card
        for player, hand in enumerate(player_hands):
            player_hands[player].append(dealer_cards.pop(0))
        dealer_hand.append(dealer_cards.pop(0))
        print("Player's First Card:", *player_hands, "Dealer's First Card:", *dealer_hand)
        # Deal SECOND card
        for player, hand in enumerate(player_hands):
            player_hands[player].append(dealer_cards.pop(0))
        dealer_hand.append(dealer_cards.pop(0))
        print("Player's Cards:", *player_hands)

    # Dealer checks for 21
    if set(dealer_hand) == blackjack:
        for player in range(players):
            if set(player_hands[player]) != blackjack:
                curr_player_results[0,player] = -1
                print("Dealer has Blackjack. You lose!")
            else:
                curr_player_results[0,player] = 0
    else:
        for player in range(players):
            # Players check for 21
            if set(player_hands[player]) == blackjack:
                curr_player_results[0,player] = 1
                print("Player has Blackjack. You win!")
            else:
                    
                    while (total_up(player_hands[player]) != bust_amount):
                        h_s = random.choice(valid_actions)
                        if h_s == 0:
                            print("Player Hits")
                            player_hands[player].append(dealer_cards.pop(0))
                            print("Player's Cards: ", *player_hands)
                            print("Player Total: ", total_up(player_hands[player]))
                        elif h_s == 1:
                            break


    # Dealer hits based on the rules
    while total_up(dealer_hand) < 17:
        dealer_hand.append(dealer_cards.pop(0))
        print("Final Dealer Hand", *dealer_hand)
        # Compare dealer hand to players hand but first check if dealer busted
        if total_up(dealer_hand) > bust_amount:
            for player in range(players):
                if curr_player_results[0,player] != -1:
                    curr_player_results[0,player] = 1
    else:
        for player in range(players):
            if total_up(player_hands[player]) > total_up(dealer_hand):
                if total_up(player_hands[player]) <= bust_amount:
                    curr_player_results[0,player] = 1
                    print("Player Wins!")
                elif total_up(player_hands[player]) == total_up(dealer_hand):
                    curr_player_results[0,player] = 0
                    print("Player and Dealer Push")
                else:
                    curr_player_results[0,player] = -1
                    print("Player Loses.")
    print('player: ' + str(total_up(player_hands[player])),
          'dealer: ' + str(total_up(dealer_hand)),
          'result: ' + str(curr_player_results)
          )



if model_type =="AI":

#INITIALIZING DECK OF CARDS FROM USER INPUT
    def make_decks(num_decks, card_types):
        new_deck = []
        for i in range(num_decks):
            for j in range(4):
                new_deck.extend(card_types)
        random.shuffle(new_deck)
        return new_deck

    def get_ace_values(temp_list):
        sum_array = np.zeros((2**len(temp_list), len(temp_list)))
        # This loop gets the permutations
        for i in range(len(temp_list)):
            n = len(temp_list) - i
            half_len = int(2**n * 0.5)
            for rep in range(int(sum_array.shape[0]/half_len/2)):
                sum_array[rep*2**n : rep*2**n+half_len, i] = 1
                sum_array[rep*2**n+half_len : rep*2**n+half_len*2, i] = 11
        # Only return values that are valid (<=21)
        return [int(s) for s in np.sum(sum_array, axis=1)]



    def ace_values(num_aces):
        temp_list = []
        for i in range(num_aces):
            temp_list.append([1,11])
        return get_ace_values(temp_list)



    def total_up(hand):
        aces = 0
        total = 0
        for card in hand:
            if card != 'A':
                total += card
            else:
                aces += 1

        ace_value_list = ace_values(aces)
        final_totals = [i+total for i in ace_value_list if i+total<=bust_amount]
        
        if final_totals == []:
            return min(ace_value_list) + total
        else:
            return max(final_totals)

    def play_game(dealer_hand, player_hands, blackjack, curr_player_results, dealer_cards, hit_stay, card_count, dealer_bust):
        action = 0
        # Dealer checks for 21
        if set(dealer_hand) == blackjack:
            for player in range(players):
                if set(player_hands[player]) != blackjack:
                    curr_player_results[0,player] = -1
                else:
                    curr_player_results[0,player] = 0
        else:
            for player in range(players):
                # Players check for 21
                if set(player_hands[player]) == blackjack:
                    curr_player_results[0,player] = 1
                else:
                    
                    if (hit_stay >= 0.5) and (total_up(player_hands[player]) != bust_amount):
                        player_hands[player].append(dealer_cards.pop(0))
                        card_count[player_hands[player][-1]] += 1
                        
                        action = 1
                        live_total.append(total_up(player_hands[player]))
                        if total_up(player_hands[player]) > bust_amount:
                            curr_player_results[0,player] = -1

    # Dealer hits based on the rules
        card_count[dealer_hand[-1]] += 1
        while total_up(dealer_hand) < 17:
            dealer_hand.append(dealer_cards.pop(0))
            card_count[dealer_hand[-1]] += 1
            print("Final Dealer Hand", *dealer_hand)
        if total_up(dealer_hand) > bust_amount:
            dealer_bust.append(1)
            for player in range(players):
                if curr_player_results[0,player] != -1:
                    curr_player_results[0,player] = 1

        else:
            dealer_bust.append(0)
            for player in range(players):
                if total_up(player_hands[player]) > total_up(dealer_hand):
                    if total_up(player_hands[player]) <= bust_amount:
                        curr_player_results[0,player] = 1
                elif total_up(player_hands[player]) == total_up(dealer_hand):
                    curr_player_results[0,player] = 0
                else:
                    curr_player_results[0,player] = -1

        return curr_player_results, dealer_cards, action, card_count, dealer_bust

    stacks = 1
    dealer_card_feature = []
    player_card_feature = []
    player_live_total = []
    player_live_action = []
    player_results = []
    dealer_bust = []

    first_game = True
    prev_stack = 0
    stack_num_list = []
    new_stack = []
    card_count_list = []
    games_played_with_stack = []

    for stack in range(stacks):
        games_played = 0

    card_count = {2: 0,
        3: 0,
            4: 0,
                5: 0,
                    6: 0,
                        7: 0,
                            8: 0,
                                9: 0,
                                    10: 0,
                                        'A': 0}

    blackjack = set(['A',10])
    dealer_cards = make_decks(num_decks, card_types)
    while len(dealer_cards) > tc - 4:
            
        curr_player_results = np.zeros((1,players))

        dealer_hand = []
        player_hands = [[] for player in range(players)]
        live_total = []
        live_action = []
            
            # Deal FIRST card
        for player, hand in enumerate(player_hands):
            player_hands[player].append(dealer_cards.pop(0))
            card_count[player_hands[player][-1]] += 1

        dealer_hand.append(dealer_cards.pop(0))
        card_count[dealer_hand[-1]] += 1
        print("Player's First Card:", *player_hands, "Dealer's First Card:", *dealer_hand)


    # Deal SECOND card
    for player, hand in enumerate(player_hands):
        player_hands[player].append(dealer_cards.pop(0))
        card_count[player_hands[player][-1]] += 1
            
    dealer_hand.append(dealer_cards.pop(0))
    print("Player's Cards:", *player_hands)
    print(card_count)

    # Record the player's live total after cards are dealt
    live_total.append(total_up(player_hands[player]))
        
    if stack < stacks/2:
        hit_stay = 1
    else:
        hit_stay = 0

    curr_player_results, dealer_cards, action, card_count, dealer_bust = play_game(dealer_hand, player_hands, blackjack, curr_player_results, dealer_cards, hit_stay, card_count, dealer_bust)
            
    dealer_card_feature.append(dealer_hand[0])
    player_card_feature.append(player_hands)
    player_results.append(list(curr_player_results[0]))
    player_live_total.append(live_total)
    player_live_action.append(action)
        
    if stack != prev_stack:
        new_stack.append(1)
    else:
        new_stack.append(0)
        if first_game == True:
            first_game = False
        else:
            games_played += 1

    stack_num_list.append(stack)
    games_played_with_stack.append(games_played)
    card_count_list.append(card_count.copy())
    prev_stack = stack

    model_df = pd.DataFrame()
    model_df['dealer_card'] = dealer_card_feature
    model_df['player_total_initial'] = [total_up(i[0][0:2]) for i in player_card_feature]
    model_df['hit?'] = player_live_action

    has_ace = []
    for i in player_card_feature:
        if ('A' in i[0][0:2]):
            has_ace.append(1)
        else:
            has_ace.append(0)
    model_df['has_ace'] = has_ace


    dealer_card_num = []
    for i in model_df['dealer_card']:
        if i=='A':
            dealer_card_num.append(11)
        else:
            dealer_card_num.append(i)
    model_df['dealer_card_num'] = dealer_card_num

    model_df['Y'] = [i[0] for i in player_results]
    lose = []
    for i in model_df['Y']:
        if i == -1:
            lose.append(1)
        else:
            lose.append(0)
    model_df['lose'] = lose

    correct = []
    for i, val in enumerate(model_df['lose']):
        if val == 1:
            if player_live_action[i] == 1:
                correct.append(0)
            else:
                correct.append(1)
        else:
            if player_live_action[i] == 1:
                correct.append(1)
            else:
                correct.append(0)
    model_df['correct_action'] = correct

    card_count_df = pd.concat([pd.DataFrame(new_stack, columns=['new_stack']),
                               pd.DataFrame(games_played_with_stack, columns=['games_played_with_stack']),
                               pd.DataFrame.from_dict(card_count_list),
                               pd.DataFrame(dealer_bust, columns=['dealer_bust'])], axis=1)

    model_df = pd.concat([model_df, card_count_df], axis=1)

    data = 1 - (model_df.groupby(by='dealer_card').sum()['lose'] /\
                model_df.groupby(by='dealer_card').count()['lose'])
                
   
    data = 1 - (model_df.groupby(by='player_total_initial').sum()['lose'] /\
                model_df.groupby(by='player_total_initial').count()['lose'])



    model_df.groupby(by='has_ace').sum()['lose'] / model_df.groupby(by='has_ace').count()['lose']

    feature_list = [i for i in model_df.columns if i not in ['dealer_card',
                                                             'Y','lose',
                                                             'correct_action',
                                                             'dealer_bust',
                                                             'dealer_bust_pred',
                                                             'new_stack', 'games_played_with_stack',
                                                             2,3,4,5,6,7,8,9,10,'A',
                                                             'blackjack?'
                                                             ]]
    train_X = np.array(model_df[feature_list])
    train_Y = np.array(model_df['correct_action']).reshape(-1,1)

    print(feature_list)
    print('\n')


    # Set up a neural net with 5 layers
    model = Sequential()

    model.add(Dense(16))
    model.add(Dense(128))
    model.add(Dense(32))
    model.add(Dense(8))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    model.fit(train_X, train_Y, epochs=200, batch_size=256, verbose=1)

    pred_Y_train = model.predict(train_X)
    actuals = train_Y[:,-1]

    def function(x):
        if x == bust_amount:
            return 1
        else:
            return 0

    model_df['blackjack?'] = model_df['player_total_initial'].apply(function)

    bust_features = [2,3,4,5,6,7,8,9,10,'A','dealer_card_num']

    feature_list_bust = [i for i in bust_features if i not in ['dealer_bust']]
    train_X_bust = np.array(model_df[feature_list_bust])
    train_Y_bust = np.array(model_df['blackjack?']).reshape(-1,1)

    print(feature_list_bust)
    print('\n')

    model_bust = Sequential()
    model_bust.add(Dense(train_X_bust.shape[1]))
    model_bust.add(Dense(128))
    model_bust.add(Dense(32, activation='relu'))
    model_bust.add(Dense(8))
    model_bust.add(Dense(1, activation='sigmoid'))
    model_bust.compile(loss='binary_crossentropy', optimizer='sgd')
    model_bust.fit(train_X_bust, train_Y_bust, epochs=200, batch_size=256, verbose=1)

    pred_Y_train_bust = model_bust.predict(train_X_bust)
    actuals = train_Y_bust[:,-1]



    def model_decision(model, player_sum, has_ace, dealer_card_num, new_stack, games_played, card_count):
        input_array = np.array([player_sum, 0, has_ace,
                                dealer_card_num, new_stack,
                                games_played]).reshape(1,-1)
        cc_array = pd.DataFrame.from_dict([card_count])
        input_array = np.concatenate([input_array, cc_array], axis=1)
        predict_correct = model.predict(input_array)
        if predict_correct >= 0.52:
            return 1, predict_correct
        else:
            return 0, predict_correct

    def model_decision_old(model, player_sum, has_ace, dealer_card_num):
        input_array = np.array([player_sum, 0, has_ace, dealer_card_num]).reshape(1,-1)
        predict_correct = model.predict(input_array)
        if predict_correct >= 0.52:
            return 1
        else:
            return 0


    def bust_Z_score(pred, pred_mean, pred_std):
        return (pred - pred_mean)/pred_std

    pred_mean = model_bust.predict(train_X_bust).mean()
    pred_std = model_bust.predict(train_X_bust).std()

    nights = 10
    bankrolls = []
    for night in range(nights):
        dollars = 10000
        bankroll = []
        stacks = 10
        players = 1

    
        card_types = ['A',2,3,4,5,6,7,8,9,10,10,10,10]
    
        dealer_card_feature = []
        player_card_feature = []
        player_live_total = []
        player_live_action = []
        player_results = []
    
        first_game = True
        prev_stack = 0
        stack_num_list = []
        new_stack = []
        card_count_list = []
        games_played_with_stack = []
            
        for stack in range(stacks):
            games_played = 0
        
            if stack != prev_stack:
                temp_new_stack = 1
            else:
                temp_new_stack = 0
        
        # Make a dict for keeping track of the count for a stack
        card_count = {2: 0,
            3: 0,
                4: 0,
                    5: 0,
                        6: 0,
                            7: 0,
                                8: 0,
                                    9: 0,
                                        10: 0,
                                            'A': 0}
                                                
        blackjack = set(['A',10])
        dealer_cards = make_decks(num_decks, card_types)
        while len(dealer_cards) > tc - 4:
            multiplier = 1
            curr_player_results = np.zeros((1,players))
    
            dealer_hand = []
            player_hands = [[] for player in range(players)]
            live_total = []
            live_action = []
            
            # Record card count
            cc_array_bust = pd.DataFrame.from_dict([card_count])
            
            # Deal FIRST card
            for player, hand in enumerate(player_hands):
                player_hands[player].append(dealer_cards.pop(0))
            dealer_hand.append(dealer_cards.pop(0))
            print("Dealer's Card: ", *dealer_hand)
            # Deal SECOND card
            for player, hand in enumerate(player_hands):
                player_hands[player].append(dealer_cards.pop(0))
                print("AI's Cards:", *player_hands)
            dealer_hand.append(dealer_cards.pop(0))
            
            # Record the player's live total after cards are dealt
            live_total.append(total_up(player_hands[player]))
            action = 0
                
            if set(dealer_hand) == blackjack:
                for player in range(players):
                    if set(player_hands[player]) != blackjack:
                        curr_player_results[0,player] = -1
                    else:
                        curr_player_results[0,player] = 0
            else:
                for player in range(players):
                    # Players check for 21
                    if set(player_hands[player]) == blackjack:
                        curr_player_results[0,player] = 1
                        multiplier = 1.25
                    else:
                        # Neural net decides whether to hit or stay
                        if 'A' in player_hands[player][0:2]:
                            ace_in_hand = 1
                        else:
                            ace_in_hand = 0
                        if dealer_hand[0] == 'A':
                            dealer_face_up_card = 11
                        else:
                            dealer_face_up_card = dealer_hand[0]
                    
                        while (model_decision_old(model, total_up(player_hands[player]),
                                                  ace_in_hand, dealer_face_up_card,
                                                  #temp_new_stack, games_played,
                                                  #card_count
                                                  ) == 1) and (total_up(player_hands[player]) != bust_amount):
                            player_hands[player].append(dealer_cards.pop(0))
                            print("AI's Final Cards: ", *player_hands)
                            action = 1
                            live_total.append(total_up(player_hands[player]))
                            if total_up(player_hands[player]) > bust_amount:
                                curr_player_results[0,player] = -1
                                break
                                    
            while total_up(dealer_hand) < 17:
                dealer_hand.append(dealer_cards.pop(0))
    # Compare dealer hand to players hand but first check if dealer busted
            if total_up(dealer_hand) > bust_amount:
                for player in range(players):
                    if curr_player_results[0,player] != -1:
                        curr_player_results[0,player] = 1
            else:
                for player in range(players):
                    if total_up(player_hands[player]) > total_up(dealer_hand):
                        if total_up(player_hands[player]) <= bust_amount:
                            curr_player_results[0,player] = 1
                    elif total_up(player_hands[player]) == total_up(dealer_hand):
                        curr_player_results[0,player] = 0
                    else:
                        curr_player_results[0,player] = -1
    
            input_array = np.concatenate([cc_array_bust,
                                      np.array(dealer_face_up_card).reshape(1,-1)], axis=1)
            bust_pred = model_bust.predict(input_array)
            bust_Z = bust_Z_score(bust_pred, pred_mean, pred_std)
            
            if bust_Z >= 0:
                bet = 100#*(1 + bust_Z_score)
            else:
                bet = 100
            dollars+=curr_player_results[0,player]*bet*multiplier
            bankroll.append(dollars)
                
            dealer_card_feature.append(dealer_hand[0])
            player_card_feature.append(player_hands)
            player_results.append(list(curr_player_results[0]))
            player_live_total.append(live_total)
            player_live_action.append(action)
            
            if stack != prev_stack:
                new_stack.append(1)
            else:
                new_stack.append(0)
                if first_game == True:
                    first_game = False
                else:
                    games_played += 1
            
            stack_num_list.append(stack)
            games_played_with_stack.append(games_played)
            card_count_list.append(card_count.copy())
            prev_stack = stack

        model_df_smart = pd.DataFrame()
        model_df_smart['dealer_card'] = dealer_card_feature
        model_df_smart['player_total_initial'] = [total_up(i[0][0:2]) for i in player_card_feature]
        model_df_smart['hit?'] = player_live_action
        
        has_ace = []
        for i in player_card_feature:
            if ('A' in i[0][0:2]):
                has_ace.append(1)
            else:
                has_ace.append(0)
        model_df_smart['has_ace'] = has_ace

        dealer_card_num = []
        for i in model_df_smart['dealer_card']:
            if i=='A':
                dealer_card_num.append(11)
            else:
                dealer_card_num.append(i)
        model_df_smart['dealer_card_num'] = dealer_card_num
    
        model_df_smart['Y'] = [i[0] for i in player_results]
        lose = []
        for i in model_df_smart['Y']:
            if i == -1:
                lose.append(1)
            else:
                lose.append(0)
        model_df_smart['lose'] = lose
    
        bankrolls.append(bankroll)
   





    print(dealer_card_feature, player_card_feature, player_live_total, player_live_action,player_results, dealer_bust)

























