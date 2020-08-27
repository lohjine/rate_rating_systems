import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

import glicko2
from skills import Match, Team, Matches
from skills.glicko import GlickoGameInfo, GlickoCalculator
from elo import Elo
from elo import expect as elo_expect
from trueskill import Rating as tsrating, rate_1vs1 as tsrate_1vs1
import trueskill


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def footrule(a,b):
    a = np.array(a)
    b = np.array(b)
    error = 0
    for idx, i in enumerate(a):
        error += np.abs(idx - np.where(b==i)[0][0])
    return error/len(a)

ratingevaluationdataset = pd.read_csv(r"dataset\ChessRatings2\primary_training_part1_processed.csv")

"""
All rating systems should implement 1 class and 3 functions


class PlayerClass:
    '''
    Class should have rating attribute, as well as any additional attributes that it needs for rating calculations.
    '''
    def __init__(self, rating=1000):
        self.rating = rating


def 1v1(a,b):
    '''
    Does rating calculation for player_a winning a round against player_b.

    Returns modified player_a, player_b class.
    '''
    a.rating += 10
    b.rating -= 10
    return a,b


def 1v1draw(a,b):
    '''
    Does rating calculation for player_a drawing a round against player_b.

    Returns modified player_a, player_b class.
    '''
    a.rating = a.rating
    b.rating = b.rating
    return a,b


def Pwin(a,b):
    '''
    Does a prediction for probability of player_a winning against player_b

    Returns probability from 0 to 1.
    '''
    if a.rating == b.rating:
        return 0.5
    elif a.rating > b.rating:
        return 0.6
    else:
        return 0.4


"""

######################
# Trueskill
######################


tsrating()


class tsRating(tsrating):
    def __init__(self, rating=25):
        super().__init__(rating)
        self.rating = self.mu


def ts_rate_1vs1(a, b):
    a, b = tsrate_1vs1(a, b)
    return tsRating(a), tsRating(b)


def ts_rate_1vs1_draw(a, b):
    a, b = tsrate_1vs1(a, b, drawn=True)
    return tsRating(a), tsRating(b)


def ts_Pwin(rA, rB):
    # obtained from https://github.com/sublee/trueskill/issues/1#issuecomment-10491635
    deltaMu = rA.mu - rB.mu
    rsss = np.sqrt(rA.sigma**2 + rB.sigma**2)
    return trueskill.TrueSkill().cdf(deltaMu / rsss)


######################
# ELO
######################


class EloRating:
    def __init__(self, rating=1200, k_factor=32):
        self.rating = rating
        self.k_factor = k_factor

    def __repr__(self):
        return str(f'EloRating(rating={self.rating})')


def Elo1v1(a, b):
    a.rating, _ = Elo(k_factor=a.k_factor).rate_1vs1(a.rating, b.rating)
    _, b.rating = Elo(k_factor=b.k_factor).rate_1vs1(a.rating, b.rating)
    return a, b


def Elo1v1draw(a, b):
    a.rating, _ = Elo(k_factor=a.k_factor).rate_1vs1(a.rating, b.rating, drawn=True)
    _, b.rating = Elo(k_factor=b.k_factor).rate_1vs1(a.rating, b.rating, drawn=True)
    return a, b


def EloPwin(a, b):
    return elo_expect(a.rating, b.rating)


######################
# Glicko
######################

glickocalculator = GlickoCalculator()


class glicko():
    def __init__(self, rating=1500, stdev=200):
        self.rating = rating
        self.stdev = stdev

    def __repr__(self):
        return str(f'glickoRating(mean={self.rating},stdev={self.stdev})')


def glicko1v1(a, b):
    player1 = Team({1: (a.rating, a.stdev)})
    player2 = Team({2: (b.rating, b.stdev)})
    matches = Matches([Match([player1, player2], [0, 1])])  # this is player 1 wins
    new_ratings = glickocalculator.new_ratings(matches, 1, GlickoGameInfo())

    return glicko(rating=new_ratings[0][new_ratings.player_by_id(1)].mean, stdev=new_ratings[0][new_ratings.player_by_id(1)].stdev), \
        glicko(rating=new_ratings[1][new_ratings.player_by_id(2)].mean, stdev=new_ratings[1][new_ratings.player_by_id(2)].stdev)


def glicko1v1draw(a, b):
    player1 = Team({1: (a.rating, a.stdev)})
    player2 = Team({2: (b.rating, b.stdev)})
    matches = Matches([Match([player1, player2], [0, 0])])  # this is draw
    new_ratings = glickocalculator.new_ratings(matches, 1, GlickoGameInfo())

    return glicko(rating=new_ratings[0][new_ratings.player_by_id(1)].mean, stdev=new_ratings[0][new_ratings.player_by_id(1)].stdev), \
        glicko(rating=new_ratings[1][new_ratings.player_by_id(2)].mean, stdev=new_ratings[1][new_ratings.player_by_id(2)].stdev)


def glickoPwin(a, b):
    return glickocalculator.expected_score(a.rating, b.rating, GlickoGameInfo())


######################
# Glicko2
######################

glicko2.Player()


def glicko21v1(a, b):
    ori_a_rating = a.rating
    ori_a_rd = a.rd
    a.update_player([b.rating], [b.rd], [1])
    b.update_player([ori_a_rating], [ori_a_rd], [0])

    return a, b


def glicko21v1draw(a, b):
    ori_a_rating = a.rating
    ori_a_rd = a.rd
    a.update_player([b.rating], [b.rd], [0.5])
    b.update_player([ori_a_rating], [ori_a_rd], [0.5])

    return a, b


def glicko2Pwin(a, b):
    return glickocalculator.expected_score(a.rating, b.rating, GlickoGameInfo())


######################
# Scenario 1
######################

class ConvergenceEvaluation:
    """
    A synthetic simulation of players and matchmaking.

    Player ratings are generated from a normal distribution, and player standard deviations are generated from a uniform distribution.

    When two players match against each other, generate 2 numbers for each distribution, higher number wins. If ranges overlap, draw.


    Results stored:
        1. kendall-tau
        2. log loss
        3. win-loss for each player
        4. win-loss-ratio interquantile range
        5. median win-loss-ratio

    """

    def __init__(self, genplayerfunc, matchfunc, drawmatchfunc, winPcalcfunc, player_count=1000, player_gen_mean=3000,
                 player_gen_sd=2000, player_sd=10, swap_passes=50,
                 seed=42):
        self.genplayerfunc = genplayerfunc
        self.matchfunc = matchfunc
        self.drawmatchfunc = drawmatchfunc
        self.winPcalcfunc = winPcalcfunc

        self.player_count = player_count
        self.swap_passes = swap_passes

        self.eval_player_pool = {}
        self.idx = 0

        self.loglosses = []
        self.kendalltaus = []
        self.correlations = []
        self.footrules = []

        # create pool of players
        if not player_count % 2 == 0:
            raise ValueError('Player count must be even')
        np.random.seed(seed)

        # this cannot be sorted, otherwise first round pairing will be deterministic
        self.true_player_pool_means = np.random.rand(player_count) * 1000
        self.true_player_pool_means = np.random.normal(player_gen_mean, player_gen_sd, size=player_count)
        self.true_player_pool_stdev = np.random.rand(player_count) * 10

        self.kendalltaus_order = np.argsort(self.true_player_pool_means)

        self.player_win_loss = []
        self.player_wlr_iqr = []
        self.player_wlr_median = []

        # eval_player_pool id corresponds to true_player_pool_means id
        for i in range(player_count):
            self.eval_player_pool[i] = self.genplayerfunc()
            self.player_win_loss.append([0, 0])

    def step(self, debug=False):
        """
        Each step consists of 1 round of matchmaking and games for each player.


        Matchmaking
        ---
        Goal is to pair up close scores to play against each other, with some variance.

        1. Players are sorted according to their current matchmaking rating (e.g. [1400, 1450, 1490, 1520, ...])
        2. A running window of 10% of players is performed across the array of players, where the indices of players are shuffled uniformly within the window at each step. (e.g. after first step, [1490, 1400, 1520, 1450, ...(first 50), 1900, 1910, 1930, ...] )
        3. Players are matched against their adjacent index. (e.g. [1490 vs 1400, 1520 vs 1450, ...])

        """

        sorted_order = sorted(self.eval_player_pool, key=lambda x: self.eval_player_pool[x].rating)

        round_loglosses = []

        # Generate shuffled list of players
        pairings = np.arange(0,self.player_count)

        for i in range(self.swap_passes):
            pre_rand = np.random.randint(0,2,len(sorted_order))
            if i%2 == 0:
                for j in range(self.player_count-1):
                    if pre_rand[j] == 0:
                        pairings[j], pairings[j+1] = pairings[j+1], pairings[j]
            else:
                for j in range(self.player_count-1,0,-1):
                    if pre_rand[j] == 0:
                        pairings[j], pairings[j-1] = pairings[j-1], pairings[j]

        interval = 2
        start_iter = 0
        end_iter = len(sorted_order)

        matchups = []

        for i in range(start_iter, end_iter, interval):
            player_a = sorted_order[pairings[i]]
            player_b = sorted_order[pairings[i + 1]]
            matchups.append((player_a, player_b))

        for player_a, player_b in matchups:

            player_a_score = np.random.normal(self.true_player_pool_means[player_a],
                                              self.true_player_pool_stdev[player_a],
                                              size=2)
            player_b_score = np.random.normal(self.true_player_pool_means[player_b],
                                              self.true_player_pool_stdev[player_b],
                                              size=2)

            if max(player_a_score) > max(player_b_score):
                score = 1
            else:
                score = 0

            if score == 1:
                if min(player_a_score) < max(player_b_score):
                    score = 0.5
            else:
                if min(player_b_score) < max(player_a_score):
                    score = 0.5

            win_prob = self.winPcalcfunc(self.eval_player_pool[player_a], self.eval_player_pool[player_b])
            # clamp values to prevent inf/-inf log loss
            win_prob = max(win_prob, 1e-320)
            win_prob = min(win_prob, 1 - 1e-16)
            # clamp to ensure log-loss fairness
            win_prob = min(win_prob, 0.99)
            win_prob = max(win_prob, 0.01)

            logloss = - (score * np.log10(win_prob) + (1 - score) * np.log10(1 - win_prob))

            if np.isnan(logloss):
                raise ValueError

            round_loglosses.append(logloss)

            if debug:
                print(
                    f"{player_a} ({self.eval_player_pool[player_a].mu}) vs {player_b} ({self.eval_player_pool[player_b].mu}), score={score}, win_prob={win_prob}, loss={logloss}")

            # process win/lose/draw
            if score == 1:
                winner, loser = player_a, player_b
                self.eval_player_pool[winner], self.eval_player_pool[loser] = self.matchfunc(
                    self.eval_player_pool[winner], self.eval_player_pool[loser])
                self.player_win_loss[player_a][0] += 1
                self.player_win_loss[player_b][1] += 1
            elif score == 0:
                winner, loser = player_b, player_a
                self.eval_player_pool[winner], self.eval_player_pool[loser] = self.matchfunc(
                    self.eval_player_pool[winner], self.eval_player_pool[loser])
                self.player_win_loss[player_b][0] += 1
                self.player_win_loss[player_a][1] += 1
            elif score == 0.5:
                self.eval_player_pool[player_a], self.eval_player_pool[player_b] = self.drawmatchfunc(
                    self.eval_player_pool[player_a], self.eval_player_pool[player_b])
            else:
                print(self.idx)
                raise ValueError(f'Unexpected Score: {score}')

        self.loglosses.append(np.mean(round_loglosses))
        sorted_order_after = sorted(self.eval_player_pool, key=lambda x: self.eval_player_pool[x].rating)
        self.kendalltaus.append(kendalltau(self.kendalltaus_order, sorted_order_after))
        self.footrules.append(footrule(self.kendalltaus_order, sorted_order_after))

        # correlation measure turns out to be very similar as using kendalltau alone
        self.correlations.append(np.corrcoef(self.kendalltaus_order, sorted_order_after)[0, 1])

        self.idx += 1
        self.player_wlr = []
        for i in self.player_win_loss:
            if i[1] == 0:
                self.player_wlr.append(999)  # cap wlr=999 if loss=0
            else:
                self.player_wlr.append(i[0] / i[1])
        self.player_wlr = np.array(self.player_wlr)

        a, b, c = np.quantile(self.player_wlr, [0.25, 0.75, 0.5])
        self.player_wlr_iqr.append(b - a)
        self.player_wlr_median.append(c)

    def simulate(self, steps=100):
        for i in tqdm(range(steps)):
            self.step()

    def output_results(self, moving_window=100, length=1e9):

        loglosses = np.array(self.loglosses)

        length = min(length, len(loglosses[loglosses != 0]))

        plt.plot(np.round(moving_average(loglosses[loglosses != 0][:length], n=moving_window), decimals=7))
        plt.title('logloss')

        print('Mean logloss:', np.mean(loglosses[loglosses != 0]))

        plt.figure()
        plt.plot(np.round(moving_average(np.array([kt.correlation for kt in self.kendalltaus]), n=moving_window), decimals=7))
        plt.title('kendall-tau')


def compare_results_convergence(list_of_obj, moving_window=100, legend=None, title_details=''):
    """
    Compare results from 1 run each of multiple rating systems.

    """
    plt.figure()
    for obj in list_of_obj:
        plt.plot(np.round(moving_average(np.array([kt.correlation for kt in obj.kendalltaus]), n=moving_window), decimals=7))
    plt.title('kendall-tau ' + title_details)
    if legend:
        plt.legend(legend)
    plt.figure()
    for obj in list_of_obj:
        loglosses = np.array(obj.loglosses)
        plt.plot(np.round(moving_average(loglosses[loglosses != 0], n=moving_window), decimals=7))
    plt.title('logloss ' + title_details)
    if legend:
        plt.legend(legend)


def compare_results_convergence_multi(list_of_obj, legend=None, title_details=''):
    """
    Compare results from multiple runs each of multiple rating systems.

    """
    plt.figure()
    for obj in list_of_obj:

        res = []
        for run in obj:
            res.append(run.footrules)

        res = np.vstack(res)
        res_avg = np.mean(res, axis=0)
        res_high = np.max(res, axis=0)
        res_low = np.min(res, axis=0)

        plt.plot(res_avg)
        plt.fill_between(range(len(res_avg)), res_high, res_low, alpha=0.1)

    plt.title(title_details)
    plt.ylabel('Footrule Error')
    plt.xlabel('Number of rounds')
    if legend:
        plt.legend(legend)

    if headless:
        plt.savefig('img/scenario1.png')

    plt.figure()
    for obj in list_of_obj:

        res = []
        for run in obj:
            res.append(run.footrules)

        res = np.vstack(res)
        res_avg = np.mean(res, axis=0)
        res_high = np.max(res, axis=0)
        res_low = np.min(res, axis=0)

        plt.plot(np.arange(1,res.shape[1])[int(-1*np.round(res.shape[1]/2)):], res_avg[int(-1*np.round(res.shape[1]/2)):])
        plt.fill_between(range(len(res_avg))[int(-1*np.round(res.shape[1]/2)):], res_high[int(-1*np.round(res.shape[1]/2)):], res_low[int(-1*np.round(res.shape[1]/2)):], alpha=0.1)

    plt.title(title_details)
    plt.ylabel('Footrule Error')
    plt.xlabel('Number of rounds')
    if legend:
        plt.legend(legend)

    if headless:
        plt.savefig('img/scenario1_2.png')


######################
# Scenario 2
######################

class RatingEvaluation:
    """
    Evaluates rating systems on chess dataset using logloss.

    """

    def __init__(self, genplayerfunc, matchfunc, drawmatchfunc, winPcalcfunc, ):
        self.genplayerfunc = genplayerfunc
        self.matchfunc = matchfunc
        self.drawmatchfunc = drawmatchfunc
        self.winPcalcfunc = winPcalcfunc

        self.player_pool = {}
        self.idx = 0

        # preallocate resulting logloss
        self.loglosses = np.zeros((ratingevaluationdataset.shape[0], 1))

    def step(self, debug=False):
        """
        Step forward 1 match
        """
        player_a, player_b, score = ratingevaluationdataset.loc[self.idx, ['WhitePlayer', 'BlackPlayer', 'WhiteScore']]

        # init players if new
        if player_a not in self.player_pool.keys():
            self.player_pool[player_a] = self.genplayerfunc()
        if player_b not in self.player_pool.keys():
            self.player_pool[player_b] = self.genplayerfunc()

        # calculate predicted win/loss probability
        win_prob = self.winPcalcfunc(self.player_pool[player_a], self.player_pool[player_b])
        # clamp values to prevent inf/-inf log loss
        win_prob = max(win_prob, 1e-320)
        win_prob = min(win_prob, 1 - 1e-16)
        # clamp to ensure log-loss fairness
        win_prob = min(win_prob, 0.99)
        win_prob = max(win_prob, 0.01)

        logloss = - (score * np.log10(win_prob) + (1 - score) * np.log10(1 - win_prob))

        if np.isnan(logloss):
            raise ValueError

        self.loglosses[self.idx] = logloss

        if debug:
            print(
                f"{player_a} ({self.player_pool[player_a].mu}) vs {player_b} ({self.player_pool[player_b].mu}), score={score}, win_prob={win_prob}, loss={logloss}")

        # process win/lose/draw
        if score == 1:
            winner, loser = player_a, player_b
            self.player_pool[winner], self.player_pool[loser] = self.matchfunc(self.player_pool[winner], self.player_pool[loser])
        elif score == 0:
            winner, loser = player_b, player_a
            self.player_pool[winner], self.player_pool[loser] = self.matchfunc(self.player_pool[winner], self.player_pool[loser])
        elif score == 0.5:
            self.player_pool[player_a], self.player_pool[player_b] = self.drawmatchfunc(self.player_pool[player_a], self.player_pool[player_b])
        else:
            print(self.idx)
            raise ValueError

        self.idx += 1

    def run_through_dataset(self, length=1e9):
        length = min(length, ratingevaluationdataset.shape[0])
        for i in tqdm(range(length)):
            self.step()

    def output_results(self, moving_window=1000, length=1e9):
        length = min(length, len(self.loglosses[self.loglosses != 0]))

        plt.plot(np.round(moving_average(self.loglosses[self.loglosses != 0][:length], n=moving_window), decimals=7))
        print(np.mean(self.loglosses[self.loglosses != 0]))


def compare_results_rating_evaluation(list_of_obj_loss, moving_window=1000, legend=None, title_details=''):
    plt.figure()
    for obj in list_of_obj_loss:
        plt.plot(np.round(moving_average(obj.loglosses[obj.loglosses != 0], n=moving_window), decimals=7))
    plt.title('' + title_details)
    if legend:
        plt.legend(legend)

    if headless:
        plt.savefig('img/scenario2.png')



######################
# Scenario 3
######################






######################
# Test Framework
######################

def scenario1():

    elocons = []
    glickocons = []
    tscons = []
    glicko2cons = []
    for seed in range(42, 52):

        elocon = ConvergenceEvaluation(EloRating, Elo1v1, Elo1v1draw, EloPwin, player_count=500, seed=seed)
        elocon.simulate(steps=200)

        glickocon = ConvergenceEvaluation(glicko, glicko1v1, glicko1v1draw, glickoPwin, player_count=500, seed=seed)
        glickocon.simulate(steps=200)

        tscon = ConvergenceEvaluation(tsRating, ts_rate_1vs1, ts_rate_1vs1_draw, ts_Pwin, player_count=500, seed=seed)
        tscon.simulate(steps=200)

        glicko2con = ConvergenceEvaluation(glicko2.Player, glicko21v1, glicko21v1draw, glicko2Pwin, player_count=500, seed=seed)
        glicko2con.simulate(steps=200)

        elocons.append(elocon)
        glickocons.append(glickocon)
        tscons.append(tscon)
        glicko2cons.append(glicko2con)

    compare_results_convergence_multi([elocons, glickocons, tscons, glicko2cons], legend=['elo', 'glicko', 'trueskill', 'glicko2'],
                                      title_details='Ranking Error (lower is better) of rating systems')


def scenario2():

    elosim = RatingEvaluation(EloRating, Elo1v1, Elo1v1draw, EloPwin)
    elosim.run_through_dataset(100000)

    glickosim = RatingEvaluation(glicko, glicko1v1, glicko1v1draw, glickoPwin)
    glickosim.run_through_dataset(length=100000)

    tssim = RatingEvaluation(tsRating, ts_rate_1vs1, ts_rate_1vs1_draw, ts_Pwin)
    tssim.run_through_dataset(length=100000)

    glicko2sim = RatingEvaluation(glicko2.Player, glicko21v1, glicko21v1draw, glicko2Pwin)
    glicko2sim.run_through_dataset(length=100000)

    compare_results_rating_evaluation([elosim, glickosim, tssim, glicko2sim],
                                      legend=['elo', 'glicko', 'trueskill', 'glicko2'], moving_window=5000,
                                      title_details='Logloss (moving average=5000, lower is better) of rating systems')


headless = False

if __name__ == "__main__":
    # headless
    headless = True
    matplotlib.use('Agg')
    print()
    print('Running scenario 1...')
    print()
    scenario1()
    print()
    print('Running scenario 2...')
    print()
    scenario2()
    print()
    print('Complete')
    print()
