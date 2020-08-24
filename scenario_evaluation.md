# Scenario 1

Scenario 1 consists of the classic 1v1 matchup. 

500 players are generated, having a rating from a normal distribution of mean=3000, sd=2000, and a standard deviation from a uniform distribution of 0 to 10. This is to mimic a typical human population that forms a normal distribution, and a variance in individual human skill.

When two players are matched-up against each other, 2 numbers are generated for each player, from a normal distribution of mean=rating, sd=standard deviation of the respective players. Hence, each player has a number range. If the number ranges of the two players overlap, the round is a draw. Otherwise, the player with the higher number range wins the round.

200 rounds are played. In each round, the pairings are determined by the following procedure:

```
1. Players are sorted according to their current matchmaking rating (e.g. [1400, 1450, 1490, 1520, ...])
2. 50 passes of swapping are done. In each pass across the array, we swap adjacent indices with probability 50%. For each alternate pass, we swap the direction of traversal through the array.
3. Players are matched against their adjacent index. (e.g. [1490 vs 1400, 1520 vs 1450, ...])
```

As each player only plays 1 game per round, all ratings are essentially simultaneously updated at the end of each round.

## Demonstration that Kendall-Tau converges correctly

In this case, we set players' standard deviation to 0 to remove the chance of a draw. Number of players are set to 40 so convergence is faster. Running window is set to 40, which means that matchmaking is completely random.

In this plot, the moving average of Kendall-Tau score is set to 1, and we only run 1 simulation for each rating system.

![scenario1kendalltauconvergence](img/scenario1kendalltauconvergence.png)


## Investigating the effect of running window in matchmaking

In the design of the matchmaking, we want to match players of close skill to each other. This is to test effect of matchmaking step 2 running window size. Kendall-Tau running average is set to 1. For window size=1, this reduces to no shuffling, players are matched against the closest skill.

![scenario1poorconvergence](img/scenario1runningwindowconvergence.png)


## Investigating the effect of reducing Glicko/Glicko2 stdev loss

In the use of Glicko/Glicko2, the ratings are recommended to be evaluated for multiple matches at a time (5-10). However, in scenario 1, we evaluate single matches at a time. One consequence is that the stdev is reduced much faster than would normally be expected to.

In this run, we half the stdev loss for each match by taking the mean of the original and adjusted stdev.

![scenario1_reducedglickosd](img/scenario1_reducedglickosd.png)

We see significant improvement to Glicko2, but similar performance for Glicko.

## Empirical results of shuffling algorithm

We run the shuffling algorithm for 100 players, with 50 swap passes, for 10000 trials. The resulting histogram for indices 1, 30, 50, 70, 100 are plotted, which shows a close-to-gaussian distribution for each of the 5 indices.

![swappingalgorithm](img/swappingalgorithm.png)


