# Scenario 1

Scenario 1 consists of the classic 1v1 matchup. 

500 players are generated, having a rating from a normal distribution of mean=3000, sd=2000, and a standard deviation from a uniform distribution of 0 to 10. This is to mimic a typical human population that forms a normal distribution, and a variance in individual human skill.

When two players are matched-up against each other, 2 numbers are generated for each player, from a normal distribution of mean=rating, sd=standard deviation of the respective players. Hence, each player has a number range. If the number ranges of the two players overlap, the round is a draw. Otherwise, the player with the higher number range wins the round.

200 rounds are played. In each round, the pairings are determined by the following procedure:

```
1. Players are sorted according to their current matchmaking rating (e.g. [1400, 1450, 1490, 1520, ...])
2. A running window of 50 is performed across the array of players, where the indices of players are shuffled uniformly within the window at each step. (e.g. after first step, [1490, 1400, 1520, 1450, ...(first 50), 1900, 1910, 1930, ...] )
3. Players are matched against their adjacent index. (e.g. [1490 vs 1400, 1520 vs 1450, ...])
```

As each player only plays 1 game per round, all ratings are essentially simultaneously updated at the end of each round.

## Demonstration that Kendall-Tau converges correctly

In this case, we set players' standard deviation to 0 to remove the chance of a draw. Number of players are set to 50 so convergence is faster. Running window is set to 50, which means that matchmaking is completely random.

In this plot, the moving average of Kendall-Tau score is set to 1, and we only run 1 simulation for each rating system.

![scenario1kendalltauconvergence](img/scenario1kendalltauconvergence.png) - need all algos


## Demonstration of the effect of running window in matchmaking

In the design of the matchmaking, we want to match players of close skill to each other. However, overly-limiting the pairings to close skill levels result in poorer convergence for all algorithms. 

![scenario1poorconvergence](img/scenario1poorconvergence.png) - Varying running window from 1, 25, 50, use ELO










