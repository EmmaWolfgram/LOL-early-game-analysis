# How Early Leads Decide League of Legends Matches
By: Emma Wolfgram

<!-- CODE TO EMBED PLOTS -->
<!-- <iframe
  src="assets/univarite_gold.html"
  width="800"
  height="600"
  frameborder="0"
></iframe> -->
## Overview
This data science project, conducted at UCSD, explores whether early-game gold leads in competitive League of Legends predict match outcomes and a machine learning model will be used to predict wins using only information available at the 15-minute mark.

## Introduction
League of Legends (LOL) is one of the most popular competitive games in the world, with professional leagues spanning across the globe. Each match pits two teams of five against each other where the goal is to destory the enemy team's Nexus. Games are won through a combination of mechanical skill, strategic decision making, and resource accumulation. The most important resource is gold (which this project will examine), which is used to purchase items that make players stronger.

This project analyzes a dataset of 12,529 competitive matches from Oracle's Elixir's 2022 League of Legends professional match dataset. This dataset captures several gameplay statistics offering a detailed source of player behavior, team summaries, and match outcomes.

**Central Question: Does a team's gold lead at the 15-minute mark predict whether they will win? Is there a gold threshold beyond which a lead becomes "unlosable"?**

The 15-minute mark is significant because it represents the end of the laning phase (the early stage of the game where players farm gold and fight for early objectives). After 15 minutes, teams begin grouping together for large-scale fights and objectives. Understanding whether the laning phase effectively decides the game has real implications for how teams draft chamption, set strategic priorites, and allocate resources.

The orginal dataframe contains 150,348 rows and 165 columns. The columns that are central to the analysis are the following:

| **Column** | **Description** |
| ---------- | --------------- |
| `golddiffat15` | Gold difference between a team and their opponent at 15 minutes (positive difference means ahead in gold, negative means behind) |
| `golddiffat10` | Gold difference at 10 minutes
| `result` | Match outcome (1 for win, 0 for loss)
| `firstblood` | Whether the team secured the first kill of the game
| `firsttower` | Whether the team secured the first tower of the game
| `firstdragon` | Whether the team secured the first dragon of the game
| `side` | Whether the team played on Blue or Red side
| `league` | The competitive league the match was played in
| `datacompleteness` | Whether the row has complete or partial data

## Data Cleaning and Exploratory Data Analysis

## Data Cleaning

The raw dataset contains 12 rows per game - 10 player rows and 2 team summary row. For this analysis, player statistics are not needed so rows are filtered to only keep team rows. This reduces the dataset to 25,058 representing 10,641 unique games.

Then complete games and partial games are separted using the `datacompleteness` column. This is done since paritial rows are missing the early-game gold columns which are central to the main question. All EDA and modeling was performed on complete rows only, while both complete and partial rows were retained for the missingness anaylsis.

Serveral columns representing yes/no events (`firstblood`, `firsttower`, `firstdragon`,  `result`) were stored as floats which where then cast to booleans. Game lenght was also converted from seconds to minutes for readability.

| gameid                | league   | side   | result   |   gamelength |   goldat10 |   goldat15 |   golddiffat10 |   golddiffat15 | firstblood   | firsttower   | firstdragon   | datacompleteness   |
|:----------------------|:---------|:-------|:---------|-------------:|-----------:|-----------:|---------------:|---------------:|:-------------|:-------------|:--------------|:-------------------|
| ESPORTSTMNT01_2690210 | LCKC     | Blue   | False    |         28.6 |      16218 |      24806 |           1523 |            107 | True         | True         | False         | complete           |
| ESPORTSTMNT01_2690210 | LCKC     | Red    | True     |         28.6 |      14695 |      24699 |          -1523 |           -107 | False        | False        | True          | complete           |
| ESPORTSTMNT01_2690219 | LCKC     | Blue   | False    |         35.2 |      14939 |      23522 |          -1619 |          -1763 | False        | False        | False         | complete           |
| ESPORTSTMNT01_2690219 | LCKC     | Red    | True     |         35.2 |      16558 |      25285 |           1619 |           1763 | True         | True         | True          | complete           |
| ESPORTSTMNT01_2690227 | LCKC     | Blue   | True     |         32.9 |      15466 |      24795 |           -103 |           1191 | False        | True         | True          | complete           |

### Univariate Analysis

This distribution shows the gold difference at 15 minutes for the team that was ahead in gold in each game (i.e. only positive gold differences are included). Smaller gold leads are much more common than large ones, with most games having a lead between 0 and 2,000 gold at 15 minutes and very few games seeing a lead beyond 5,000 gold. This suggests that while gold leads are nearly universal by 15 minutes, truly dominant early games are relatively rare in professional play.

### Bivariate Analysis

This bar chart shows win rate across gold difference buckets at 15 minutes. The relationship is striking, showing that win rate increases consistently as gold lead grows (from under 10% for teams down more than 3,000 gold to over 90% for teams ahead by more than 3,000 gold. Surprisingly, even a small lead of 500 to 1,500 gold pushed win rate well above 50%, suggesting that the laning phase is highly predictive of match outcomes.

### Aggregates

|   goldat10 |   goldat15 |   golddiffat10 |   golddiffat15 |
|-----------:|-----------:|---------------:|---------------:|
|      15733 |      24952 |             88 |            250 |
|      15645 |      24702 |            -88 |           -250 |

NOTE: A 250 gold advantage for Blue side at 15 minutes is real but small. The bivariate plot showed that teams need roughly +500 to +1500 gold before win rate meaningfully climbs above 50%. So Blue side's inherent advantage alone isn't enough to decide games, but it's a subtle thing to notice. This also points at maybe including side as a feature in the final model alongside golddiffat15.

Since I am testing whether early game leads predict wins, I thought it would be interesting to look at the average gold stats for the red and blue sides, since the blue side is often seen as the more advantageous side, since they get first pick. From the aggregate data above, we can see that the blue side holds a small but consistent gold advantage over the red side at both the 10 and 15 minute marks. While this advantage is modest relative to the gold differentials from the bivarate plot, it led me to make the decision to account for side as a factor when interpreting early gold leads. 

## Assessment of Missingness

### NMAR Analysis

REWRTIE SO NOT IN FIRST PERSON:

In this dataset, I believe that `playername` is Not Missing At Random (NMAR). Looking into this column more, there are rows where this value is clearly missing because the `position` column of that row corresponds to a 'team' row. In the 'team' rows, it makes sense that this data is missing, since teams do not have player names, just team names. This missingness of `playername` is likely NMAR because the reason a player's name is absent is directly related to who that player is, specifically, whether they are a registered professional player or an unregistered stand-in. In competitive League games, teams occasionally have substitute players who are not officially registered with the league, meaning the league's data provider has no verified identity to record. If there was additional data on league registration records showing which players were registered vs. unregistered substitutes for each match, you could have a column `is_sub` with a value of 1 if they were a sub and a value of 0 if they weren't. You could then explain the missingness of `playername` entirely through the column `is_sub`, making `playername` Missing At Random (MAR).

### Missingness Dependency

Going to test if the missingness of the `golddiffat15` column depends on other columns. The two columns I am going to test this on is `league` and `side`.

First, perform a permutation test on `golddiffat15` and `league`.

Null Hypo: Distribution of `league` when `golddiffat15` is missing is the same as the distribution of `league` when `golddiffat15` is not missing.

Alt Hypo: Distribution of `league` when `golddiffat15` is missing is NOT the same as the distribution of `league` when `golddiffat15` is not missing.

Test stat: TVD (total variation distance)

Significance Level: 0.05

(test 1 prop plot)
WRITE

(test 1 emp plot)
WRITE MORE: The observed statistic of 0.9926 is shown on the graph by a red vertical line. Since the p-value that I found (0.0) is < 0.05, we reject the null hypothesis. Thus, the missingness of golddiffat15 depends on the league the team is a part of.

Second, perform a permutation test on `golddiffat15` and `side`.

Null Hypo: Distribution of `side` when `golddiffat15` is missing is the same as the distribution of `side` when `golddiffat15` is not missing.

Alt Hypo: Distribution of `side` when `golddiffat15` is missing is NOT the same as the distribution of `side` when `golddiffat15` is not missing.

Test stat: difference in proportions

Significance Level: 0.05

(test 2 prop plot)
WRITE

(test 2 emp plot)
WRITE MORE: The observed statistic of 0 is shown on the graph by a red vertical line. Since the p-value that I found (1.00) is > 0.05, we fail to reject the null. Thus, the missingness of golddiffat15 does not depend on the side the team plays for.

## Hypothesis Testing

REWRITE SO NOT IN FIRST PERSON
In the hypothesis test, I aimed to assess whether there is an observed difference in win rate between gold-ahead and gold-behind teams at the 15 minute mark is statistically significant. A gold-ahead team is any team with a positive `golddiffat15` value and a gold-behind team is any any team with a negative `golddiffat15` value. This investigation allows us to determine if games are "unlosable" for a team if they are ahead in gold 15 minutes into the game.

Null hypothesis: A team's gold difference at 15 minutes has no effect on win rate. Any observed difference in win rates between gold-ahead and gold-behind teams is due to random chance.

Alternative hypothesis: Teams with a positive gold difference at 15 minutes win at a significantly higher rate than teams with a negative gold difference.

Test Statistic: Difference in win rates -> win rate of gold-ahead teams - win rate of gold-behind teams

Significance Level: 0.05

Gold ahead win rate:   0.743
Gold behind win rate:  0.257
Observed difference:   0.485
P-value: 0.0000

(hypo emp plot)
CONCLUSION OF PERM TEST

Since the p-value that we found (0.0) is less than the significance level of 0.05, we reject the null hypothesis. This suggests that holding a gold lead at the 15-minute mark is associated with a significantly higher win rate.

## Framing a Prediction Problem

Two columns were engineered for use in modeling: `gold_momentum` (`golddiffat15 - golddiffat10`), capturing whether a gold lead was growing or shrinking between 10 and 15 minutes, and `first_obj_count` (sum of `firstblood`, `firsttower`, and `firstdragon`), a 0-3 scale of early map dominance.

Prediction problem: Given a team's early-game stats (first blood, gold at 10, first tower), predict whether they will eventually win — essentially testing how early the game is "decided."

The exploratory analysis and hypothesis testing done above confirmed that gold differences at 15 minutes is strongly associated with match outcomes. This raises the question: can we build a model that predicts, at the 15-minute mark, whether a team will win? This will be framed as a binary classification problem. The response variable is `result` (1 = win, 0 = loss) since it is the most direct measure of a team's success and ties directly to the question about early gold leads. The "time of prediction" is the 15-minute mark. I chose this not only because I have data for it, but also because it is the end of the laning phase, which is when the early-game advantages are fully established, but the game is far from over.

All features used to train the model are observable by this point:

| Feature      | Type | Description |
| ------------ | ---- | ----------- |
| golddiffat15 | Quantitative | Gold difference at 15 min |
| golddiffat10 | Quantitative | Gold difference at 10 min |
| firstblood   | Nominal      | Whether team got first blood |
| firsttower   | Nominal      | Whether team got first tower |
| firstdragon  | Nominal      | Whether team got first dragon |
| side         | Nominal      | Blue or Red side |

All post-game columns, such as kills, deaths, totalgold, and gamelength are excluded from this test since they are stats only known after the game ends. The model will be evaluated using both F1-score and accuracy. Since the dataset is perfectly balanced (every game has one winner and one loser, giving a natural 50/50 class split), accuracy is a valid metric. F1-score will also be reported since it accounts for both precision and recall, giving a more complete picture of model performance. A random baseline of this model would achieve 50% accuracy, but any model worth using should exceed this.

## Baseline Model

WRITEEEE

Accuracy:  0.7388
F1-score:  0.7396

## Final Model

WRITEEEEE

Final Model Accuracy:  0.7425
Final Model F1-score:  0.7440

Baseline Accuracy:  0.7388  →  Final Accuracy:  0.7425
Baseline F1-score:  0.7396  →  Final F1-score:  0.7440

## Fairness Analysis

Determined which teams were Tier 1 and which were tier two from Riot Games' global power rankings. The LCK (League Champions Korea), LPL (League of Legends Pro League - China), LEC (League of Legends EMEA Championship - Europe), and LCS (League Championship Series - North America) leagues hold the highest rankings globally. Thus, we will consider these leagues Tier 1, and the rest of the leagues will be Tier 2.

Null: The model is fair. Its precision for Tier 1 leagues and Tier 2 leagues are roughly the same, any observed difference is due to random chance

Alt: The model is unfair. Its precision differs slightly between Tier 1 and Tier 2 leagues

Test Stat: Absolute difference in precision between Tier 1 and Tier 2 groups

Significance Level: 0.05

Tier 1 precision:  0.7015
Tier 2 precision:  0.7618
Observed diff:     0.0603
P-value: 0.0634

REWRTIE

This fairness analysis evaluates whether the final model performs equally well for teams in Tier 1 leagues (LCK, LPL, LEC, LCS) versus Tier 2 leagues (all others). This is a meaningful fairness question because the model was trained on data from all leagues combined and if Tier 1 games follow systematically different patterns (e.g. gold leads are more decisive, comebacks are rarer, or data is missing from one tier more than the other), the model may perform better for one group than the other.

Precision is used as the evaluation metric, measuring how often the model is correct when it predicts a win. The test statistic is the absolute difference in precision between the two groups.

| Tier 1 | Tier 2 |
| ------ | ------ |
| Precision | 0.7015 | 0.7618 |
| Games | 2032 | 19250 |

p-value: 0.0634 > 0.05

A permutation test with 10,000 iterations at α = 0.05 yields a p-value of 0.0634. Since 0.0634 > 0.05, we fail to reject the null hypothesis; there is not sufficient evidence to conclude that the model's precision differs significantly between Tier 1 and Tier 2 leagues.

Interestingly, the model achieves slightly higher precision on Tier 2 games (0.7618) than Tier 1 games (0.7015). This may reflect the fact that Tier 2 games are more lopsided. When a team builds a gold lead in a less competitive league, they are even more likely to convert it into a win, making predictions more reliable. However, given the p-value of 0.0634 sitting just above the significance threshold, this difference is not statistically significant at the chosen level.

Note that failing to reject the null hypothesis does not prove the model is perfectly fair; it only means there is insufficient evidence of a significant precision gap at α = 0.05. The p-value of 0.0634 is close enough to the threshold that a larger Tier 1 sample might tell a different story.