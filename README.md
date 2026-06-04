# How Early Leads Decide League of Legends Matches
By: Emma Wolfgram

<!-- CODE TO EMBED PLOTS -->
<!-- <iframe
  src="assets/univariate_gold.html"
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

| Column | Description |
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

### Data Cleaning

The raw dataset contains 12 rows per game - 10 player rows and 2 team summary row. For this analysis, player statistics are not needed so rows are filtered to only keep team rows. This reduces the dataset to 25,058 representing 10,641 unique games.

Then complete games and partial games are separted using the `datacompleteness` column. This is done since paritial rows are missing the early-game gold columns which are central to the main question. All EDA and modeling was performed on complete rows only, while both complete and partial rows were retained for the missingness anaylsis.

Serveral columns representing yes/no events (`firstblood`, `firsttower`, `firstdragon`,  `result`) were stored as floats which where then cast to booleans. Game length was also converted from seconds to minutes for readability.

| gameid                | league   | side   | result   |   gamelength |   goldat10 |   goldat15 |   golddiffat10 |   golddiffat15 | firstblood   | firsttower   | firstdragon   | datacompleteness   |
|:----------------------|:---------|:-------|:---------|-------------:|-----------:|-----------:|---------------:|---------------:|:-------------|:-------------|:--------------|:-------------------|
| ESPORTSTMNT01_2690210 | LCKC     | Blue   | False    |         28.6 |      16218 |      24806 |           1523 |            107 | True         | True         | False         | complete           |
| ESPORTSTMNT01_2690210 | LCKC     | Red    | True     |         28.6 |      14695 |      24699 |          -1523 |           -107 | False        | False        | True          | complete           |
| ESPORTSTMNT01_2690219 | LCKC     | Blue   | False    |         35.2 |      14939 |      23522 |          -1619 |          -1763 | False        | False        | False         | complete           |
| ESPORTSTMNT01_2690219 | LCKC     | Red    | True     |         35.2 |      16558 |      25285 |           1619 |           1763 | True         | True         | True          | complete           |
| ESPORTSTMNT01_2690227 | LCKC     | Blue   | True     |         32.9 |      15466 |      24795 |           -103 |           1191 | False        | True         | True          | complete           |

### Univariate Analysis

The first univariate variable examined was the distribution of gold difference at 15 minutes (`golddiffat15`). To make this distribution more interpretable, only the team ahead in gold is included for each game. Including both teams would produce a distribution perfectly symmetric about zero, since every positive gold difference has an equal and opposite negative counterpart from the other team. By filtering to only gold-ahead teams, the shape of the early gold leads in professional play becomes much clearer.

<iframe
  src="assets/univariate_gold.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This distribution shows the gold difference at 15 minutes for the team that was ahead in gold in each game (i.e. only positive gold differences are included). Smaller gold leads are much more common than large ones, with most games having a lead between 0 and 2,000 gold at 15 minutes and very few games seeing a lead beyond 5,000 gold. This suggests that while gold leads are nearly universal by 15 minutes, truly dominant early games are relatively rare in professional play.

### Bivariate Analysis

The bivariate analysis examines how gold differences at 15 minutes relates to two key outcomes: win rate and game length. The first plot bins gold difference into ranges and computers the win rate for each bucket, revealing how strongly early gold leads translate to victories. The second plot examines whether the size of a gold lead also affects how quickly the game ends, testing whether large early leads not only predict wins but actively accelerate them.

This bar chart shows win rate across gold difference buckets at 15 minutes. 
<iframe
  src="assets/bivariate_gold_winrate.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The relationship is striking, showing that win rate increases consistently as gold lead grows (from under 10% for teams down more than 3,000 gold to over 90% for teams ahead by more than 3,000 gold. Surprisingly, even a small lead of 500 to 1,500 gold pushed win rate well above 50%, suggesting that the laning phase is highly predictive of match outcomes.

This scatter plot shows game length against gold difference at 15 minutes for winning teams only. 
<iframe
  src="assets/bivariate_gold_gamelength.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The plot suggests that larger gold leads are associated with shorter, more decisive games. Winning teams with a gold lead greater than 3,000 at 15 minutes tend to close out games within 25-40 minutes, while winning teams with smaller gold leads face longer, more contested games that extend past 35 minutes. This suggests that large early gold leads don't just correlate with wins, they accelerate them, leaving the losing team with little time or opportunity for a comeback.

### Aggregates

| side |   goldat10 |   goldat15 |   golddiffat10 |   golddiffat15 |
|------|-----------:|-----------:|---------------:|---------------:|
| Blue |      15733 |      24952 |             88 |            250 |
| Red |      15645 |      24702 |            -88 |           -250 |

Since the analysis focuses on whether early game leads predict wins, it is worth examining the average gold statistics for red and blue sides because the blue side is widely considered to be the more advantageous side due to first pick priority. The aggregate data shows that the blue side hold a small but consistent gold advantage over the red side at both the 10 and 15 minute marks. While this advantage is modest relative to the gold differentials observed in the bivariate plot, it suggests the side selection should be accounted for as a factor when interpreting early gold leads.

## Assessment of Missingness

### NMAR Analysis

In this dataset, `playername` is most likely Not Missing At Random (NMAR). Looking at this column more closely, there are rows where the value is clearly missing because the `position` column corresponds to a 'team' row and in these cases, missingness makes sense since teams do not have player names. However, the more notable source of NMAR missingness is that a player name's absence is directly related to the player's identity, specifically whether they are a registered professional player or an unregistered stand-in. In competitive League games, teams occasionally have substitute players who are not officially registered with the league, meaning the league's data provider has no verified identity to record. If additional data on league registration records were available, showing which players were registered vs. unregistered substitutes for each match, a column `is_sub` could be constructed with a value of 1 for substitutes and 0 otherwise. The missingness of `playername` could then be fully explained through `is_sub`, making it Missing At Random (MAR).

### Missingness Dependency

In this section, the missingness of the `golddiffat15` column is tested for dependency on two other columns: `league` and `side`.

**Permutation test 1: `golddiffat15` and `league`.**

**Null Hypothesis:** The distribution of `league` when `golddiffat15` is missing is the same as the distribution of `league` when `golddiffat15` is not missing.

**Alternative Hypothesis:** The distribution of `league` when `golddiffat15` is missing is NOT the same as the distribution of `league` when `golddiffat15` is not missing.

**Test statistic:** Total Variation Distance (TVD)

**Significance Level:** 0.05

<iframe
  src="assets/missing_test1_prop.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The distribution of `league` proportions is visualized above for both groups, rows where `golddiffat15` is missing and rows where is is not.

<iframe
  src="assets/missing_test1_emp.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The observed TVD of 0.9926 is marked by the red vertical line. With a p-value of 0.0, which falls below the significance level of 0.05, we reject the null hypothesis. Therefore, the missingness of `golddiffat15` **is dependent** on `league`.

**Permutation Test 2: `golddiffat15` and `side`.**

**Null Hypothesis:** The distribution of `side` when `golddiffat15` is missing is the same as the distribution of `side` when `golddiffat15` is not missing.

**Alternative Hypothesis:** The distribution of `side` when `golddiffat15` is missing is NOT the same as the distribution of `side` when `golddiffat15` is not missing.

**Test statistic:** Difference in Proportions

**Significance Level:** 0.05

<iframe
  src="assets/missing_test2_prop.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The distribution of `side` proportions is visualed above for both groups, rows where `golddiffat15` is missing and row where it is not.

<iframe
  src="assets/missing_test2_emp.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The observed difference in proportions of 0.0 is marked by the red vertical line. With a p-value of 1.0, which exceeds the significance level of 0.05, we fail to reject the null. Therefore, the missingness of `golddiffat15` **is not dependent** on `side`.

## Hypothesis Testing

This hypothesis test assesses whether the observed difference in win rate between gold-ahead and gold-behind teams at the 15 minute mark is statistically significant. A gold-ahead team is defined as any team with a positive `golddiffat15` value, and a gold-behind team is defined as any team with a negative `golddiffat15` value. This investigation aims to determine whether games become "unlosable" for a team that holds a gold lead 15 minutes in.

**Null hypothesis:** A team's gold difference at 15 minutes has no effect on win rate. Any observed difference in win rates between gold-ahead and gold-behind teams is due to random chance.

**Alternative hypothesis:** Teams with a positive gold difference at 15 minutes win at a significantly higher rate than teams with a negative gold difference.

**Test Statistic:** Difference in win rates (win rate of gold-ahead teams - win rate of gold-behind teams)

**Significance Level:** 0.05

The results of this hypothesis test are:

| | Value |
|-|-------|
|Gold ahead win rate|  0.743|
|Gold behind win rate|  0.257|
|Observed difference|   0.485|
|P-value| 0.0000

<iframe
  src="assets/missing_hypo_emp.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

With a p-value of 0.0, which falls below the significance level of 0.05, we reject the null hypothesis. This suggests that holding a gold lead at the 15-minute mark is associated with a significantly higher win rate.

## Framing a Prediction Problem

The exploratory analysis and hypothesis testing done above confirmed that gold differences at 15 minutes is strongly associated with match outcomes. This raises the question: can we build a model that predicts, at the 15-minute mark, whether a team will win? This will be framed as a binary classification problem. The response variable is `result` (1 = win, 0 = loss) since it is the most direct measure of a team's success and ties directly to the question about early gold leads. The "time of prediction" is the 15-minute mark. This was chosen since there is the data for it, but also because it is the end of the laning phase, which is when the early-game advantages are fully established, but the game is far from over.

**Prediction problem: Given a team's early-game stats (first blood, gold at 10, first tower, etc.), predict whether they will eventually win (essentially testing how early the game is "decided").**

All features used to train the model are observable by this point:

| Feature      | Type | Description |
| ------------ | ---- | ----------- |
| golddiffat15 | Quantitative | Gold difference at 15 min |
| golddiffat10 | Quantitative | Gold difference at 10 min |
| firstblood   | Nominal      | Whether team got first blood |
| firsttower   | Nominal      | Whether team got first tower |
| firstdragon  | Nominal      | Whether team got first dragon |
| side         | Nominal      | Blue or Red side |

All post-game columns, such as kills, deaths, totalgold, and gamelength are excluded from this test since they are statistics only known after the game ends. The model will be evaluated using both F1-score and accuracy. Since the dataset is perfectly balanced (every game has one winner and one loser, giving a natural 50/50 class split), accuracy is a valid metric. F1-score will also be reported since it accounts for both precision and recall, giving a more complete picture of model performance. A random baseline of this model would achieve 50% accuracy, but any model worth using should exceed this.

## Baseline Model

The baseline model is a Logistic Regression classifier trained on four features:
`golddiffat15` (quantitative, scaled with `StandardScaler`), and `firstblood` ,`firsttower`, and `side` (all nominal, encoded with `OneHotEncoder`). 

The results for this baseline model are:

| Metric | Score |
| ------ | ----- |
| Accuracy | 0.7388 |
| F1-score | 0.7396 |

This baseline is a reasonable but imperfect model. At ~74% accuracy and F1-score, it performs well above the 50% random baseline, meaning early gold and objective features do carry genuine predictive signal. However, it is not a particularly "good" model in an absolute sense, roughly 1 in 4 predictions is wrong and logistic regression's linear decision boundary is likely too rigid to capture hte non-linear interactions between early-game features. It serves as a functional starting point, but there is clear room for improvement.

## Final Model

Two features were engineered on top of the baseline: `gold_momentum` (`golddiffat15 - golddiffat10`), which captures whether a gold lead was growing or shrinking between 10 and 15 minutes, and `first_obj_count` (sum of `firstblood`, `firsttower`, and `firstdragon`), a 0-3 scale of early map control. A team steadily extending its lead carries different implications than one that peaked early, and objective control reflects map pressure that raw gold difference alone does not capture.

The final model swaps Logistic Regression for a Random Forest Classifier, which can model non-linear interactions between features. All three quantitative features (`golddiffat15`, `gold_momentum`, `first_obj_count`) are scaled with `StandardScaler`, and the same nomial features from the baseline are encoded identically. 

Hyperparameters were tuned via `GridSearchCV` with 5-fold cross-validation over `max_depth`, `n_estimators` and `min_sample_leaf`. The best performing combination was `max_depth = 10`, `n_estimators = 200`, and `min_samples_leaf = 20`.

The results for the final model are:

| Metric | Baseline | Final |
| ------ | ----- | -------- |
| Accuracy | 0.7388| 0.7425 |
| F1-score | 0.7396 | 0.7440|

The final model improves on the baseline across both metrics. The modest gain reflects a genuine ceiling in the predictive power of 15-minute data. This leads to the conclusion that early-game advantages are strongly associated with wins, but League of Legends games are not decided in the first 15 minutes alone.

## Fairness Analysis

Determined which teams were Tier 1 and which were tier two from Riot Games' global power rankings. The LCK (League Champions Korea), LPL (League of Legends Pro League - China), LEC (League of Legends EMEA Championship - Europe), and LCS (League Championship Series - North America) leagues hold the highest rankings globally. Thus, we will consider these leagues Tier 1, and the rest of the leagues will be Tier 2.

**Null Hypothesis:** The model is fair. Its precision for Tier 1 leagues and Tier 2 leagues are roughly the same, any observed difference is due to random chance

**Alternative Hypothesis:** The model is unfair. Its precision differs slightly between Tier 1 and Tier 2 leagues

**Test Statistic:** Absolute difference in precision between Tier 1 and Tier 2 groups

**Significance Level:** 0.05

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