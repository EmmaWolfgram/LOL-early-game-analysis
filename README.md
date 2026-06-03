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

The orginal dataframe contains **ADD NUMBERS** rows and 165 columns. The columns that are central to the analysis are the following:

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

| gameid                | league   | side   | result   |   gamelength |   goldat10 |   goldat15 |   golddiffat10 |   golddiffat15 | firstblood   | firsttower   | firstdragon   | firstherald   | firstbaron   | datacompleteness   |
|:----------------------|:---------|:-------|:---------|-------------:|-----------:|-----------:|---------------:|---------------:|:-------------|:-------------|:--------------|:--------------|:-------------|:-------------------|
| ESPORTSTMNT01_2690210 | LCKC     | Blue   | False    |         28.6 |      16218 |      24806 |           1523 |            107 | True         | True         | False         | True          | False        | complete           |
| ESPORTSTMNT01_2690210 | LCKC     | Red    | True     |         28.6 |      14695 |      24699 |          -1523 |           -107 | False        | False        | True          | False         | False        | complete           |
| ESPORTSTMNT01_2690219 | LCKC     | Blue   | False    |         35.2 |      14939 |      23522 |          -1619 |          -1763 | False        | False        | False         | True          | False        | complete           |
| ESPORTSTMNT01_2690219 | LCKC     | Red    | True     |         35.2 |      16558 |      25285 |           1619 |           1763 | True         | True         | True          | False         | True         | complete           |
| 8401-8401_game_1      | LPL      | Blue   | True     |         22.8 |        nan |        nan |            nan |            nan | False        | True         | False         | <NA>          | <NA>         | partial            |

### Univariate Analysis

### Bivariate Analysis

### Aggregates

## Assessment of Missingness

### NMAR Analysis

### Missingness Dependency

## Hypothesis Testing

## Framing a Prediction Problem

Two columns were engineered for use in modeling: `gold_momentum` (`golddiffat15 - golddiffat10`), capturing whether a gold lead was growing or shrinking between 10 and 15 minutes, and `first_obj_count` (sum of `firstblood`, `firsttower`, and `firstdragon`), a 0-3 scale of early map dominance.

## Baseline Model

## Final Model

## Fairness Analysis