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

Central Question: Does a team's gold lead at the 15-minute mark predict whether they will win? Is there a gold threshold beyond which a lead becomes "unlosable"?

The 15-minute mark is significant because it represents the end of the laning phase (the early stage of the game where players farm gold and fight for early objectives). After 15 minutes, teams begin grouping together for large-scale fights and objectives. Understanding whether the laning phase effectively decides the game has real implications for how teams draft chamption, set strategic priorites, and allocate resources.

The orginal dataframe contains 150,348 rows and 165 columns. There are 12 rows per match - 10 player rows (5 for Blue side, 5 for Red) and 2 team rows detailing cumlative statistics. The columns that are central to the analysis are the following:

| Column | Description |
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

### Univariate Analysis

### Bivariate Analysis

### Aggregates

## Assessment of Missingness

### NMAR Analysis

### Missingness Dependency

## Hypothesis Testing

## Framing a Prediction Problem

## Baseline Model

## Final Model

## Fairness Analysis