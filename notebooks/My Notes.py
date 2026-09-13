# Databricks notebook source
I user ID - small and capital both letters are there

Time handling should be correctly checked and also for one customer consistency of data needs to be checked

Daily context should be analyzed heavily because it will be most resposible for coming night and morning checkins

There should be feedback also from user, whatever we predicted, Is it correct or not



# COMMAND ----------

Important findings already: 469 unique users (brief says 120!), 221 duplicate session_ids, 1,731 parse failures. Let me fix the timezone issue:

# COMMAND ----------

See duplicate session ids and duplicate users,

timezone is in timestamp format correct that 

Analyze sleep sessions data - multiple users session - choose the best one

see calculation session time vs reported is there any issues

Normalized data should be checked 

check if data can be extrapolated, using one data source to other source since some data sources have more users data

User can be shown that other timezones are winning, as a motivation and jealousy competition sort of thing 



# COMMAND ----------

What does this mean exactly

===========================================================
  VARIANCE DECOMPOSITION
============================================================
Total variance:   1.0381
Between-user var: 0.0702 (6.8%)
Within-user var:  0.9912 (95.5%)

→ Within-user variance dominates
→ Implication: User baselines less critical

# COMMAND ----------

Weights > 120 (possibly in lbs):
  Count: 12

  for this check the timezone of that user and what metrics are generally used there

# COMMAND ----------

Mood tags are informative but NOT prediction-time inputs: They're recorded at the same time as the target → can't use as features

Can be used for second level validation

