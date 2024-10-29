library(stargazer)
library(tidyverse)

df <- read_csv("data/viz_learn_preds.csv")

first_mod <- lm(log(apt_rent) ~ b_age, data=df)
