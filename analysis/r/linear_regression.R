## First we import
library(tidyverse)
library(arrow)
df <- (read_feather("data/suumo23日.ar"))

df <- head(df,1000 )

library(ggplot2)

ggplot(as.data.frame(df), aes(log(apt_rent), b_no_floors)) + geom_point()

df %>% group
