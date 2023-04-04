library(tidyverse)
library(ggpubr)
library(DescTools)
library(scales)
library(ppcor)

theme_set(theme_bw())

save <- TRUE # whether to save figures or not

# Initializing the model
n_iter <- 100
alpha <- 1
cost <- c(0, 10)

# Running the model (this should be uncommented if the reader wants 
# to run the model by himself)
# NB: the N/S ratio is scalled, but we should prob. scale 
# it down according to the size of the corpus
system2("python3",
        args = c("src/model/model.py",
                 str_glue("--n_iter={n_iter}"),
                 str_glue("--alpha={alpha}"),
                 str_glue("--cost={paste(cost, collapse = ',')}")),
)

# Loading the data
data <- read_csv("src/logs/logs-100-1.0.csv")