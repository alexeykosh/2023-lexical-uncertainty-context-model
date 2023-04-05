library(tidyverse)
library(ggpubr)
library(DescTools)
library(scales)
library(ppcor)

theme_set(theme_bw())

# Initializing the model
n_iter <- 1000
alpha <- 5
cost <- c(0, 20)

system2("python3",
        args = c("src/model/model.py",
                 str_glue("--n_iter={n_iter}"),
                 str_glue("--alpha={alpha}"),
                 str_glue("--cost={paste(cost, collapse = ',')}")),
)

# Loading the data
# inser n_iter and alpha in the file name
data <- read_csv(str_glue("src/logs/logs-{n_iter}-{alpha}.csv"))
nrow(data)

# convert the correct column to 1 or 0
data$correct <- as.numeric(data$correct)

# Average accuracy per final lexicon
data <- data %>%
  group_by(trial) %>%
  # compute mean in the last 50 indices
  mutate(avg_acc = mean(correct[index > 50]),
  lexicon_last = last(lexicon_1)) %>%
  mutate(message_length = ifelse(message == 0, "short", "long"))

data %>%
  group_by(lexicon_last) %>%
  summarise(count = n() / nrow(.)) %>%
  arrange(desc(count)) %>%
  mutate(lexicon_last = factor(lexicon_last, levels = lexicon_last)) %>%
  ggplot(aes(x = lexicon_last, y = count)) +
  geom_col(fill = "blue", alpha=0.5) +
  labs(x = "Lexicon",
  y = "Percentage of total",
  title = "Lexicon distribution") +
  theme(plot.title = element_text(face = "bold", size = 12))

data %>%
  ggplot(aes(x = avg_acc)) +
  geom_density(aes(y = ..scaled..),
  fill = "blue",
  alpha = 0.5,
  color = alpha("grey", 0)) +
  facet_wrap(~lexicon_last, scales = "free_y") +
  geom_vline(xintercept = 2 / 3, colour = "red", linetype = "dashed") +
  xlim(0, 1) +
  # remove grey background from title but preserve the rest
  theme(plot.title = element_text(face = "bold", size = 12),
  strip.text = element_text(size = 12),
  strip.background = element_blank()) +
  labs(x = "Average accuracy",
  y = "Density",
  title = "Average accuracy in the last 50 trials")

data %>%
  group_by(trial) %>%
  mutate(index_long = index %% 10) %>%
  filter(avg_acc >= 2 / 3) %>%
  group_by(index_long, lexicon_last) %>%
  summarise(correct = mean(correct))  %>%
  ggplot(aes(x = index_long, y = correct, colour = lexicon_last)) +
  geom_path() +
  ylim(0, 1) +
  geom_hline(yintercept = 2 / 3, colour = "red", linetype = "dashed") +
  labs(x = "Trial number (grouped by 10)",
  y = "Average accuracy", colour = "Lexicon") +
  scale_colour_viridis_d(option = "D")

data %>%
  group_by(meaning_guess) %>%
  summarise(short = sum(message_length == "short") / n(),
  long = sum(message_length == "long") / n()) %>%
  gather(key = "message_length", value = "count", -meaning_guess) %>%
  ggplot(aes(x = meaning_guess, y = count, fill = message_length)) +
  geom_col(position = "dodge") +
  labs(x = "Meaning guess",
  y = "Proportion of messages",
  fill = "Message length") +
  scale_fill_viridis_d(option = "D")
