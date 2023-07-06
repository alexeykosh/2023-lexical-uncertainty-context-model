library(tidyverse)
library(ggpubr)
library(DescTools)
library(scales)
library(ppcor)
library(reshape2)
library(viridis)

theme_set(theme_bw())

# Initializing the model
n_iter <- 200
alpha <- 10
cost <- c(0, 20)
prior <- c(1, 1, 1)
model <- "rsa"

system2("python3",
        args = c("src/model/model.py",
                 str_glue("--n_iter={n_iter}"),
                 str_glue("--alpha={alpha}"),
                 str_glue("--cost={paste(cost, collapse = ',')}"),
                 str_glue("--prior={paste(prior, collapse = ',')}"), 
                 str_glue("--mode={model}"))
)

# Loading the data
# inser n_iter and alpha in the file name
data <- read_csv(str_glue("src/data/logs/logs-{n_iter}-{alpha}.csv"))
nrow(data)

# convert the correct column to 1 or 0
data$correct <- as.numeric(data$correct)
data$word_length <- as.numeric(data$word_length)

data$meaning_text <- ifelse(data$meaning == 0, "no",
                            "yes")

# Average accuracy per final lexicon
data <- data %>%
  group_by(trial) %>%
  # compute mean in the last 50 indices
  mutate(avg_acc = mean(correct[round >= 10]),
         lexicon_last = last(lexicon_a), word_length_ = word_length) %>%
  filter(avg_acc >= 1 / 3) %>%
  filter(mean(correct[round <= 10]) < mean(correct[round >= 10])) %>%
  mutate(word_length = ifelse(word_length_ > 0, "long", "short")) %>%
  mutate(lexicon_group = ifelse(lexicon_last == "011100", "efficient",
                              ifelse(lexicon_last == "100011", "non-efficient",
                                     "other"))) 

# data %>%
#   group_by(lexicon_last) %>%
#   summarise(count = n() / nrow(.)) %>%
#   arrange(desc(count)) %>%
#   mutate(lexicon_last = factor(lexicon_last, levels = lexicon_last)) %>%
#   ggplot(aes(x = lexicon_last, y = count)) +
#   geom_col(fill = "blue", alpha=0.5) +
#   labs(x = "Lexicon",
#   y = "Percentage of total",
#   title = "Lexicon distribution") +
#   theme(plot.title = element_text(face = "bold", size = 12)) 

# data %>%
#   ggplot(aes(x = avg_acc)) +
#   geom_density(aes(y = ..scaled..),
#   fill = "blue",
#   alpha = 0.5,
#   color = alpha("grey", 0)) +
#   facet_wrap(~lexicon_last, scales = "free_y") +
#   geom_vline(xintercept = 2 / 3, colour = "red", linetype = "dashed") +
#   xlim(0, 1) +
#   # remove grey background from title but preserve the rest
#   theme(plot.title = element_text(face = "bold", size = 12),
#   strip.text = element_text(size = 12),
#   strip.background = element_blank()) +
#   labs(x = "Average accuracy",
#   y = "Density",
#   title = "Average accuracy in the last 50 trials")

# # save avg_acc as pdf
# ggsave(filename = "figures/avg_acc.pdf",
#        plot = avg_acc,
#        width = 8,
#        height = 4)

p1 <- data %>%
  group_by(round, meaning_text) %>%
  summarise(short = sum(word_length == 'short') / n(),
  long = sum(word_length == 'long') / n()) %>%
  reshape2::melt(., id.vars=c("round", "meaning_text"))  %>%
  ggplot(aes(x=round, y=value, fill=variable))+
  geom_bar(stat="identity") +
  facet_wrap(~meaning_text, ncol=1) +
  theme(plot.title = element_text(face = "bold", size = 12),
        strip.text = element_text(size = 12),
        strip.background = element_blank())+
  scale_fill_brewer(palette = "Set2") +
  labs(x = 'Round',
       y = 'Share of word-lengths',
       fill = 'Word length')

p2 <- data %>%
  group_by(trial) %>%
  mutate(index_long = round, lexicon_last = last(lexicon_a)) %>%
  filter(lexicon_last %in%  c("011100", "100011")) %>%
  mutate(lexicon_group = ifelse(lexicon_last %in% c("011100"),
                                "efficient", "non-efficient")) %>%
  group_by(index_long, lexicon_group) %>%
  mutate(correct_mean = mean(correct),
         se = sd(correct) / sqrt(n()),
         lower = correct_mean - 1.96 * se,
         upper = correct_mean + 1.96 * se) %>%
  ggplot(aes(x = index_long, y = correct_mean, group = lexicon_group)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "grey70",
              alpha=0.5) +
  geom_hline(yintercept = 2 / 3, colour = "red", linetype = "dashed",
             linewidth = 1, alpha = 1) +
  geom_line(aes(color = lexicon_group), linewidth = 1) + 
  scale_color_brewer(palette = "Set1") +
  # add legend title (Lexicon group)
  guides(color = guide_legend(title = "Lexicon group")) +
  labs(x = "Round",
       y = "Average accuracy") +
  # add legend for the ribbon
  guides(fill = guide_legend(title = "95% confidence interval")) +
  theme(plot.title = element_text(face = "bold", size = 12),
        strip.text = element_text(size = 12),
        strip.background = element_blank(),
        legend.position = "None") +
  facet_wrap(~lexicon_group) + 
  geom_hline(yintercept = 1)

arr <- ggarrange(p2, p1,
          legend="bottom", 
          labels = c("A", "B"), 
          widths = c(7, 5))
arr

ggsave(plot = arr, 
       filename = "figures/share_length.pdf", 
       width = 10,
       height = 5)

# draft for the lexicon distribution
data %>%
  group_by(lexicon_last) %>%
  summarize(n = n()) %>%
  mutate(n = n /  sum(n)) %>%
  mutate(lexicon_group = ifelse(lexicon_last == "011100", "efficient",
  ifelse(lexicon_last == "100011", "non-efficient", "other"))) %>%
  group_by(lexicon_group) %>%
  summarize(n = sum(n)) %>%
  mutate(n = n / sum(n)) %>%
  mutate(lexicon_group = factor(lexicon_group, levels = c("efficient",
                                                          "non-efficient", "other"))) %>%
  ggplot(aes(x = lexicon_group, y = n, fill = lexicon_group)) +
  geom_col(alpha=1) +
  labs(x = "Lexicon",
       y = "Percentage of total",
       title = "Different costs (0, 1) + non-flat prior") +
  theme(plot.title = element_text(face = "bold", size = 12),
        legend.position = 'None') +
  scale_fill_brewer(palette = "Set1")

data %>%
  group_by(round) %>%
  mutate(avg_entropy_a = mean(entropy_a), avg_entropy_b = mean(entropy_b)) %>%
  ggplot() + 
  geom_line(aes(x=round, y=avg_entropy_a), linewidth = 2) +
  ylim(0, 3)
  # geom_point(aes(x=round, y=avg_entropy_b), color='blue')

data %>%
  group_by(trial) %>%
  mutate(index_long = round) %>%
  group_by(index_long) %>%
  mutate(correct_mean = mean(correct),
         se = sd(correct) / sqrt(n()),
         lower = correct_mean - 1.96 * se,
         upper = correct_mean + 1.96 * se) %>%
  ggplot(aes(x = index_long, y = correct_mean)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "grey70",
              alpha=0.5) +
  geom_hline(yintercept = 2 / 3, colour = "red", linetype = "dashed",
             linewidth = 1, alpha = 1) +
  geom_line(linewidth = 1) +
  ylim(0, 1.2)

data %>%
  group_by(round, meaning_text) %>%
  summarise(short = sum(word_length == 'short') / n(),
            long = sum(word_length == 'long') / n()) %>%
  reshape2::melt(., id.vars=c("round", "meaning_text"))  %>%
  ggplot(aes(x=round, y=value, fill=variable))+
  geom_bar(stat="identity") +
  facet_wrap(~meaning_text, ncol=1) +
  theme(plot.title = element_text(face = "bold", size = 12),
        strip.text = element_text(size = 12),
        strip.background = element_blank())+
  scale_fill_brewer(palette = "Set2") +
  labs(x = 'Round',
       y = 'Share of word-lengths',
       fill = 'Word length')

# library(brms)
# 
# data <- data %>%
#   mutate(player = ifelse(round %% 2 == 0, "Player 1", "Player 2"))
# 
# model_logit <- brm(
#   bf(word ~ meaning_text + (1 | trial / player)),
#   family = bernoulli(link = "logit"),
#   data = data[data$round >= 10,]
# )
# 
# summary(model_logit)
# 
# mcmc_plot(model_logit, 
#          type = "areas",
#          prob = 0.95)
# 
# model_logit

a <- data[data$round >= 10,] %>%
  group_by(word_length, meaning_text, trial, lexicon_group) %>%
  summarize(count = n()) %>%
  group_by(meaning_text, trial, lexicon_group) %>% 
  mutate(count = count / sum(count)) %>% 
  ggplot(aes(x=meaning_text, y=count, color=word_length)) +
  geom_boxplot() +
  ylim(0, 1) +
  facet_wrap(~lexicon_group, nrow = 3) +
  xlab('') +
  ylab('frequency')


b <- data[data$round >= 10,] %>%
  group_by(word_length, meaning_text, trial, lexicon_group) %>%
  summarize(count = n()) %>%
  group_by(meaning_text, trial, lexicon_group) %>% 
  mutate(count = count / sum(count)) %>% 
  ggplot(aes(x=meaning_text, y=count, color=word_length)) +
  geom_boxplot() +
  ylim(0, 1) +
  xlab('') +
  ylab('frequency')


ggarrange(a, b, common.legend = TRUE, labels = c("A", "B"))


data[data$round >= 10,] %>%
  distinct(trial, .keep_all=TRUE) %>%
  group_by(lexicon_group) %>%
  summarize(count = n()) %>%
  mutate(count / sum(count))

data[data$round >= 10,] %>% 
  group_by(meaning, trial) %>%
  summarize(accuracy = mean(correct), count = n()) %>%
  group_by(trial) %>%
  mutate(count = count / sum(count)) %>%
  ggplot(aes(x=count, y=accuracy, )) +
  ylim(0, 1) +
  geom_point()

data[data$round >= 10,]  %>% 
  group_by(lexicon_group, round) %>% 
  summarize(accuracy = mean(correct)) %>% 
  ggplot(aes(x=round, y=accuracy)) +
  geom_line(linewidth=1) +
  ylim(0, 1) +
  facet_wrap(~lexicon_group)

data %>% 
  group_by(trial, word_length, meaning_text, lexicon_group) %>% 
  summarize(count = n()) %>% 
  group_by(trial, word_length)  %>%  
  mutate(count = count / sum(count))  %>%
  unite(word_context, c("word_length", "meaning_text")) %>%
  pivot_wider(names_from = word_context,
              values_from = count) %>%
  replace(is.na(.), 0)  %>%
  ggplot(aes(x=long_no, y=short_yes, color = lexicon_group))+
  # geom_hex(bins = 10, aes(fill=after_stat(log(count)))) +
  geom_point() +
  xlim(0, 1) +
  ylim(0, 1) +
  geom_hline(yintercept = 0.5, color = 'red', alpha = 0.2) +
  geom_vline(xintercept = 0.5, color = 'red', alpha = 0.2) +
  facet_wrap(~lexicon_group) +
  theme(aspect.ratio = 1, legend.position = 'none') +
  labs(x = 'Long word & non-informative context',
       y = 'Short word & informative context')

for (x in seq(10, 100, 10)) {
  p <- data[data$round <= x,] %>% 
    group_by(trial, word_length, meaning_text, lexicon_group) %>% 
    summarize(count = n()) %>% 
    group_by(trial, word_length)  %>%  
    mutate(count = count / sum(count))  %>%
    unite(word_context, c("word_length", "meaning_text")) %>%
    pivot_wider(names_from = word_context,
                values_from = count) %>%
    replace(is.na(.), 0) %>% 
    mutate(efficient_square = ifelse(long_no >= 0.5 & short_yes >= 0.5,
                                     'efficent', 'non-efficient')) %>% 
    ggplot(aes(x = long_no, y = short_yes, color=lexicon_group)) +
    geom_count(size = 5, alpha = 0.5) +
    labs(x = 'Long word & non-informative context',
         y = 'Short word & informative context',
         color = 'Lexicon type') +
    geom_hline(yintercept = 0.5, color = 'red', alpha = 0.2) +
    geom_vline(xintercept = 0.5, color = 'red', alpha = 0.2) +
    xlim(0, 1) +
    ylim(0, 1) +
    annotate("text", label = "non-efficient", 
             x = 0.07, y = 0, size = 8, colour = "black") +
    annotate("text", label = "efficient", 
             x = 1 - 0.07, y = 1, size = 8, colour = "black") +
    annotate("text", label = "only short", 
             x = 0.07, y = 1, size = 8, colour = "black") +
    annotate("text", label = "only long", 
             x = 1 - 0.07, y = 0, size = 8, colour = "black") +
    theme(text = element_text(size = 20),
          legend.position = 'none') +
    scale_color_viridis(discrete = TRUE) +
    labs(title = sprintf("round: %s" , x))
  
  ggsave(plot = p, 
         filename = sprintf("figures/gif/share_length_%s.png" , x), 
         width = 10,
         height = 10)
}

