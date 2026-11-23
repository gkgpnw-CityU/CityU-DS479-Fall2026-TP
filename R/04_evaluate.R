# Step 4: evaluate both models on the held-out test set.
# Outputs (all in outputs/): metrics_by_class.csv, metrics_overall.csv,
# confusion_<model>.csv, accuracy_by_source.csv, top_words.csv and plots.

source("R/utils.R")
library(caret)
library(glmnet)
library(naivebayes)
message("== Step 4: evaluate")

f     <- readRDS(file.path(PATHS$processed, "features.rds"))
model <- readRDS(file.path(PATHS$models, "model.rds"))
x_test <- tfidf(f$test_counts, f$idf)
truth  <- f$y_test

pred <- list(
  lasso = factor(predict(model$lasso, x_test, s = model$lambda, type = "class")[, 1],
                 levels = CLASSES),
  naive_bayes = factor(predict(model$nb, f$test_counts, type = "class"), levels = CLASSES)
)

# --- per-class precision / recall / F1 --------------------------------------
cms <- map(pred, ~ confusionMatrix(.x, truth, mode = "prec_recall"))

by_class <- imap_dfr(cms, function(cm, m) {
  as_tibble(cm$byClass, rownames = "class") |>
    transmute(model = m, class = str_remove(class, "^Class: "),
              precision = Precision, recall = Recall, f1 = F1,
              support = as.integer(colSums(cm$table)))
})

overall <- imap_dfr(cms, function(cm, m) {
  tibble(model = m,
         accuracy = cm$overall[["Accuracy"]],
         macro_f1 = mean(cm$byClass[, "F1"], na.rm = TRUE),
         malicious_recall = cm$byClass["Class: malicious", "Recall"])
})

confusion <- imap(cms, ~ as_tibble(.x$table) |>
                    rename(predicted = Prediction, actual = Reference, n = n) |>
                    mutate(model = .y))

# Accuracy per source dataset: a big gap between sources suggests the model is
# picking up on where an email came from rather than what it says.
by_source <- imap_dfr(pred, ~ tibble(model = .y, source = f$test_source,
                                     class = truth, correct = .x == truth)) |>
  group_by(model, source, class) |>
  summarise(n = n(), accuracy = mean(correct), .groups = "drop")

# --- most influential words per class (LASSO coefficients) ------------------
coefs <- coef(model$lasso, s = model$lambda)
top_words <- imap_dfr(coefs, function(b, cls) {
  tibble(class = cls, word = rownames(b), weight = as.numeric(b))
}) |>
  filter(word != "(Intercept)", weight > 0) |>
  group_by(class) |>
  slice_max(weight, n = 15, with_ties = FALSE) |>
  ungroup()

n_nonzero <- sum(rowSums(do.call(cbind, map(coefs, ~ abs(.x[-1, , drop = FALSE]))) > 0))

print(overall)
print(by_class)
message("Words used by the LASSO model: ", n_nonzero, " of ", length(model$vocab))

write_csv(by_class,  file.path(PATHS$outputs, "metrics_by_class.csv"))
write_csv(overall,   file.path(PATHS$outputs, "metrics_overall.csv"))
write_csv(bind_rows(confusion), file.path(PATHS$outputs, "confusion_matrices.csv"))
write_csv(by_source, file.path(PATHS$outputs, "accuracy_by_source.csv"))
write_csv(top_words, file.path(PATHS$outputs, "top_words.csv"))
write_lines(n_nonzero, file.path(PATHS$outputs, "lasso_nonzero_words.txt"))

# --- plots ------------------------------------------------------------------
theme_set(theme_minimal(base_size = 12) +
            theme(panel.grid.minor = element_blank(),
                  panel.grid.major = element_line(color = "#e6e5e0"),
                  axis.text = element_text(color = "#52514e"),
                  plot.background = element_rect(fill = "#fcfcfb", color = NA)))

# Confusion matrix: share of each actual class, so the small malicious class
# is readable next to the large ones.
cm_plot <- bind_rows(confusion) |>
  mutate(model = recode(model, lasso = "LASSO logistic regression",
                        naive_bayes = "Naive Bayes"),
         across(c(predicted, actual), ~ factor(.x, levels = CLASSES))) |>
  group_by(model, actual) |>
  mutate(share = n / sum(n)) |>
  ungroup() |>
  ggplot(aes(predicted, fct_rev(actual), fill = share)) +
  geom_tile(color = "#fcfcfb", linewidth = 1.5) +
  geom_text(aes(label = sprintf("%.1f%%\n(%s)", 100 * share, scales::comma(n)),
                color = share > 0.5), size = 3.4, lineheight = 0.9) +
  scale_fill_gradient(low = "#f0efec", high = "#1c5cab", limits = c(0, 1),
                      labels = scales::percent, name = "Share of\nactual class") +
  scale_color_manual(values = c(`TRUE` = "#ffffff", `FALSE` = "#0b0b0b"), guide = "none") +
  facet_wrap(~ model) +
  labs(x = "Predicted class", y = "Actual class",
       title = "Confusion matrices on the held-out test set") +
  theme(panel.grid = element_blank())
ggsave(file.path(PATHS$outputs, "confusion_matrices.png"), cm_plot,
       width = 9, height = 4, dpi = 150)

words_plot <- top_words |>
  mutate(class = factor(class, levels = CLASSES),
         word = reorder(paste(word, class, sep = "___"), weight)) |>
  ggplot(aes(weight, word, fill = class)) +
  geom_col(width = 0.7) +
  scale_y_discrete(labels = function(x) str_remove(x, "___.*$")) +
  scale_fill_manual(values = CLASS_COLORS, guide = "none") +
  facet_wrap(~ class, scales = "free_y") +
  labs(x = "LASSO coefficient (higher = stronger evidence for the class)", y = NULL,
       title = "Stemmed words that most push an email toward each class")
ggsave(file.path(PATHS$outputs, "top_words.png"), words_plot,
       width = 10, height = 5, dpi = 150)

message("Evaluation outputs written to ", PATHS$outputs, "/")
