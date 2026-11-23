# Step 3: train the models.
#   * Main model: multinomial LASSO logistic regression (glmnet) on TF-IDF
#     features, with class weights, lambda chosen by 5-fold cross-validation.
#   * Baseline: multinomial Naive Bayes (naivebayes) on raw word counts.
# Both models are also scored on the same 5 cross-validation folds, which gives
# validation results separate from the final test set.
# Output: models/model.rds, outputs/validation_*.csv, outputs/cv_lambda.*

source("R/utils.R")
library(glmnet)
library(naivebayes)
message("== Step 3: train models")

f <- readRDS(file.path(PATHS$processed, "features.rds"))
x_train <- tfidf(f$train_counts, f$idf)

# Class weights: each class contributes equally to the loss.
class_freq <- table(f$y_train)
w <- as.numeric(sum(class_freq) / (length(class_freq) * class_freq[as.character(f$y_train)]))

# --- LASSO: 5-fold CV picks lambda (the penalty strength) -------------------
set.seed(CONFIG$seed)
t0 <- Sys.time()
lasso <- cv.glmnet(x_train, f$y_train, family = "multinomial", alpha = 1,
                   weights = w, nfolds = 5, type.measure = "deviance", keep = TRUE)
message(sprintf("glmnet trained in %.1f min; lambda.min = %.5f",
                as.numeric(difftime(Sys.time(), t0, units = "mins")), lasso$lambda.min))

# Out-of-fold predictions at the chosen lambda = validation predictions.
i_min <- which(lasso$lambda == lasso$lambda.min)
oof_probs <- lasso$fit.preval[, , i_min]
val_pred_lasso <- colnames(oof_probs)[max.col(oof_probs, ties.method = "first")]

# --- Naive Bayes: fit on all training data; validate on the same folds -----
nb <- multinomial_naive_bayes(x = f$train_counts, y = f$y_train, laplace = 1)
val_pred_nb <- character(length(f$y_train))
for (k in sort(unique(lasso$foldid))) {
  hold <- lasso$foldid == k
  nb_k <- multinomial_naive_bayes(x = f$train_counts[!hold, ], y = f$y_train[!hold], laplace = 1)
  val_pred_nb[hold] <- as.character(predict(nb_k, f$train_counts[hold, ], type = "class"))
}
message("Naive Bayes trained")

# --- save validation results ------------------------------------------------
val <- list(lasso = class_metrics(val_pred_lasso, f$y_train, "lasso"),
            naive_bayes = class_metrics(val_pred_nb, f$y_train, "naive_bayes"))
write_csv(map_dfr(val, "by_class"), file.path(PATHS$outputs, "validation_by_class.csv"))
write_csv(map_dfr(val, "overall"),  file.path(PATHS$outputs, "validation_overall.csv"))
print(map_dfr(val, "overall"))

cv_curve <- tibble(lambda = lasso$lambda, cv_deviance = lasso$cvm,
                   cv_se = lasso$cvsd,
                   chosen = lasso$lambda == lasso$lambda.min)
write_csv(cv_curve, file.path(PATHS$outputs, "cv_lambda.csv"))

cv_plot <- ggplot(cv_curve, aes(log10(lambda), cv_deviance)) +
  geom_ribbon(aes(ymin = cv_deviance - cv_se, ymax = cv_deviance + cv_se),
              fill = "#cde2fb") +
  geom_line(color = "#2a78d6", linewidth = 0.7) +
  geom_point(data = filter(cv_curve, chosen), color = "#0b0b0b", size = 3) +
  ggplot2::annotate("text", x = log10(lasso$lambda.min), y = min(cv_curve$cv_deviance),
           label = sprintf("chosen: lambda = %.5f\n(lowest validation error)", lasso$lambda.min),
           hjust = -0.1, vjust = -0.6, size = 3.6, color = "#0b0b0b") +
  labs(x = "log10(lambda)  (left = weaker penalty, more words kept)",
       y = "Validation error (deviance)",
       title = "Picking the penalty strength with 5-fold cross-validation") +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "#fcfcfb", color = NA))
ggsave(file.path(PATHS$outputs, "cv_lambda.png"), cv_plot, width = 8, height = 4, dpi = 150)

model <- list(
  lasso = lasso, lambda = "lambda.min", nb = nb,
  vocab = f$vocab, idf = f$idf, classes = CLASSES, config = CONFIG,
  trained_on = Sys.Date()
)
# Drop the stored out-of-fold predictions to keep the saved model small.
model$lasso$fit.preval <- NULL
saveRDS(model, file.path(PATHS$models, "model.rds"), compress = "xz")
