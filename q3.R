#   7. Produces diagnostic plots and a comparison table

library(faraway)
library(pls)

# 1. Load and inspect the data

data("meatspec", package = "faraway")

target <- names(meatspec)[ncol(meatspec)]
features <- names(meatspec)[-ncol(meatspec)]

# 2. Training/test split
set.seed(2026)

n <- nrow(meatspec)
train_size <- floor(0.70 * n)
train_index <- sample(seq_len(n), size = train_size, replace = FALSE)

data_train <- meatspec[train_index, , drop = FALSE]
data_test <- meatspec[-train_index, , drop = FALSE]

model_formula <- as.formula(
  paste(target, "~", paste(features, collapse = " + "))
)

max_components <- min(ncol(data_train) - 1, nrow(data_train) - 1)

# 3. Principal Components Regression (PCR)
set.seed(2026)
pcr_model <- pcr(
  formula = model_formula,
  data = data_train,
  ncomp = max_components,
  scale = TRUE,
  validation = "CV",
  segments = 10
)

pcr_rmsep <- drop(RMSEP(pcr_model, estimate = "CV")$val[1, 1, ])
pcr_components <- seq_along(pcr_rmsep) - 1

pcr_best_component <- pcr_components[-1][
  which.min(pcr_rmsep[-1])
]

cat("\nSelected PCR components:", pcr_best_component, "\n")

# 4. Partial Least Squares (PLS) regression
set.seed(2026)
pls_model <- plsr(
  formula = model_formula,
  data = data_train,
  ncomp = max_components,
  scale = TRUE,
  validation = "CV",
  segments = 10
)

pls_rmsep <- drop(RMSEP(pls_model, estimate = "CV")$val[1, 1, ])
pls_components <- seq_along(pls_rmsep) - 1

pls_best_component <- pls_components[-1][
  which.min(pls_rmsep[-1])
]

cat("Selected PLS components:", pls_best_component, "\n")

# 5. Predictions on the held-out test set
actual_test <- data_test[[target]]

pcr_pred <- as.vector(
  predict(pcr_model, newdata = data_test, ncomp = pcr_best_component)
)
pls_pred <- as.vector(
  predict(pls_model, newdata = data_test, ncomp = pls_best_component)
)

# 6. Performance measures
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

mae <- function(actual, predicted) {
  mean(abs(actual - predicted))
}

r_squared <- function(actual, predicted) {
  1 - sum((actual - predicted)^2) /
    sum((actual - mean(actual))^2)
}

performance <- data.frame(
  Method = c("PCR", "PLS"),
  Components = c(pcr_best_component, pls_best_component),
  Test_RMSE = c(
    rmse(actual_test, pcr_pred),
    rmse(actual_test, pls_pred)
  ),
  Test_MAE = c(
    mae(actual_test, pcr_pred),
    mae(actual_test, pls_pred)
  ),
  Test_R_squared = c(
    r_squared(actual_test, pcr_pred),
    r_squared(actual_test, pls_pred)
  )
)

cat("\nHeld-out test-set performance:\n")
print(performance, row.names = FALSE)

# 7. Cross-validation plot: PCR versus PLS
comparison_rmsep <- data.frame(
  Components = c(pcr_components, pls_components),
  CV_RMSEP = c(pcr_rmsep, pls_rmsep),
  Method = rep(c("PCR", "PLS"), each = length(pcr_components))
)

plot(
  pcr_components[-1], pcr_rmsep[-1],
  type = "o", pch = 16, col = "blue",
  xlab = "Number of components",
  ylab = "Cross-validated RMSEP",
  main = "Cross-validation comparison: PCR and PLS",
  ylim = range(comparison_rmsep$CV_RMSEP[-c(1, length(comparison_rmsep$CV_RMSEP))])
)
lines(
  pls_components[-1], pls_rmsep[-1],
  type = "o", pch = 17, col = "red"
)
abline(v = pcr_best_component, col = "blue", lty = 2)
abline(v = pls_best_component, col = "red", lty = 2)
legend(
  "topright",
  legend = c("PCR", "PLS"),
  col = c("blue", "red"),
  pch = c(16, 17),
  lty = 1,
  bty = "n"
)

# Observed vs predicted plot
plot(
  actual_test, pcr_pred,
  pch = 16, col = "blue",
  xlab = "Observed fat content",
  ylab = "Predicted fat content",
  main = "Held-out test-set predictions"
)
points(actual_test, pls_pred, pch = 17, col = "red")
abline(0, 1, lty = 2, lwd = 2)
legend(
  "topleft",
  legend = c("PCR", "PLS", "Ideal prediction"),
  col = c("blue", "red", "black"),
  pch = c(16, 17, NA),
  lty = c(NA, NA, 2),
  bty = "n"
)

# 9. Test-set residuals
pcr_residuals <- actual_test - pcr_pred
pls_residuals <- actual_test - pls_pred

par(mfrow = c(1, 2))
plot(
  pcr_pred, pcr_residuals,
  pch = 16, col = "blue",
  abline(h = 0, lty = 2),
  xlab = "PCR fitted values",
  ylab = "PCR residuals",
  main = "PCR residuals"
)
plot(
  pls_pred, pls_residuals,
  pch = 16, col = "red",
  abline(h = 0, lty = 2),
  xlab = "PLS fitted values",
  ylab = "PLS residuals",
  main = "PLS residuals"
)
par(mfrow = c(1, 1))