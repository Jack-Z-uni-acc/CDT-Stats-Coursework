library(glmnet)
library(ggplot2)
library(dplyr)
library(tidyr)
library(purrr)
library(tibble)

set.seed(2026)

#### 1.a Generate the data matrix
n <- 200
p <- 400
n_replicates <- 100
n_true <- 10
noise_sd <- 1

correlation_values <- c(0, 0.25, 0.50, 0.75, 0.90)
block_size <- 10

true_variables <- 1:n_true
true_beta <- c(2.0, 1.75, 1.5, 1.25, 1.0,
               -2.0, -1.75, -1.5, -1.25, -1.0)

# Function to generate X, the data matrix
generate_X <- function(n, p, rho, block_size = 10) {
  # Initialise X with zeros
  X <- matrix(0, nrow = n, ncol = p)
  # Divide into blocks for within block correlation strength rho
  n_blocks <- ceiling(p / block_size)
  # Buid X by generating standard normal distributed blocks Z and
  # applying the correlation matrix sigma for within block correlation
  for (block in seq_len(n_blocks)) {
    start <- (block - 1) * block_size + 1
    end <- min(block * block_size, p)
    block_p <- end - start + 1

    Sigma <- matrix(rho, nrow = block_p, ncol = block_p)
    diag(Sigma) <- 1

    Z <- matrix(rnorm(n * block_p), nrow = n, ncol = block_p)
    L <- chol(Sigma)
    X[, start:end] <- Z %*% L
  }
  # Normalise the generated data
  scale(X)
}

# Function to fit a lasso model
fit_one_simulation <- function(rho, replicate_id,
                               n = 200, p = 400,
                               true_variables = 1:10,
                               true_beta = rep(1, 10),
                               noise_sd = 1) {
  # Generate the data matrix
  X <- generate_X(n = n, p = p, rho = rho, block_size = block_size)

  #### 1.b Generate a continuous response vector
  beta <- numeric(p)
  beta[true_variables] <- true_beta
  y <- drop(X %*% beta + rnorm(n, mean = 0, sd = noise_sd))

  # Cross-validation selects lambda using training data.
  cv_fit <- cv.glmnet(
    x = X,
    y = y,
    alpha = 1,
    family = "gaussian",
    standardize = TRUE,
    nfolds = 10,
    type.measure = "mse"
  )

  # lambda.1se gives a more parsimonious model. lambda.min is retained for
  # comparison and sensitivity analysis.
  selected_1se <- which(as.vector(coef(cv_fit, s = "lambda.1se"))[-1] != 0)
  selected_min <- which(as.vector(coef(cv_fit, s = "lambda.min"))[-1] != 0)

  calculate_metrics <- function(selected) {
    tp <- sum(selected %in% true_variables)
    fp <- sum(!(selected %in% true_variables))
    fn <- sum(!(true_variables %in% selected))
    tn <- p - tp - fp - fn

    tibble(
      replicate = replicate_id,
      rho = rho,
      true_positives = tp,
      false_positives = fp,
      false_negatives = fn,
      true_negatives = tn,
      sensitivity = ifelse(length(true_variables) == 0, NA, tp / length(true_variables)),
      specificity = ifelse((p - length(true_variables)) == 0, NA,
                           tn / (p - length(true_variables))),
      precision = ifelse((tp + fp) == 0, 1, tp / (tp + fp)),
      selected_variables = length(selected),
      exact_support_recovery = as.integer(setequal(selected, true_variables)),
      selected = list(selected)
    )
  }

  metrics_1se <- calculate_metrics(selected_1se) %>%
    mutate(lambda_rule = "lambda.1se")
  metrics_min <- calculate_metrics(selected_min) %>%
    mutate(lambda_rule = "lambda.min")

  bind_rows(metrics_1se, metrics_min)
}

# 5. Run the simulation study
simulation_grid <- expand_grid(
  rho = correlation_values,
  replicate = seq_len(n_replicates)
)

results <- map2_dfr(
  simulation_grid$rho,
  simulation_grid$replicate,
  ~ fit_one_simulation(
    rho = .x,
    replicate_id = .y,
    n = n,
    p = p,
    true_variables = true_variables,
    true_beta = true_beta,
    noise_sd = noise_sd
  )
)

# 6. Summarise performance
summary_results <- results %>%
  group_by(rho, lambda_rule) %>%
  summarise(
    mean_sensitivity = mean(sensitivity),
    sd_sensitivity = sd(sensitivity),
    mean_specificity = mean(specificity),
    sd_specificity = sd(specificity),
    mean_precision = mean(precision),
    sd_precision = sd(precision),
    mean_selected_variables = mean(selected_variables),
    sd_selected_variables = sd(selected_variables),
    exact_recovery_rate = mean(exact_support_recovery),
    .groups = "drop"
  )

print(summary_results)

# 7. Visualise lasso performance
plot_data <- summary_results %>%
  select(rho, lambda_rule, mean_sensitivity, mean_specificity,
         mean_precision, mean_selected_variables) %>%
  pivot_longer(
    cols = starts_with("mean_"),
    names_to = "metric",
    values_to = "value"
  )

performance_plot <- ggplot(plot_data, aes(x = rho, y = value,
                                          colour = lambda_rule,
                                          group = lambda_rule)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2) +
  facet_wrap(~ metric, scales = "free_y") +
  scale_x_continuous(breaks = correlation_values) +
  labs(
    title = "Lasso performance as predictor correlation increases",
    x = "Within-block predictor correlation",
    y = "Mean value",
    colour = "Penalty choice"
  ) +
  theme_minimal(base_size = 12)

print(performance_plot)
ggsave("q4_lasso_performance.png", performance_plot,
       width = 10, height = 7, dpi = 300)