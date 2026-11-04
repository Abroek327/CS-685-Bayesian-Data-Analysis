
# ============================================================
# WINBUGS VOLUME 1: SURGICAL EXAMPLE
# HIERARCHICAL BETA-BINOMIAL MODEL USING RSTAN
# ============================================================

# ============================================================
# STEP 1: LOAD REQUIRED PACKAGES
# ============================================================

# Install once if necessary:
# install.packages(c("rstan", "bayesplot", "ggplot2"))

library(rstan)
library(bayesplot)
library(ggplot2)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# ============================================================
# STEP 2: ORIGINAL WINBUGS SURGICAL DATA
# ============================================================

hospital <- LETTERS[1:12]

n <- c(47, 148, 119, 810, 211, 196,
       148, 215, 207, 97, 256, 360)

y <- c(0, 18, 8, 46, 8, 13,
       9, 31, 14, 8, 29, 24)

N <- length(n)

stopifnot(
  length(y) == N,
  all(y >= 0),
  all(y <= n),
  all(n > 0)
)

surgical_data <- data.frame(
  hospital = hospital,
  operations = n,
  deaths = y,
  observed_rate = y / n
)

print(surgical_data)

stan_data <- list(
  N = N,
  n = as.integer(n),
  y = as.integer(y)
)

# ============================================================
# STEP 3: SPECIFY THE STAN HIERARCHICAL MODEL
# ============================================================

stan_code <- "
data {
  int<lower=1> N;
  array[N] int<lower=0> n;
  array[N] int<lower=0> y;
}

parameters {
  real<lower=0> alpha;
  real<lower=0> beta;
  vector<lower=0, upper=1>[N] p;
}

transformed parameters {
  real<lower=0, upper=1> mu;
  real<lower=0> kappa;

  mu = alpha / (alpha + beta);
  kappa = alpha + beta;
}

model {
  // Hyperpriors: Gamma(shape, rate)
  alpha ~ gamma(1, 0.1);
  beta  ~ gamma(1, 0.1);

  // Hierarchical prior
  p ~ beta(alpha, beta);

  // Binomial likelihood
  y ~ binomial(n, p);
}

generated quantities {
  array[N] int y_rep;
  real p_new;

  // Posterior predictive deaths
  for (i in 1:N) {
    y_rep[i] = binomial_rng(n[i], p[i]);
  }

  // Mortality probability for a new hospital
  p_new = beta_rng(alpha, beta);
}
"

# ============================================================
# STEP 4: FIT MODEL USING HMC / NUTS
# ============================================================

set.seed(12345)

fit <- rstan::stan(
  model_code = stan_code,
  data = stan_data,
  chains = 4,
  iter = 4000,
  warmup = 2000,
  seed = 12345,
  control = list(
    adapt_delta = 0.95,
    max_treedepth = 12
  ),
  refresh = 100
)

# ============================================================
# STEP 5: POSTERIOR SUMMARIES
# ============================================================

print(
  fit,
  pars = c("alpha", "beta", "mu", "kappa",
           "p_new", "p"),
  probs = c(0.025, 0.5, 0.975),
  digits = 3
)

post <- rstan::extract(fit)

# Hospital-specific mortality summaries
p_mean <- colMeans(post$p)

p_ci <- t(apply(
  post$p, 2, quantile,
  probs = c(0.025, 0.975)
))

results <- data.frame(
  hospital = hospital,
  n = n,
  deaths = y,
  observed_rate = y / n,
  posterior_mean = p_mean,
  lower_95 = p_ci[, 1],
  upper_95 = p_ci[, 2]
)

print(round(results[, -1], 4))

# Population-level posterior summaries
cat("\nPopulation mortality mean:\n")
print(quantile(
  post$mu,
  probs = c(0.025, 0.5, 0.975)
))

cat("\nConcentration parameter:\n")
print(quantile(
  post$kappa,
  probs = c(0.025, 0.5, 0.975)
))

cat("\nNew hospital mortality prediction:\n")
print(quantile(
  post$p_new,
  probs = c(0.025, 0.5, 0.975)
))

# ============================================================
# STEP 6: MCMC CONVERGENCE DIAGNOSTICS
# ============================================================

summary_fit <- summary(fit)$summary

print(
  summary_fit[
    c("alpha", "beta", "mu", "kappa"),
    c("mean", "sd", "n_eff", "Rhat")
  ]
)

# Trace plots
mcmc_trace(
  as.array(fit),
  pars = c("alpha", "beta", "mu", "kappa")
)

# Posterior density plots
mcmc_dens_overlay(
  as.array(fit),
  pars = c("alpha", "beta", "mu", "kappa")
)

# Posterior intervals for hospital probabilities
mcmc_intervals(
  as.array(fit),
  pars = paste0("p[", 1:N, "]"),
  prob = 0.80,
  prob_outer = 0.95
)

# Check divergent transitions
np <- rstan::get_sampler_params(
  fit, inc_warmup = FALSE
)

divergences <- sum(vapply(
  np,
  function(x) sum(x[, "divergent__"]),
  numeric(1)
))

cat("\nDivergent transitions:", divergences, "\n")

# ============================================================
# STEP 7: POSTERIOR MORTALITY INTERVALS
# ============================================================

ggplot(results, aes(x = hospital,
                    y = posterior_mean)) +
  geom_point(size = 3) +
  geom_errorbar(
    aes(ymin = lower_95, ymax = upper_95),
    width = 0.2
  ) +
  labs(
    title = "Posterior Mortality by Hospital",
    x = "Hospital",
    y = "Mortality probability"
  ) +
  theme_minimal(base_size = 13)

# ============================================================
# STEP 8A: OBSERVED VS HIERARCHICAL ESTIMATES
# ============================================================

ggplot(results, aes(x = hospital)) +
  geom_point(
    aes(y = observed_rate, color = "Observed"),
    size = 3
  ) +
  geom_point(
    aes(y = posterior_mean, color = "Hierarchical"),
    size = 3
  ) +
  geom_errorbar(
    aes(ymin = lower_95, ymax = upper_95),
    width = 0.15
  ) +
  scale_color_manual(
    values = c(
      Observed = "red",
      Hierarchical = "blue"
    )
  ) +
  labs(
    title = "Observed vs Hierarchical Mortality Rates",
    x = "Hospital",
    y = "Mortality probability",
    color = "Estimate"
  ) +
  theme_minimal(base_size = 13)

# ============================================================
# STEP 8B: POSTERIOR PREDICTIVE CHECK
# ============================================================

y_rep <- post$y_rep

# Predictive interval for number of deaths
y_rep_mean <- colMeans(y_rep)

y_rep_ci <- t(apply(
  y_rep, 2, quantile,
  probs = c(0.025, 0.975)
))

ppc_results <- data.frame(
  hospital = hospital,
  observed = y,
  predicted = y_rep_mean,
  lower_95 = y_rep_ci[, 1],
  upper_95 = y_rep_ci[, 2]
)

ggplot(ppc_results,
       aes(x = hospital, y = predicted)) +
  geom_errorbar(
    aes(ymin = lower_95, ymax = upper_95),
    width = 0.2
  ) +
  geom_point(size = 3, color = "blue") +
  geom_point(
    aes(y = observed),
    color = "red",
    shape = 17,
    size = 3
  ) +
  labs(
    title = "Posterior Predictive Check",
    subtitle = "Blue: predicted; Red: observed",
    x = "Hospital",
    y = "Number of deaths"
  ) +
  theme_minimal(base_size = 13)

# ============================================================
# STEP 8C: SHRINKAGE TOWARD POPULATION MEAN
# ============================================================

population_mean <- mean(post$mu)

ggplot(results,
       aes(x = observed_rate, y = posterior_mean)) +
  geom_point(size = 3) +
  geom_text(
    aes(label = hospital),
    vjust = -0.8
  ) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed"
  ) +
  geom_hline(
    yintercept = population_mean,
    linetype = "dotted",
    color = "blue"
  ) +
  labs(
    title = "Hierarchical Shrinkage",
    subtitle = "Dashed: no shrinkage; Dotted: population mean",
    x = "Observed mortality rate",
    y = "Posterior mortality probability"
  ) +
  theme_minimal(base_size = 13)

# ============================================================
# STEP 9: SAVE MODEL AND RESULTS
# ============================================================

saveRDS(
  fit,
  file = "surgical_beta_binomial_fit.rds"
)

write.csv(
  results,
  file = "surgical_posterior_results.csv",
  row.names = FALSE
)

saveRDS(
  list(
    model = stan_code,
    data = stan_data,
    fit = fit,
    results = results,
    ppc = ppc_results
  ),
  file = "surgical_complete_analysis.rds"
)

cat("\nAnalysis completed successfully.\n")
