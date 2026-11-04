
# ============================================================
# WINBUGS VOLUME 1 - EXAMPLE 1: RATS
# Bayesian Hierarchical Normal Model Using RStan
# ============================================================

# STEP 1: Load packages
# Install first if necessary:

#install.packages(c("rstan", "bayesplot", "ggplot2"))

library(rstan)
library(bayesplot)
library(ggplot2)

rstan_options(auto_write = TRUE)
options(mc.cores = min(4, parallel::detectCores()))

# STEP 2: Load original WinBUGS rats data

data_url <- paste0(
  "https://raw.githubusercontent.com/wiki/",
  "stan-dev/rstan/rats.txt"
)

Y <- as.matrix(read.table(data_url, header = TRUE))

x <- c(8, 15, 22, 29, 36)
xbar <- mean(x)

N <- nrow(Y)
T <- ncol(Y)

stopifnot(N == 30, T == 5)

stan_data <- list(
  N = N,
  T = T,
  Y = Y,
  x = x,
  xbar = xbar
)

print(dim(Y))
print(head(Y))

# STEP 3: Define the Stan model

stan_code <- "
data {
  int<lower=1> N;
  int<lower=1> T;
  matrix[N, T] Y;
  vector[T] x;
  real xbar;
}

parameters {
  vector[N] alpha;
  vector[N] beta;

  real alpha_c;
  real beta_c;

  real<lower=0> tau_c;
  real<lower=0> tau_alpha;
  real<lower=0> tau_beta;
}

transformed parameters {
  real<lower=0> sigma;
  real<lower=0> sigma_alpha;
  real<lower=0> sigma_beta;

  sigma = inv_sqrt(tau_c);
  sigma_alpha = inv_sqrt(tau_alpha);
  sigma_beta = inv_sqrt(tau_beta);
}

model {
  // Original WinBUGS hyperpriors
  alpha_c ~ normal(0, 1000);
  beta_c ~ normal(0, 1000);

  tau_c ~ gamma(0.001, 0.001);
  tau_alpha ~ gamma(0.001, 0.001);
  tau_beta ~ gamma(0.001, 0.001);

  // Hierarchical distributions
  alpha ~ normal(alpha_c, sigma_alpha);
  beta ~ normal(beta_c, sigma_beta);

  // Observation model
  for (i in 1:N) {
    for (j in 1:T) {
      Y[i, j] ~ normal(
        alpha[i] + beta[i] * (x[j] - xbar),
        sigma
      );
    }
  }
}

generated quantities {
  real alpha0;
  alpha0 = alpha_c - xbar * beta_c;
}
"

# STEP 4: Fit the model using NUTS

set.seed(123)

fit <- stan(
  model_code = stan_code,
  data = stan_data,
  chains = 4,
  iter = 4000,
  warmup = 2000,
  seed = 123,
  control = list(
    adapt_delta = 0.95,
    max_treedepth = 12
  )
)

# STEP 5: Posterior summaries

print(
  fit,
  pars = c(
    "alpha_c", "beta_c", "alpha0",
    "sigma", "sigma_alpha", "sigma_beta"
  ),
  probs = c(0.025, 0.5, 0.975)
)

# STEP 6: MCMC diagnostic plots

posterior_draws <- as.array(fit)

mcmc_trace(
  posterior_draws,
  pars = c("alpha_c", "beta_c", "sigma", "alpha0")
)

mcmc_dens_overlay(
  posterior_draws,
  pars = c("alpha_c", "beta_c", "sigma", "alpha0")
)

# STEP 7: Posterior credible intervals

mcmc_intervals(
  posterior_draws,
  pars = c(
    "alpha_c", "beta_c", "sigma",
    "alpha0"
  ),
  prob = 0.95
)

# STEP 8: Posterior estimates for individual rats

samples <- rstan::extract(fit)

alpha_hat <- colMeans(samples$alpha)
beta_hat <- colMeans(samples$beta)

# Plot observed growth curves
matplot(
  x,
  t(Y),
  type = "l",
  lty = 1,
  col = "gray70",
  xlab = "Age (days)",
  ylab = "Weight (grams)",
  main = "Observed and Fitted Rat Growth Curves"
)

# Add posterior mean growth trajectories
for (i in 1:N) {
  lines(
    x,
    alpha_hat[i] + beta_hat[i] * (x - xbar),
    col = "steelblue",
    lwd = 1.2
  )
}

# Add population-average trajectory
lines(
  x,
  mean(samples$alpha_c) +
    mean(samples$beta_c) * (x - xbar),
  col = "red",
  lwd = 3
)

legend(
  "topleft",
  legend = c("Observed", "Individual fitted", "Population mean"),
  col = c("gray70", "steelblue", "red"),
  lty = 1,
  lwd = c(1, 1.2, 3),
  bty = "n"
)

# STEP 9: Save fitted model

#saveRDS(fit, file = "rats_rstan_fit.rds")

#write.csv()
# Remove the final quotation mark above if copying
# from a renderer that adds one after the code block.
