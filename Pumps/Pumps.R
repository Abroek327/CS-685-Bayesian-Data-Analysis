
# ============================================================
# WINBUGS VOLUME 1 - EXAMPLE 2: PUMPS
# Bayesian Hierarchical Gamma-Poisson Model Using RStan
# ============================================================

# STEP 1: Load packages
# Install once if necessary:

#install.packages(c("rstan", "bayesplot", "ggplot2"))

library(rstan)
library(bayesplot)
library(ggplot2)

rstan_options(auto_write = TRUE)
options(mc.cores = min(4, parallel::detectCores()))

# STEP 2: Load original WinBUGS pumps data

# Observed failures
x <- c(5, 1, 5, 14, 3, 19, 1, 1, 4, 22)

# Operating times (thousands of hours)
t <- c(94.3, 15.7, 62.9, 126, 5.24,
       31.4, 1.05, 1.05, 2.1, 10.5)

N <- length(x)

stopifnot(N == 10, length(t) == N, all(t > 0))

stan_data <- list(
  N = N,
  x = as.integer(x),
  t = t
)

print(data.frame(
  pump = 1:N,
  failures = x,
  operating_time = t,
  observed_rate = x / t
))

# STEP 3: Define the Stan model

stan_code <- "
data {
  int<lower=1> N;
  array[N] int<lower=0> x;
  vector<lower=0>[N] t;
}

parameters {
  vector<lower=0>[N] theta;

  real<lower=0> alpha;
  real<lower=0> beta;
}

transformed parameters {
  vector<lower=0>[N] lambda;

  lambda = theta .* t;
}

model {
  // Original WinBUGS hyperpriors
  alpha ~ exponential(1);
  beta ~ gamma(0.1, 1);

  // Hierarchical Gamma model
  theta ~ gamma(alpha, beta);

  // Poisson observation model
  x ~ poisson(lambda);
}

generated quantities {
  real<lower=0> population_mean_rate;
  array[N] int<lower=0> x_rep;

  population_mean_rate = alpha / beta;

  for (i in 1:N) {
    x_rep[i] = poisson_rng(lambda[i]);
  }
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
    "alpha", "beta",
    "population_mean_rate", "theta"
  ),
  probs = c(0.025, 0.5, 0.975)
)

# STEP 6: MCMC diagnostic plots

posterior_draws <- as.array(fit)

print(
  mcmc_trace(
    posterior_draws,
    pars = c(
      "alpha", "beta",
      "theta[1]", "theta[6]"
    )
  )
)

print(
  mcmc_dens_overlay(
    posterior_draws,
    pars = c(
      "alpha", "beta",
      "theta[1]", "theta[6]"
    )
  )
)

# Check divergent transitions
sampler_params <- get_sampler_params(
  fit,
  inc_warmup = FALSE
)

print(
  sapply(
    sampler_params,
    function(z) sum(z[, "divergent__"])
  )
)

# STEP 7: Posterior credible intervals

print(
  mcmc_intervals(
    posterior_draws,
    pars = c(
      "alpha", "beta",
      "theta[1]", "theta[5]",
      "theta[6]", "theta[10]"
    ),
    prob = 0.95
  )
)

# STEP 8: Posterior estimates for individual pumps

samples <- rstan::extract(fit)

theta_hat <- colMeans(samples$theta)

pump_results <- data.frame(
  pump = 1:N,
  failures = x,
  operating_time = t,
  observed_rate = x / t,
  posterior_mean_rate = theta_hat,
  lower_95 = apply(
    samples$theta, 2, quantile, 0.025
  ),
  upper_95 = apply(
    samples$theta, 2, quantile, 0.975
  )
)

print(pump_results)

# Plot observed and posterior pump failure rates
pump_results$Pump <- factor(pump_results$pump)

print(
  ggplot(pump_results, aes(x = Pump)) +
    geom_point(
      aes(y = observed_rate, color = "Observed"),
      size = 2
    ) +
    geom_point(
      aes(y = posterior_mean_rate,
          color = "Posterior mean"),
      size = 2
    ) +
    geom_errorbar(
      aes(ymin = lower_95, ymax = upper_95),
      width = 0.15
    ) +
    scale_color_manual(
      values = c(
        "Observed" = "gray45",
        "Posterior mean" = "steelblue"
      )
    ) +
    labs(
      title = "Observed and Bayesian Pump Failure Rates",
      x = "Pump",
      y = "Failures per 1,000 hours",
      color = NULL
    ) +
    theme_minimal()
)

# Posterior predictive check
print(
  ppc_bars(
    y = x,
    yrep = samples$x_rep[1:200, ]
  )
)

# STEP 9: Save fitted model and posterior results

#saveRDS(
#Q  fit,
#  file = "pumps_rstan_fit.rds"
#)

#Awrite.csv(
#Q  pump_results,
#  file = "pumps_posterior_summary.csv",
#  row.names = FALSE
#)
