/**
 * Hierarchical Simulation Model, menu-size variant (h_m01_size_sim)
 *
 * Generates synthetic choice data from the h_m01_size prior for prior
 * predictive analysis and parameter recovery (study plan §8.5(e), §8.6).
 *
 * The one addition over h_m01_sim is gamma_size, and with it the control that
 * makes the NULL-CALIBRATION check possible:
 *
 *   gamma_size_sd = 0  =>  gamma_size is FIXED at gamma_size_mean.
 *
 * Setting mean = 0, sd = 0 simulates choices at a single, size-invariant true
 * alpha. Refitting h_m01_size to that data must return gamma_size ~ 0. This is
 * the check that a slope detected in the real study is behavioral rather than
 * an artifact of choice-set geometry (§3.4, §8.5(e), §9 item 8) -- without it,
 * order statistics on the menu could manufacture a size slope on their own.
 *
 * Drawing gamma_size from a degenerate normal_rng(mean, 0) is not portable, so
 * the fixed case is branched explicitly.
 */
data {
  // --- Dimensions ---
  int<lower=1> J;                         // number of cells
  int<lower=2> K;                         // number of consequences
  int<lower=1> D;                         // embedding dimensions
  int<lower=2> R;                         // number of distinct alternatives
  int<lower=1> P;                         // number of predictors

  // --- Shared alternatives ---
  array[R] vector[D] w;                   // feature vectors

  // --- Study design ---
  array[J] int<lower=1> M_per_cell;       // observations per cell
  matrix[J, P] X;                         // design matrix

  // --- Indicator arrays (stacked) ---
  int<lower=1> M_total;                   // sum(M_per_cell)
  array[M_total] int<lower=1,upper=J> cell;
  array[M_total, R] int<lower=0,upper=1> I;

  // --- Menu-size covariate (RQ6) ---
  vector[M_total] s;                      // centered menu size per observation

  // --- Hyperparameter controls ---
  real gamma0_mean;                        // mean for gamma0 prior
  real<lower=0> gamma0_sd;                 // sd for gamma0 prior
  real<lower=0> gamma_sd;                  // sd for gamma coefficients
  real gamma_size_mean;                    // mean for gamma_size (0 for null calibration)
  real<lower=0> gamma_size_sd;             // sd for gamma_size; 0 fixes it at the mean
  real<lower=0> sigma_cell_sd;             // sd for half-normal on sigma_cell
  real<lower=0> beta_sd;                   // sd for beta coefficients
}

transformed data {
  array[M_total] int<lower=2> N_obs;
  int total_alts = 0;
  for (m in 1:M_total) {
    N_obs[m] = sum(I[m]);
    total_alts += N_obs[m];
  }

  {
    real mean_size = mean(to_vector(N_obs));
    if (abs(mean(s)) > 1e-6)
      reject("s must be centered (mean 0); got mean(s) = ", mean(s));
    for (m in 1:M_total) {
      if (abs(s[m] - (N_obs[m] - mean_size)) > 1e-6)
        reject("s[", m, "] = ", s[m], " but observation ", m, " has menu size ",
               N_obs[m], " and the mean menu size is ", mean_size);
    }
  }

  array[total_alts] vector[D] x_flat;
  {
    int pos = 1;
    for (m in 1:M_total) {
      for (r in 1:R) {
        if (I[m, r] == 1) {
          x_flat[pos] = w[r];
          pos += 1;
        }
      }
    }
  }
}

generated quantities {
  // Draw regression parameters
  real gamma0 = normal_rng(gamma0_mean, gamma0_sd);
  vector[P] gamma;
  for (p in 1:P) {
    gamma[p] = normal_rng(0, gamma_sd);
  }

  // Null calibration fixes the slope; recovery draws it from the prior.
  real gamma_size = (gamma_size_sd > 0)
                    ? normal_rng(gamma_size_mean, gamma_size_sd)
                    : gamma_size_mean;

  real<lower=0> sigma_cell = abs(normal_rng(0, sigma_cell_sd));

  // Cell-level alpha at the mean menu size (s is centered)
  vector[J] log_alpha_cell;
  vector[J] alpha_cell;
  for (j in 1:J) {
    real z_j = normal_rng(0, 1);
    log_alpha_cell[j] = gamma0 + X[j] * gamma + sigma_cell * z_j;
    alpha_cell[j] = exp(log_alpha_cell[j]);
  }

  // Draw per-cell betas
  array[J] matrix[K, D] beta;
  for (j in 1:J) {
    for (k in 1:K) {
      for (d in 1:D) {
        beta[j][k, d] = normal_rng(0, beta_sd);
      }
    }
  }

  // Draw shared utilities
  simplex[K-1] delta = dirichlet_rng(rep_vector(1.0, K-1));
  vector[K] upsilon = cumulative_sum(append_row(0, delta));

  // Observation-level alpha
  vector[M_total] alpha_obs;
  for (m in 1:M_total) {
    alpha_obs[m] = exp(log_alpha_cell[cell[m]] + gamma_size * s[m]);
  }

  // Generate choices
  array[M_total] int y;
  array[M_total] int<lower=0,upper=1> selected_seu_max;
  int<lower=0,upper=M_total> total_seu_max_selected;
  array[J] int seu_max_by_cell;
  // Deterministic-choice rate by menu size is the quantity the RQ6 slope is
  // read off, so it is reported directly rather than reconstructed later.
  array[M_total] int<lower=2> menu_size_out;

  {
    for (j in 1:J) {
      seu_max_by_cell[j] = 0;
    }

    int pos = 1;
    for (m in 1:M_total) {
      int j = cell[m];
      vector[N_obs[m]] problem_eta;
      for (idx in 1:N_obs[m]) {
        vector[K] psi_i = softmax(beta[j] * x_flat[pos]);
        problem_eta[idx] = dot_product(psi_i, upsilon);
        pos += 1;
      }
      y[m] = categorical_rng(softmax(alpha_obs[m] * problem_eta));
      menu_size_out[m] = N_obs[m];

      real max_eta = max(problem_eta);
      if (abs(problem_eta[y[m]] - max_eta) < 1e-10) {
        selected_seu_max[m] = 1;
        seu_max_by_cell[j] += 1;
      } else {
        selected_seu_max[m] = 0;
      }
    }

    total_seu_max_selected = sum(selected_seu_max);
  }
}
