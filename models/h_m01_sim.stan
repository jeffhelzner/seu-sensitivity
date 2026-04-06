/**
 * Hierarchical Simulation Model (h_m01_sim)
 *
 * Generates synthetic choice data from the h_m01 prior for:
 * - Prior predictive analysis
 * - Parameter recovery studies
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

  // --- Hyperparameter controls ---
  real gamma0_mean;                        // mean for gamma0 prior
  real<lower=0> gamma0_sd;                 // sd for gamma0 prior
  real<lower=0> gamma_sd;                  // sd for gamma coefficients
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
  real<lower=0> sigma_cell = abs(normal_rng(0, sigma_cell_sd));

  // Draw cell-level alphas
  vector[J] log_alpha;
  vector[J] alpha;
  for (j in 1:J) {
    real z_j = normal_rng(0, 1);
    log_alpha[j] = gamma0 + X[j] * gamma + sigma_cell * z_j;
    alpha[j] = exp(log_alpha[j]);
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

  // Compute expected utilities and generate choices
  array[M_total] int y;
  {
    int pos = 1;
    for (m in 1:M_total) {
      int j = cell[m];
      vector[N_obs[m]] problem_eta;
      for (idx in 1:N_obs[m]) {
        vector[K] psi_i = softmax(beta[j] * x_flat[pos]);
        problem_eta[idx] = dot_product(psi_i, upsilon);
        pos += 1;
      }
      y[m] = categorical_rng(softmax(alpha[j] * problem_eta));
    }
  }
}
