/**
 * Hierarchical SBC Model (h_m01_sbc)
 *
 * Draws true parameters from the prior in transformed data,
 * generates choice data, then fits the model to the generated data.
 * Produces rank statistics for SBC validation.
 */
data {
  // --- Dimensions ---
  int<lower=1> J;
  int<lower=2> K;
  int<lower=1> D;
  int<lower=2> R;
  int<lower=1> P;

  // --- Shared alternatives ---
  array[R] vector[D] w;

  // --- Study design ---
  array[J] int<lower=1> M_per_cell;
  matrix[J, P] X;

  // --- Stacked indicators ---
  int<lower=1> M_total;
  array[M_total] int<lower=1,upper=J> cell;
  array[M_total, R] int<lower=0,upper=1> I;
}

transformed data {
  // Compute alternative counts
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

  // --- Draw true parameters ---
  real gamma0_ = normal_rng(3.0, 1.0);

  vector[P] gamma_;
  for (p in 1:P) {
    gamma_[p] = normal_rng(0, 1.0);
  }

  real<lower=0> sigma_cell_ = abs(normal_rng(0, 0.5));

  vector[J] z_alpha_;
  vector[J] log_alpha_;
  for (j in 1:J) {
    z_alpha_[j] = normal_rng(0, 1);
    log_alpha_[j] = gamma0_ + X[j] * gamma_ + sigma_cell_ * z_alpha_[j];
  }
  vector<lower=0>[J] alpha_ = exp(log_alpha_);

  array[J] matrix[K, D] beta_;
  for (j in 1:J) {
    for (k in 1:K) {
      for (d in 1:D) {
        beta_[j][k, d] = normal_rng(0, 1);
      }
    }
  }

  simplex[K-1] delta_ = dirichlet_rng(rep_vector(1.0, K-1));
  vector[K] upsilon_;
  upsilon_[1] = 0;
  for (k in 2:K) {
    upsilon_[k] = upsilon_[k-1] + delta_[k-1];
  }

  // --- Generate choice data ---
  array[M_total] int<lower=1> y;
  {
    int pos = 1;
    for (m in 1:M_total) {
      int j = cell[m];
      vector[N_obs[m]] problem_eta;
      for (idx in 1:N_obs[m]) {
        vector[K] psi_i = softmax(beta_[j] * x_flat[pos]);
        problem_eta[idx] = dot_product(psi_i, upsilon_);
        pos += 1;
      }
      y[m] = categorical_rng(softmax(alpha_[j] * problem_eta));
    }
  }
}

parameters {
  real gamma0;
  vector[P] gamma;
  real<lower=0> sigma_cell;
  vector[J] z_alpha;
  array[J] matrix[K, D] beta;
  simplex[K-1] delta;
}

transformed parameters {
  vector[J] log_alpha;
  for (j in 1:J) {
    log_alpha[j] = gamma0 + X[j] * gamma + sigma_cell * z_alpha[j];
  }
  vector<lower=0>[J] alpha = exp(log_alpha);
  ordered[K] upsilon = cumulative_sum(append_row(0, delta));
}

model {
  // Priors (must match generation in transformed data)
  gamma0 ~ normal(3.0, 1.0);
  gamma ~ normal(0, 1.0);
  sigma_cell ~ normal(0, 0.5);
  z_alpha ~ std_normal();

  for (j in 1:J) {
    to_vector(beta[j]) ~ std_normal();
  }

  delta ~ dirichlet(rep_vector(1, K-1));

  // Likelihood
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
      y[m] ~ categorical(softmax(alpha[j] * problem_eta));
    }
  }
}

generated quantities {
  // Copy generated data
  array[M_total] int y_ = y;

  // --- Rank statistics ---
  // Total scalar parameters to track:
  //   gamma0 (1) + gamma (P) + sigma_cell (1) + alpha (J) + delta (K-1)
  //   = 2 + P + J + K - 1
  // Note: We track alpha[j] rather than z_alpha[j] since alpha is the
  // scientifically meaningful quantity.
  // We do NOT track per-cell beta (J*K*D parameters) in SBC — too many
  // parameters make rank histograms unreadable. Beta calibration can be
  // spot-checked separately.

  vector[2 + P + J + (K - 1)] pars_;
  vector[2 + P + J + (K - 1)] ranks_;
  {
    int idx = 1;

    // gamma0
    pars_[idx] = gamma0_;
    ranks_[idx] = (gamma0 > gamma0_) ? 1 : 0;
    idx += 1;

    // gamma
    for (p in 1:P) {
      pars_[idx] = gamma_[p];
      ranks_[idx] = (gamma[p] > gamma_[p]) ? 1 : 0;
      idx += 1;
    }

    // sigma_cell
    pars_[idx] = sigma_cell_;
    ranks_[idx] = (sigma_cell > sigma_cell_) ? 1 : 0;
    idx += 1;

    // alpha (per cell)
    for (j in 1:J) {
      pars_[idx] = alpha_[j];
      ranks_[idx] = (alpha[j] > alpha_[j]) ? 1 : 0;
      idx += 1;
    }

    // delta
    for (k in 1:(K-1)) {
      pars_[idx] = delta_[k];
      ranks_[idx] = (delta[k] > delta_[k]) ? 1 : 0;
      idx += 1;
    }
  }

  // Log-likelihood
  vector[M_total] log_lik;
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
      log_lik[m] = categorical_lpmf(y[m] | softmax(alpha[j] * problem_eta));
    }
  }
}
