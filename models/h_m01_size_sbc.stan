/**
 * Hierarchical SBC Model, menu-size variant (h_m01_size_sbc)
 *
 * Draws true parameters from the h_m01_size prior in transformed data,
 * generates choice data, then fits the model to that data and emits rank
 * statistics (study plan §8.6).
 *
 * The generation block and the model block must draw from and declare the SAME
 * priors, or the rank histograms are meaningless. gamma_size is drawn from
 * normal(0, 0.5) here and given normal(0, 0.5) in the model block, matching
 * h_m01_size.stan and §4.
 *
 * Note this is SBC, not the null-calibration check: here gamma_size is drawn
 * from its prior and must be recovered with uniform ranks. The null check
 * (gamma_size fixed at 0) lives in h_m01_size_sim.stan, because it asks a
 * different question -- whether a size slope appears when there is none.
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

  // --- Menu-size covariate (RQ6) ---
  vector[M_total] s;
}

transformed data {
  // Compute alternative counts
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

  // --- Draw true parameters (must match the model block's priors) ---
  real gamma0_ = normal_rng(2.5, 0.5);

  vector[P] gamma_;
  for (p in 1:P) {
    gamma_[p] = normal_rng(0, 0.5);
  }

  real gamma_size_ = normal_rng(0, 0.5);

  real<lower=0> sigma_cell_ = abs(normal_rng(0, 0.3));

  vector[J] z_alpha_;
  vector[J] log_alpha_cell_;
  for (j in 1:J) {
    z_alpha_[j] = normal_rng(0, 1);
    log_alpha_cell_[j] = gamma0_ + X[j] * gamma_ + sigma_cell_ * z_alpha_[j];
  }
  vector<lower=0>[J] alpha_cell_ = exp(log_alpha_cell_);

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
      real alpha_m = exp(log_alpha_cell_[j] + gamma_size_ * s[m]);
      vector[N_obs[m]] problem_eta;
      for (idx in 1:N_obs[m]) {
        vector[K] psi_i = softmax(beta_[j] * x_flat[pos]);
        problem_eta[idx] = dot_product(psi_i, upsilon_);
        pos += 1;
      }
      y[m] = categorical_rng(softmax(alpha_m * problem_eta));
    }
  }
}

parameters {
  real gamma0;
  vector[P] gamma;
  real gamma_size;
  real<lower=0> sigma_cell;
  vector[J] z_alpha;
  array[J] matrix[K, D] beta;
  simplex[K-1] delta;
}

transformed parameters {
  vector[J] log_alpha_cell;
  for (j in 1:J) {
    log_alpha_cell[j] = gamma0 + X[j] * gamma + sigma_cell * z_alpha[j];
  }
  vector<lower=0>[J] alpha_cell = exp(log_alpha_cell);
  ordered[K] upsilon = cumulative_sum(append_row(0, delta));
}

model {
  // Priors (must match generation in transformed data)
  gamma0 ~ normal(2.5, 0.5);
  gamma ~ normal(0, 0.5);
  gamma_size ~ normal(0, 0.5);
  sigma_cell ~ normal(0, 0.3);
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
      real alpha_m = exp(log_alpha_cell[j] + gamma_size * s[m]);
      vector[N_obs[m]] problem_eta;
      for (idx in 1:N_obs[m]) {
        vector[K] psi_i = softmax(beta[j] * x_flat[pos]);
        problem_eta[idx] = dot_product(psi_i, upsilon);
        pos += 1;
      }
      y[m] ~ categorical(softmax(alpha_m * problem_eta));
    }
  }
}

generated quantities {
  // Copy generated data
  array[M_total] int y_ = y;

  // --- Rank statistics ---
  // Tracked scalars:
  //   gamma0 (1) + gamma (P) + gamma_size (1) + sigma_cell (1)
  //   + alpha_cell (J) + delta (K-1)  =  3 + P + J + K - 1
  // As in h_m01_sbc, per-cell beta is not tracked: J*K*D rank histograms are
  // unreadable, and §4's inherited caveat is that beta and delta are only
  // weakly informed anyway.
  vector[3 + P + J + (K - 1)] pars_;
  vector[3 + P + J + (K - 1)] ranks_;
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

    // gamma_size (RQ6)
    pars_[idx] = gamma_size_;
    ranks_[idx] = (gamma_size > gamma_size_) ? 1 : 0;
    idx += 1;

    // sigma_cell
    pars_[idx] = sigma_cell_;
    ranks_[idx] = (sigma_cell > sigma_cell_) ? 1 : 0;
    idx += 1;

    // alpha_cell (per cell, at the mean menu size)
    for (j in 1:J) {
      pars_[idx] = alpha_cell_[j];
      ranks_[idx] = (alpha_cell[j] > alpha_cell_[j]) ? 1 : 0;
      idx += 1;
    }

    // delta
    for (k in 1:(K-1)) {
      pars_[idx] = delta_[k];
      ranks_[idx] = (delta[k] > delta_[k]) ? 1 : 0;
      idx += 1;
    }
  }
}
