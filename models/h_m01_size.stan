/**
 * Hierarchical Bayesian Decision Theory Model, menu-size variant (h_m01_size)
 *
 * RQ6 (study plan §2, §3.4, §4). This is a *bounded* change to the frozen
 * h_m01: it adds exactly one parameter, gamma_size, and makes alpha vary by
 * observation rather than only by cell:
 *
 *   log alpha_m = gamma0 + X[cell(m)] * gamma + gamma_size * s_m
 *                 + sigma_cell * z_alpha[cell(m)]
 *
 * where s_m is the CENTERED menu size (N_m minus its pool mean). Everything
 * else -- the belief map beta_j, the shared utility increments delta, the
 * softmax choice rule, the priors, and the PPC statistics -- is identical to
 * h_m01, so this variant inherits that model's validation rather than
 * requiring one from scratch (§8.6).
 *
 * Two consequences of centering s that matter for reading the output:
 *
 *   1. alpha_cell[j] = exp(gamma0 + X[j]*gamma + sigma_cell*z_alpha[j]) is
 *      alpha AT THE POOL'S MEAN MENU SIZE, not at size zero. It is therefore
 *      directly comparable to h_m01's alpha[j], which is what makes the two
 *      models' RQ1-RQ3 estimands commensurable.
 *   2. gamma_size is the change in log alpha per ADDITIONAL ALTERNATIVE. It is
 *      identified from the spread of menu sizes *within* each cell, which is
 *      why the design must not collapse that spread (§3.1, §3.4).
 *
 * Linear coding is deliberate: it costs one parameter, so it does not worsen
 * the saturation budget of §8.1. A per-size factor would cost one parameter
 * per size and would not answer RQ6's directional question.
 */
data {
  // --- Dimensions ---
  int<lower=1> J;                         // number of experimental cells
  int<lower=2> K;                         // number of consequences
  int<lower=1> D;                         // embedding dimensions per alternative
  int<lower=2> R;                         // number of distinct alternatives (shared pool)
  int<lower=1> P;                         // number of predictors in design matrix (excluding intercept)

  // --- Shared alternatives ---
  array[R] vector[D] w;                   // feature vectors for alternatives (shared across cells)

  // --- Stacked observations ---
  int<lower=1> M_total;                   // total observations across all cells
  array[M_total] int<lower=1,upper=J> cell; // cell membership for each observation
  array[M_total, R] int<lower=0,upper=1> I; // indicator: I[m,r]=1 if alt r available in obs m
  array[M_total] int<lower=1> y;          // observed choices (1-indexed within active set)

  // --- Menu-size covariate (RQ6) ---
  vector[M_total] s;                      // centered menu size for each observation

  // --- Cell-level design matrix ---
  matrix[J, P] X;                         // predictor matrix (centered/coded), no intercept column

  // --- Per-cell observation counts (for bookkeeping/validation) ---
  array[J] int<lower=1> M_per_cell;       // M_per_cell[j] = number of obs in cell j
}

transformed data {
  // Validate stacked structure
  {
    array[J] int cell_count = rep_array(0, J);
    for (m in 1:M_total) {
      cell_count[cell[m]] += 1;
    }
    for (j in 1:J) {
      if (cell_count[j] != M_per_cell[j])
        reject("cell_count[", j, "] = ", cell_count[j],
               " but M_per_cell[", j, "] = ", M_per_cell[j]);
    }
  }

  // Calculate number of alternatives per observation
  array[M_total] int<lower=2> N_obs;
  int total_alts = 0;
  for (m in 1:M_total) {
    N_obs[m] = sum(I[m]);
    total_alts += N_obs[m];
    if (y[m] > N_obs[m])
      reject("y[", m, "] = ", y[m], " must be <= N_obs[", m, "] = ", N_obs[m]);
  }

  // s must actually be the centered menu size. Getting this wrong (passing raw
  // sizes, or centering on a different mean) would silently shift gamma0 and
  // change what alpha_cell means, with no other symptom -- so check it rather
  // than trust the caller.
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

  // Flatten feature vectors based on I (same pattern as h_m01)
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

parameters {
  // --- Regression on log(alpha) ---
  real gamma0;                            // intercept (grand mean of log-alpha at mean menu size)
  vector[P] gamma;                        // predictor coefficients
  real gamma_size;                        // RQ6: menu-size slope on log-alpha
  real<lower=0> sigma_cell;               // residual cell-level SD on log scale
  vector[J] z_alpha;                      // non-centered cell deviations (standard normal)

  // --- Per-cell belief parameters ---
  array[J] matrix[K, D] beta;             // cell-specific feature-to-probability mappings

  // --- Shared utility ---
  simplex[K-1] delta;                     // utility increments (shared)
}

transformed parameters {
  // Cell-level log-alpha at the MEAN menu size (s is centered)
  vector[J] log_alpha_cell;
  for (j in 1:J) {
    log_alpha_cell[j] = gamma0 + X[j] * gamma + sigma_cell * z_alpha[j];
  }
  vector<lower=0>[J] alpha_cell = exp(log_alpha_cell);

  // Observation-level alpha: the cell effect plus the menu-size slope
  vector[M_total] log_alpha_obs;
  for (m in 1:M_total) {
    log_alpha_obs[m] = log_alpha_cell[cell[m]] + gamma_size * s[m];
  }
  vector<lower=0>[M_total] alpha_obs = exp(log_alpha_obs);

  // Shared ordered utilities
  ordered[K] upsilon = cumulative_sum(append_row(0, delta));

  // Compute expected utilities for all observations
  vector[total_alts] eta;
  {
    int pos = 1;
    for (m in 1:M_total) {
      int j = cell[m];
      for (idx in 1:N_obs[m]) {
        vector[K] psi_i = softmax(beta[j] * x_flat[pos]);
        eta[pos] = dot_product(psi_i, upsilon);
        pos += 1;
      }
    }
  }
}

model {
  // --- Priors (identical to h_m01; gamma_size gets the same shrinkage as the
  //     other regression coefficients, per §4) ---
  gamma0 ~ normal(2.5, 0.5);
  gamma ~ normal(0, 0.5);
  gamma_size ~ normal(0, 0.5);
  sigma_cell ~ normal(0, 0.3);
  z_alpha ~ std_normal();

  for (j in 1:J) {
    to_vector(beta[j]) ~ std_normal();
  }

  delta ~ dirichlet(rep_vector(1, K-1));

  // --- Likelihood ---
  {
    int pos = 1;
    for (m in 1:M_total) {
      vector[N_obs[m]] problem_eta = segment(eta, pos, N_obs[m]);
      y[m] ~ categorical(softmax(alpha_obs[m] * problem_eta));
      pos += N_obs[m];
    }
  }
}

generated quantities {
  // Log-likelihood per observation
  vector[M_total] log_lik;
  {
    int pos = 1;
    for (m in 1:M_total) {
      vector[N_obs[m]] problem_eta = segment(eta, pos, N_obs[m]);
      log_lik[m] = categorical_lpmf(y[m] | softmax(alpha_obs[m] * problem_eta));
      pos += N_obs[m];
    }
  }

  // Posterior predictive samples
  array[M_total] int y_pred;
  {
    int pos = 1;
    for (m in 1:M_total) {
      vector[N_obs[m]] problem_eta = segment(eta, pos, N_obs[m]);
      y_pred[m] = categorical_rng(softmax(alpha_obs[m] * problem_eta));
      pos += N_obs[m];
    }
  }

  // === Posterior Predictive Check Statistics (as in h_m01) ===

  // 1. Log-likelihood discrepancy (global)
  real T_obs_ll = sum(log_lik);
  real T_rep_ll = 0;
  {
    int pos = 1;
    for (m in 1:M_total) {
      vector[N_obs[m]] problem_eta = segment(eta, pos, N_obs[m]);
      T_rep_ll += categorical_lpmf(y_pred[m] | softmax(alpha_obs[m] * problem_eta));
      pos += N_obs[m];
    }
  }
  int<lower=0,upper=1> ppc_ll = (T_rep_ll >= T_obs_ll) ? 1 : 0;

  // 2. Modal choice accuracy (global)
  int T_obs_modal = 0;
  int T_rep_modal = 0;
  {
    int pos = 1;
    for (m in 1:M_total) {
      vector[N_obs[m]] problem_eta = segment(eta, pos, N_obs[m]);
      vector[N_obs[m]] choice_probs = softmax(alpha_obs[m] * problem_eta);
      real max_prob = max(choice_probs);
      T_obs_modal += (choice_probs[y[m]] >= max_prob - 1e-9) ? 1 : 0;
      T_rep_modal += (choice_probs[y_pred[m]] >= max_prob - 1e-9) ? 1 : 0;
      pos += N_obs[m];
    }
  }
  int<lower=0,upper=1> ppc_modal = (T_rep_modal >= T_obs_modal) ? 1 : 0;

  // 3. Sum of chosen probabilities (global)
  real T_obs_prob = 0;
  real T_rep_prob = 0;
  {
    int pos = 1;
    for (m in 1:M_total) {
      vector[N_obs[m]] problem_eta = segment(eta, pos, N_obs[m]);
      vector[N_obs[m]] choice_probs = softmax(alpha_obs[m] * problem_eta);
      T_obs_prob += choice_probs[y[m]];
      T_rep_prob += choice_probs[y_pred[m]];
      pos += N_obs[m];
    }
  }
  int<lower=0,upper=1> ppc_prob = (T_rep_prob >= T_obs_prob) ? 1 : 0;

  // 4. Per-cell log-likelihoods (for cell-level model comparison)
  vector[J] log_lik_cell = rep_vector(0, J);
  for (m in 1:M_total) {
    log_lik_cell[cell[m]] += log_lik[m];
  }

  // 5. RQ6 reporting: multiplicative change in alpha per additional
  //    alternative, which is the scale the slope is discussed on.
  real alpha_ratio_per_alt = exp(gamma_size);
}
