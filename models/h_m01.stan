/**
 * Hierarchical Bayesian Decision Theory Model (h_m01)
 *
 * Extends m_01 to handle J experimental cells with:
 * - Cell-specific α_j via regression on log scale
 * - Cell-specific β_j (feature-to-probability mapping)
 * - Shared δ (utility increments) across all cells
 *
 * The regression structure on log(α) enables formal inference about
 * experimental factors (model identity, prompt condition) affecting
 * SEU sensitivity.
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

  // Flatten feature vectors based on I (same pattern as m_01)
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
  // --- Regression on log(α) ---
  real gamma0;                            // intercept (grand mean of log-alpha)
  vector[P] gamma;                        // predictor coefficients
  real<lower=0> sigma_cell;               // residual cell-level SD on log scale
  vector[J] z_alpha;                      // non-centered cell deviations (standard normal)

  // --- Per-cell belief parameters ---
  array[J] matrix[K, D] beta;            // cell-specific feature-to-probability mappings

  // --- Shared utility ---
  simplex[K-1] delta;                     // utility increments (shared)
}

transformed parameters {
  // Cell-specific log-alpha via non-centered parameterization
  vector[J] log_alpha;
  for (j in 1:J) {
    log_alpha[j] = gamma0 + X[j] * gamma + sigma_cell * z_alpha[j];
  }
  vector<lower=0>[J] alpha = exp(log_alpha);

  // Shared ordered utilities
  ordered[K] upsilon = cumulative_sum(append_row(0, delta));

  // Compute choice probabilities for all observations
  vector[total_alts] eta;                 // expected utilities (flat)
  {
    int pos = 1;
    for (m in 1:M_total) {
      int j = cell[m];
      for (idx in 1:N_obs[m]) {
        // Subjective probabilities from cell j's beta
        vector[K] psi_i = softmax(beta[j] * x_flat[pos]);
        eta[pos] = dot_product(psi_i, upsilon);
        pos += 1;
      }
    }
  }
}

model {
  // --- Priors ---
  // Regression
  gamma0 ~ normal(3.0, 1.0);             // prior on intercept: median alpha ~ 20, same scale as m_01
  gamma ~ normal(0, 1.0);                // shrinkage toward no effect
  sigma_cell ~ normal(0, 0.5);           // half-normal (constrained positive): modest cell variation
  z_alpha ~ std_normal();                // non-centered parameterization

  // Beliefs: per-cell, same prior as m_01
  for (j in 1:J) {
    to_vector(beta[j]) ~ std_normal();
  }

  // Utilities: shared
  delta ~ dirichlet(rep_vector(1, K-1));

  // --- Likelihood ---
  {
    int pos = 1;
    for (m in 1:M_total) {
      vector[N_obs[m]] problem_eta = segment(eta, pos, N_obs[m]);
      y[m] ~ categorical(softmax(alpha[cell[m]] * problem_eta));
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
      log_lik[m] = categorical_lpmf(y[m] | softmax(alpha[cell[m]] * problem_eta));
      pos += N_obs[m];
    }
  }

  // Posterior predictive samples
  array[M_total] int y_pred;
  {
    int pos = 1;
    for (m in 1:M_total) {
      vector[N_obs[m]] problem_eta = segment(eta, pos, N_obs[m]);
      y_pred[m] = categorical_rng(softmax(alpha[cell[m]] * problem_eta));
      pos += N_obs[m];
    }
  }

  // === Posterior Predictive Check Statistics ===

  // 1. Log-likelihood discrepancy (global)
  real T_obs_ll = sum(log_lik);
  real T_rep_ll = 0;
  {
    int pos = 1;
    for (m in 1:M_total) {
      vector[N_obs[m]] problem_eta = segment(eta, pos, N_obs[m]);
      T_rep_ll += categorical_lpmf(y_pred[m] | softmax(alpha[cell[m]] * problem_eta));
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
      vector[N_obs[m]] choice_probs = softmax(alpha[cell[m]] * problem_eta);
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
      vector[N_obs[m]] choice_probs = softmax(alpha[cell[m]] * problem_eta);
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
}
