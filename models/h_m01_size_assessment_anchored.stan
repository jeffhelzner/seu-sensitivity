/**
 * Hierarchical SEU-sensitivity model with assessment-anchored expected utility.
 *
 * Each eta[j,r] is computed before fitting from model j's neutral elicited
 * consequence probabilities and the frozen utility vector. Prompt-condition
 * siblings therefore share eta while retaining distinct alpha cells. This
 * directly implements identification of alpha conditional on fixed eta and
 * avoids estimating a flexible belief map from the same choices.
 */
data {
  int<lower=1> J;
  int<lower=2> K;
  int<lower=2> R;
  int<lower=1> P;

  matrix<lower=0,upper=1>[J, R] eta;
  vector<lower=0,upper=1>[K] utility_values;

  int<lower=1> M_total;
  array[M_total] int<lower=1,upper=J> cell;
  array[M_total, R] int<lower=0,upper=1> I;
  array[M_total] int<lower=1> y;
  vector[M_total] s;

  matrix[J, P] X;
  array[J] int<lower=1> M_per_cell;
}

transformed data {
  array[J] int cell_count = rep_array(0, J);
  array[M_total] int<lower=2> N_obs;
  int total_alts = 0;

  if (utility_values[1] != 0 || utility_values[K] != 1)
    reject("utility_values must have endpoints 0 and 1");
  for (k in 2:K) {
    if (utility_values[k] <= utility_values[k - 1])
      reject("utility_values must be strictly increasing");
  }

  for (m in 1:M_total) {
    cell_count[cell[m]] += 1;
    N_obs[m] = sum(I[m]);
    total_alts += N_obs[m];
    if (y[m] > N_obs[m])
      reject("y[", m, "] = ", y[m], " must be <= N_obs[", m, "] = ", N_obs[m]);
  }
  for (j in 1:J) {
    if (cell_count[j] != M_per_cell[j])
      reject("cell_count[", j, "] = ", cell_count[j],
             " but M_per_cell[", j, "] = ", M_per_cell[j]);
  }

  {
    real mean_size = mean(to_vector(N_obs));
    if (abs(mean(s)) > 1e-6)
      reject("s must be centered (mean 0); got mean(s) = ", mean(s));
    for (m in 1:M_total) {
      if (abs(s[m] - (N_obs[m] - mean_size)) > 1e-6)
        reject("s[", m, "] = ", s[m], " but observation ", m,
               " has menu size ", N_obs[m], " and mean menu size ", mean_size);
    }
  }
}

parameters {
  real gamma0;
  vector[P] gamma;
  real gamma_size;
  real<lower=0> sigma_cell;
  vector[J] z_alpha;
}

transformed parameters {
  vector[J] log_alpha_cell;
  vector<lower=0>[J] alpha_cell;
  vector[M_total] log_alpha_obs;
  vector<lower=0>[M_total] alpha_obs;
  simplex[K - 1] delta;
  ordered[K] upsilon;

  for (j in 1:J) {
    log_alpha_cell[j] = gamma0 + X[j] * gamma + sigma_cell * z_alpha[j];
  }
  alpha_cell = exp(log_alpha_cell);

  for (m in 1:M_total) {
    log_alpha_obs[m] = log_alpha_cell[cell[m]] + gamma_size * s[m];
  }
  alpha_obs = exp(log_alpha_obs);

  for (k in 1:(K - 1)) {
    delta[k] = utility_values[k + 1] - utility_values[k];
  }
  upsilon = utility_values;
}

model {
  gamma0 ~ normal(2.5, 0.5);
  gamma ~ normal(0, 0.5);
  gamma_size ~ normal(0, 0.5);
  sigma_cell ~ normal(0, 0.3);
  z_alpha ~ std_normal();

  for (m in 1:M_total) {
    vector[N_obs[m]] problem_eta;
    int pos = 1;
    for (r in 1:R) {
      if (I[m, r] == 1) {
        problem_eta[pos] = eta[cell[m], r];
        pos += 1;
      }
    }
    y[m] ~ categorical_logit(alpha_obs[m] * problem_eta);
  }
}

generated quantities {
  vector[M_total] log_lik;
  array[M_total] int y_pred;
  real T_obs_ll = 0;
  real T_rep_ll = 0;
  int T_obs_modal = 0;
  int T_rep_modal = 0;
  real T_obs_prob = 0;
  real T_rep_prob = 0;
  vector[J] log_lik_cell = rep_vector(0, J);
  real alpha_ratio_per_alt = exp(gamma_size);

  for (m in 1:M_total) {
    vector[N_obs[m]] problem_eta;
    vector[N_obs[m]] choice_probs;
    int pos = 1;
    for (r in 1:R) {
      if (I[m, r] == 1) {
        problem_eta[pos] = eta[cell[m], r];
        pos += 1;
      }
    }
    choice_probs = softmax(alpha_obs[m] * problem_eta);
    log_lik[m] = categorical_lpmf(y[m] | choice_probs);
    y_pred[m] = categorical_rng(choice_probs);
    T_obs_ll += log_lik[m];
    T_rep_ll += categorical_lpmf(y_pred[m] | choice_probs);

    {
      real max_prob = max(choice_probs);
      T_obs_modal += (choice_probs[y[m]] >= max_prob - 1e-9) ? 1 : 0;
      T_rep_modal += (choice_probs[y_pred[m]] >= max_prob - 1e-9) ? 1 : 0;
    }
    T_obs_prob += choice_probs[y[m]];
    T_rep_prob += choice_probs[y_pred[m]];
    log_lik_cell[cell[m]] += log_lik[m];
  }

  int<lower=0,upper=1> ppc_ll = (T_rep_ll >= T_obs_ll) ? 1 : 0;
  int<lower=0,upper=1> ppc_modal = (T_rep_modal >= T_obs_modal) ? 1 : 0;
  int<lower=0,upper=1> ppc_prob = (T_rep_prob >= T_obs_prob) ? 1 : 0;
}