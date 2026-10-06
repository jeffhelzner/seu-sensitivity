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
  real prior_gamma0_mean;
  real<lower=0> prior_gamma0_sd;
  real<lower=0> prior_gamma_sd;
  real<lower=0> prior_sigma_cell_sd;
  real<lower=0> prior_gamma_size_sd;
}
transformed data {
  array[J] int cell_count = rep_array(0, J);
  array[M_total] int<lower=2> N_obs;
  if (utility_values[1] != 0 || utility_values[K] != 1)
    reject("utility_values must have endpoints 0 and 1");
  for (outcome in 2:K)
    if (utility_values[outcome] <= utility_values[outcome - 1])
      reject("utility_values must be strictly increasing");
  if (min([prior_gamma0_sd, prior_gamma_sd, prior_sigma_cell_sd, prior_gamma_size_sd]) <= 0)
    reject("Prior scales must be positive");
  for (observation in 1:M_total) {
    cell_count[cell[observation]] += 1;
    N_obs[observation] = sum(I[observation]);
    if (y[observation] > N_obs[observation])
      reject("Choice exceeds active menu size");
  }
  for (cell_index in 1:J)
    if (cell_count[cell_index] != M_per_cell[cell_index])
      reject("Cell counts disagree");
  for (observation in 1:M_total)
    if (abs(s[observation] - (N_obs[observation] - mean(to_vector(N_obs)))) > 1e-6)
      reject("Size must equal centered menu size");
}
parameters {
  real gamma0;
  vector[P] gamma;
  real gamma_size;
  real<lower=0> sigma_cell;
  vector[J] z_alpha;
}
transformed parameters {
  vector[J] log_alpha_cell = gamma0 + X * gamma + sigma_cell * z_alpha;
  vector<lower=0>[J] alpha_cell = exp(log_alpha_cell);
  vector[M_total] log_alpha_obs;
  vector<lower=0>[M_total] alpha_obs;
  simplex[K - 1] delta;
  ordered[K] upsilon = utility_values;
  for (observation in 1:M_total)
    log_alpha_obs[observation] = log_alpha_cell[cell[observation]] + gamma_size * s[observation];
  alpha_obs = exp(log_alpha_obs);
  for (outcome in 1:(K - 1))
    delta[outcome] = utility_values[outcome + 1] - utility_values[outcome];
}
model {
  gamma0 ~ normal(prior_gamma0_mean, prior_gamma0_sd);
  gamma ~ normal(0, prior_gamma_sd);
  gamma_size ~ normal(0, prior_gamma_size_sd);
  sigma_cell ~ normal(0, prior_sigma_cell_sd);
  z_alpha ~ std_normal();
  for (observation in 1:M_total) {
    vector[N_obs[observation]] problem_eta;
    int position = 1;
    for (item in 1:R)
      if (I[observation, item] == 1) {
        problem_eta[position] = eta[cell[observation], item];
        position += 1;
      }
    y[observation] ~ categorical_logit(alpha_obs[observation] * (problem_eta - max(problem_eta)));
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
  vector[5] prior_settings = [prior_gamma0_mean, prior_gamma0_sd, prior_gamma_sd,
                              prior_sigma_cell_sd, prior_gamma_size_sd]';
  for (observation in 1:M_total) {
    vector[N_obs[observation]] problem_eta;
    vector[N_obs[observation]] logits;
    vector[N_obs[observation]] choice_probs;
    int position = 1;
    for (item in 1:R)
      if (I[observation, item] == 1) {
        problem_eta[position] = eta[cell[observation], item];
        position += 1;
      }
    logits = alpha_obs[observation] * (problem_eta - max(problem_eta));
    choice_probs = softmax(logits);
    log_lik[observation] = categorical_logit_lpmf(y[observation] | logits);
    y_pred[observation] = categorical_rng(choice_probs);
    T_obs_ll += log_lik[observation];
    T_rep_ll += categorical_logit_lpmf(y_pred[observation] | logits);
    T_obs_modal += (choice_probs[y[observation]] >= max(choice_probs) - 1e-9) ? 1 : 0;
    T_rep_modal += (choice_probs[y_pred[observation]] >= max(choice_probs) - 1e-9) ? 1 : 0;
    T_obs_prob += choice_probs[y[observation]];
    T_rep_prob += choice_probs[y_pred[observation]];
    log_lik_cell[cell[observation]] += log_lik[observation];
  }
  int<lower=0,upper=1> ppc_ll = (T_rep_ll >= T_obs_ll) ? 1 : 0;
  int<lower=0,upper=1> ppc_modal = (T_rep_modal >= T_obs_modal) ? 1 : 0;
  int<lower=0,upper=1> ppc_prob = (T_rep_prob >= T_obs_prob) ? 1 : 0;
}