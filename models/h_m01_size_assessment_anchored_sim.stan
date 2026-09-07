/**
 * Prior simulator for h_m01_size_assessment_anchored.
 *
 * Expected utilities are fixed data, as they are in the inference model. This
 * lets recovery tests isolate the hierarchical alpha parameters without
 * regenerating or estimating a latent belief map from the simulated choices.
 */
data {
  int<lower=1> J;
  int<lower=2> K;
  int<lower=2> R;
  int<lower=1> P;

  matrix<lower=0,upper=1>[J, R] eta;
  vector<lower=0,upper=1>[K] utility_values;

  array[J] int<lower=1> M_per_cell;
  matrix[J, P] X;
  int<lower=1> M_total;
  array[M_total] int<lower=1,upper=J> cell;
  array[M_total, R] int<lower=0,upper=1> I;
  vector[M_total] s;

  real gamma0_mean;
  real<lower=0> gamma0_sd;
  real<lower=0> gamma_sd;
  real gamma_size_mean;
  real<lower=0> gamma_size_sd;
  real<lower=0> sigma_cell_sd;
}

transformed data {
  array[M_total] int<lower=2> N_obs;

  if (utility_values[1] != 0 || utility_values[K] != 1)
    reject("utility_values must have endpoints 0 and 1");
  for (k in 2:K) {
    if (utility_values[k] <= utility_values[k - 1])
      reject("utility_values must be strictly increasing");
  }

  for (m in 1:M_total) {
    N_obs[m] = sum(I[m]);
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

generated quantities {
  real gamma0 = normal_rng(gamma0_mean, gamma0_sd);
  vector[P] gamma;
  real gamma_size = (gamma_size_sd > 0)
                    ? normal_rng(gamma_size_mean, gamma_size_sd)
                    : gamma_size_mean;
  real<lower=0> sigma_cell = abs(normal_rng(0, sigma_cell_sd));
  vector[J] log_alpha_cell;
  vector[J] alpha_cell;
  vector[M_total] alpha_obs;
  simplex[K - 1] delta;
  ordered[K] upsilon;
  array[M_total] int y;
  array[M_total] int<lower=0,upper=1> selected_seu_max;
  int<lower=0,upper=M_total> total_seu_max_selected;
  array[J] int seu_max_by_cell;
  array[M_total] int<lower=2> menu_size_out;

  for (p in 1:P) {
    gamma[p] = normal_rng(0, gamma_sd);
  }
  for (j in 1:J) {
    real z_j = normal_rng(0, 1);
    log_alpha_cell[j] = gamma0 + X[j] * gamma + sigma_cell * z_j;
    alpha_cell[j] = exp(log_alpha_cell[j]);
    seu_max_by_cell[j] = 0;
  }
  for (m in 1:M_total) {
    alpha_obs[m] = exp(log_alpha_cell[cell[m]] + gamma_size * s[m]);
  }
  for (k in 1:(K - 1)) {
    delta[k] = utility_values[k + 1] - utility_values[k];
  }
  upsilon = utility_values;

  for (m in 1:M_total) {
    int j = cell[m];
    vector[N_obs[m]] problem_eta;
    int pos = 1;
    for (r in 1:R) {
      if (I[m, r] == 1) {
        problem_eta[pos] = eta[j, r];
        pos += 1;
      }
    }
    y[m] = categorical_rng(softmax(alpha_obs[m] * problem_eta));
    menu_size_out[m] = N_obs[m];
    if (abs(problem_eta[y[m]] - max(problem_eta)) < 1e-10) {
      selected_seu_max[m] = 1;
      seu_max_by_cell[j] += 1;
    } else {
      selected_seu_max[m] = 0;
    }
  }
  total_seu_max_selected = sum(selected_seu_max);
}