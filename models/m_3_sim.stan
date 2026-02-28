/**
 * Data Simulation Model for m_3
 * 
 * Generates synthetic data for combined uncertain and risky choice problems
 * with proportionally related sensitivity parameters:
 *   - alpha: sensitivity for uncertain choices
 *   - kappa: association parameter
 *   - omega = kappa * alpha: sensitivity for risky choices
 *
 * Features (w) and risky probabilities (x) are provided as input.
 */
data {
  // Uncertain decision problems
  int<lower=1> M;                 // number of uncertain decision problems
  int<lower=2> K;                 // number of possible consequences
  int<lower=1> D;                 // dimensions of alternative features
  int<lower=2> R;                 // number of distinct uncertain alternatives
  array[R] vector[D] w;           // descriptions of each uncertain alternative
  array[M,R] int<lower=0,upper=1> I; // indicator: I[m,r] = 1 if alternative r in problem m
  
  // Risky decision problems
  int<lower=1> N;                 // number of risky decision problems
  int<lower=2> S;                 // number of distinct risky alternatives
  array[S] simplex[K] x;          // objective probability simplexes for risky alternatives
  array[N,S] int<lower=0,upper=1> J; // indicator: J[n,s] = 1 if risky alt s in problem n
  
  // Parameter generation controls
  real<lower=0> alpha_mean;       // mean for log(alpha) (default=0)
  real<lower=0> alpha_sd;         // sd for log(alpha) (default=1)
  real<lower=0> kappa_mean;       // mean for log(kappa) (default=0)
  real<lower=0> kappa_sd;         // sd for log(kappa) (default=0.5)
  real<lower=0> beta_sd;          // sd for beta coefficients (default=1)
}

transformed data {
  // Calculate counts for uncertain problems
  array[M] int<lower=2> N_uncertain;
  int total_uncertain_alts = 0;
  
  for (m in 1:M) {
    N_uncertain[m] = sum(I[m]);
    total_uncertain_alts += N_uncertain[m];
  }
  
  // Construct feature vectors for uncertain alternatives
  array[total_uncertain_alts] vector[D] x_uncertain;
  {
    int pos = 1;
    for (m in 1:M) {
      for (r in 1:R) {
        if (I[m, r] == 1) {
          x_uncertain[pos] = w[r];
          pos += 1;
        }
      }
    }
  }
  
  // Calculate counts for risky problems
  array[N] int<lower=2> N_risky;
  int total_risky_alts = 0;
  
  for (n in 1:N) {
    N_risky[n] = sum(J[n]);
    total_risky_alts += N_risky[n];
  }
  
  // Flatten risky probability simplexes
  array[total_risky_alts] simplex[K] x_risky;
  {
    int pos = 1;
    for (n in 1:N) {
      for (s in 1:S) {
        if (J[n, s] == 1) {
          x_risky[pos] = x[s];
          pos += 1;
        }
      }
    }
  }
}

generated quantities {
  // Generate model parameters
  real alpha = lognormal_rng(alpha_mean, alpha_sd);
  real kappa = lognormal_rng(kappa_mean, kappa_sd);
  real omega = kappa * alpha;  // derived sensitivity for risky choices
  
  matrix[K,D] beta;
  for (k in 1:K) {
    for (d in 1:D) {
      beta[k,d] = normal_rng(0, beta_sd);
    }
  }
  
  simplex[K-1] delta = dirichlet_rng(rep_vector(1.0, K-1));
  
  // Construct shared utility function
  vector[K] upsilon = cumulative_sum(append_row(0, delta));
  
  // === UNCERTAIN CHOICE SIMULATION ===
  array[total_uncertain_alts] vector[K] psi; // subjective probabilities
  
  // Calculate subjective probabilities
  for (i in 1:total_uncertain_alts) {
    psi[i] = softmax(beta * x_uncertain[i]);
  }
  
  // Calculate expected utilities for uncertain alternatives
  vector[total_uncertain_alts] eta_uncertain;
  for (i in 1:total_uncertain_alts) {
    eta_uncertain[i] = dot_product(to_vector(psi[i]), upsilon);
  }
  
  // Store expected utilities and choice probabilities for uncertain problems
  array[M] vector[max(N_uncertain)] uncertain_etas;
  array[M] vector[max(N_uncertain)] uncertain_probs;
  for (m in 1:M) {
    uncertain_etas[m] = rep_vector(-1e10, max(N_uncertain));
    uncertain_probs[m] = rep_vector(0, max(N_uncertain));
  }
  
  // Generate choices for uncertain problems (using alpha)
  array[M] int y;
  array[M] int<lower=0,upper=1> selected_seu_max_uncertain;
  
  {
    int pos = 1;
    for (m in 1:M) {
      vector[N_uncertain[m]] problem_eta = segment(eta_uncertain, pos, N_uncertain[m]);
      
      // Store expected utilities
      for (j in 1:N_uncertain[m]) {
        uncertain_etas[m][j] = problem_eta[j];
      }
      
      // Calculate and store choice probabilities
      vector[N_uncertain[m]] choice_probs = softmax(alpha * problem_eta);
      for (j in 1:N_uncertain[m]) {
        uncertain_probs[m][j] = choice_probs[j];
      }
      
      // Generate choice
      y[m] = categorical_rng(choice_probs);
      
      // Check if SEU maximizer was selected
      real max_eta = max(problem_eta);
      if (abs(problem_eta[y[m]] - max_eta) < 1e-10) {
        selected_seu_max_uncertain[m] = 1;
      } else {
        selected_seu_max_uncertain[m] = 0;
      }
      
      pos += N_uncertain[m];
    }
  }
  
  int total_seu_max_selected_uncertain = sum(selected_seu_max_uncertain);
  
  // === RISKY CHOICE SIMULATION ===
  
  // Calculate expected utilities for risky alternatives
  vector[total_risky_alts] eta_risky;
  for (i in 1:total_risky_alts) {
    eta_risky[i] = dot_product(to_vector(x_risky[i]), upsilon);
  }
  
  // Store expected utilities and choice probabilities for risky problems
  array[N] vector[max(N_risky)] risky_etas;
  array[N] vector[max(N_risky)] risky_probs;
  for (n in 1:N) {
    risky_etas[n] = rep_vector(-1e10, max(N_risky));
    risky_probs[n] = rep_vector(0, max(N_risky));
  }
  
  // Generate choices for risky problems (using omega = kappa * alpha)
  array[N] int z;
  array[N] int<lower=0,upper=1> selected_seu_max_risky;
  
  {
    int pos = 1;
    for (n in 1:N) {
      vector[N_risky[n]] problem_eta = segment(eta_risky, pos, N_risky[n]);
      
      // Store expected utilities
      for (j in 1:N_risky[n]) {
        risky_etas[n][j] = problem_eta[j];
      }
      
      // Calculate and store choice probabilities
      vector[N_risky[n]] choice_probs = softmax(omega * problem_eta);
      for (j in 1:N_risky[n]) {
        risky_probs[n][j] = choice_probs[j];
      }
      
      // Generate choice
      z[n] = categorical_rng(choice_probs);
      
      // Check if SEU maximizer was selected
      real max_eta = max(problem_eta);
      if (abs(problem_eta[z[n]] - max_eta) < 1e-10) {
        selected_seu_max_risky[n] = 1;
      } else {
        selected_seu_max_risky[n] = 0;
      }
      
      pos += N_risky[n];
    }
  }
  
  int total_seu_max_selected_risky = sum(selected_seu_max_risky);
  int total_seu_max_selected = total_seu_max_selected_uncertain + total_seu_max_selected_risky;
}
