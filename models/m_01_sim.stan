/**
 * Data Simulation Model for m_01
 *
 * Structurally identical to m_0_sim.  The caller controls the
 * alpha prior through the alpha_mean / alpha_sd data inputs,
 * so no Stan-level changes are needed; only the values passed
 * at runtime differ between m_0 and m_01.
 *
 * For m_01 the intended inputs are alpha_mean=3.0, alpha_sd=0.75
 * (i.e. lognormal(3.0, 0.75), calibrated via prior predictive analysis).
 */
data {
  // Simulation control parameters
  int<lower=1> M;                 // number of decision problems to generate
  int<lower=2> K;                 // number of possible consequences
  int<lower=1> D;                 // dimensions of alternative features
  int<lower=2> R;                 // number of distinct alternatives
  array[R] vector[D] w;           // descriptions of each distinct alternative
  array[M,R] int<lower=0,upper=1> I; // indicator array: I[m,r] = 1 if alternative r is in problem m
  
  // Parameter generation controls
  real<lower=0> alpha_mean;       // mean for alpha parameter (default=1)
  real<lower=0> alpha_sd;         // sd for alpha parameter (default=0.5)
  real<lower=0> beta_sd;          // sd for beta coefficients (default=1)
}

transformed data {
  // Calculate N and total_alts
  array[M] int<lower=2> N;        // number of alternatives in each decision problem
  int total_alts = 0;
  
  for (m in 1:M) {
    N[m] = sum(I[m]);
    total_alts += N[m];
  }
  
  // Construct x from w and I
  array[total_alts] vector[D] x;
  {
    int pos = 1;
    for (m in 1:M) {
      for (r in 1:R) {
        if (I[m, r] == 1) {
          x[pos] = w[r];
          pos += 1;
        }
      }
    }
  }
}

generated quantities {
  // Generate model parameters
  real alpha = lognormal_rng(alpha_mean, alpha_sd);
  
  matrix[K,D] beta;
  for (k in 1:K) {
    for (d in 1:D) {
      beta[k,d] = normal_rng(0, beta_sd);
    }
  }
  
  // Generate utility differences using Dirichlet prior
  simplex[K-1] delta = dirichlet_rng(rep_vector(1.0, K-1));
  
  // Subjective probabilities over consequences
  array[total_alts] vector[K] psi;
  for (i in 1:total_alts) {
    psi[i] = softmax(beta * x[i]);
  }
  
  // Construct ordered utilities
  vector[K] upsilon = cumulative_sum(append_row(0, delta));
  
  // Calculate expected utilities
  vector[total_alts] eta;
  for (i in 1:total_alts) {
    eta[i] = dot_product(to_vector(psi[i]), upsilon);
  }
  
  // Store expected utilities for each decision problem (for easier extraction)
  array[M] vector[max(N)] problem_etas;
  for (i in 1:M) {
    problem_etas[i] = rep_vector(-1e10, max(N)); // Initialize with very negative values
  }
  
  // Store choice probabilities for each decision problem
  array[M] vector[max(N)] choice_probabilities;
  for (i in 1:M) {
    choice_probabilities[i] = rep_vector(0, max(N)); // Initialize with zeros
  }
  
  // Generate choices
  array[M] int y;
  // Indicator: 1 if SEU maximizer was selected, 0 otherwise
  array[M] int<lower=0,upper=1> selected_seu_max;
  // Total number of problems where SEU maximizer was selected
  int<lower=0,upper=M> total_seu_max_selected;
  
  {
    int pos = 1;
    for (i in 1:M) {
      vector[N[i]] problem_eta = segment(eta, pos, N[i]);
      
      // Store the expected utilities for this problem (for extraction)
      for (j in 1:N[i]) {
        problem_etas[i][j] = problem_eta[j];
      }
      
      // Calculate choice probabilities
      vector[N[i]] choice_probs = softmax(alpha * problem_eta);
      
      // Store the choice probabilities (for extraction)
      for (j in 1:N[i]) {
        choice_probabilities[i][j] = choice_probs[j];
      }
      
      // Generate choice from categorical distribution
      y[i] = categorical_rng(choice_probs);
      
      // Determine if an SEU maximizer was selected
      // Find the maximum expected utility for this problem
      real max_eta = max(problem_eta);
      
      // Check if the selected alternative has the maximum expected utility
      // (within numerical tolerance)
      if (abs(problem_eta[y[i]] - max_eta) < 1e-10) {
        selected_seu_max[i] = 1;
      } else {
        selected_seu_max[i] = 0;
      }
      
      pos += N[i];
    }
  }
  
  // Calculate total number of problems where SEU maximizer was selected
  total_seu_max_selected = sum(selected_seu_max);
}



