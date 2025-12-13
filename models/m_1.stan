/**
 * Bayesian Decision Theory Model (m_1)
 * 
 * Extends m_0 by combining decision-making under uncertainty and risk.
 * This allows separate identification of utilities and subjective probabilities.
 * 
 * - Uncertain choice problems: probabilities derived from features via beta
 * - Risky choice problems: objective probabilities given as data
 * - Shared: same alpha sensitivity parameter and same utility function
 */
data {
  // Decision problems under uncertainty
  int<lower=1> M;                    // number of uncertain decision problems
  int<lower=2> K;                    // number of possible consequences
  int<lower=1> D;                    // dimensions of alternative features
  int<lower=2> R;                    // number of distinct uncertain alternatives
  array[R] vector[D] w;              // feature vectors for uncertain alternatives
  array[M,R] int<lower=0,upper=1> I; // indicator: I[m,r] = 1 if alternative r in problem m
  array[M] int<lower=1> y;           // choices for uncertain problems
  
  // Decision problems under risk  
  int<lower=1> N;                    // number of risky decision problems
  int<lower=2> S;                    // number of distinct risky alternatives
  array[S] simplex[K] x;             // objective probability simplexes for risky alternatives
  array[N,S] int<lower=0,upper=1> J; // indicator: J[n,s] = 1 if risky alt s in problem n
  array[N] int<lower=1> z;           // choices for risky problems
}

transformed data {
  // Calculate counts for uncertain problems
  array[M] int<lower=2> N_uncertain;
  int total_uncertain_alts = 0;
  
  for (m in 1:M) {
    N_uncertain[m] = sum(I[m]);
    total_uncertain_alts += N_uncertain[m];
    
    // Validate y is within bounds
    if (y[m] > N_uncertain[m]) {
      reject("y[", m, "] = ", y[m], " must be <= N_uncertain[", m, "] = ", N_uncertain[m]);
    }
  }
  
  // Calculate counts for risky problems
  array[N] int<lower=2> N_risky;
  int total_risky_alts = 0;
  
  for (n in 1:N) {
    N_risky[n] = sum(J[n]);
    total_risky_alts += N_risky[n];
    
    // Validate z is within bounds
    if (z[n] > N_risky[n]) {
      reject("z[", n, "] = ", z[n], " must be <= N_risky[", n, "] = ", N_risky[n]);
    }
  }
  
  // Construct feature vectors for uncertain alternatives (flatten w based on I)
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
  
  // Flatten risky probability simplexes based on J
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

parameters {
  real<lower=0> alpha;           // sensitivity parameter (shared across contexts)
  matrix[K,D] beta;              // feature-to-probability mapping (uncertain only)
  simplex[K-1] delta;            // utility increments (shared across contexts)
}

transformed parameters {
  // Shared utility function
  ordered[K] upsilon = cumulative_sum(append_row(0, delta));
  
  // === UNCERTAIN CHOICE COMPONENTS ===
  array[total_uncertain_alts] simplex[K] psi;  // subjective probabilities
  vector[total_uncertain_alts] eta_uncertain;   // expected utilities
  array[M] simplex[max(N_uncertain)] chi_uncertain; // choice probabilities
  
  // Calculate subjective probabilities for uncertain alternatives
  for (i in 1:total_uncertain_alts) {
    psi[i] = softmax(beta * x_uncertain[i]);
  }
  
  // Calculate expected utilities for uncertain alternatives
  for (i in 1:total_uncertain_alts) {
    eta_uncertain[i] = dot_product(psi[i], upsilon);
  }
  
  // Construct choice probabilities for uncertain problems
  {
    int pos = 1;
    for (m in 1:M) {
      vector[N_uncertain[m]] problem_eta = segment(eta_uncertain, pos, N_uncertain[m]);
      chi_uncertain[m] = append_row(
        softmax(alpha * problem_eta),
        rep_vector(0, max(N_uncertain) - N_uncertain[m])
      );
      pos += N_uncertain[m];
    }
  }
  
  // === RISKY CHOICE COMPONENTS ===
  vector[total_risky_alts] eta_risky;           // expected utilities
  array[N] simplex[max(N_risky)] chi_risky;     // choice probabilities
  
  // Calculate expected utilities for risky alternatives
  for (i in 1:total_risky_alts) {
    eta_risky[i] = dot_product(x_risky[i], upsilon);
  }
  
  // Construct choice probabilities for risky problems
  {
    int pos = 1;
    for (n in 1:N) {
      vector[N_risky[n]] problem_eta = segment(eta_risky, pos, N_risky[n]);
      chi_risky[n] = append_row(
        softmax(alpha * problem_eta),
        rep_vector(0, max(N_risky) - N_risky[n])
      );
      pos += N_risky[n];
    }
  }
}

model {
  // Priors
  alpha ~ lognormal(0, 1);
  to_vector(beta) ~ std_normal();
  delta ~ dirichlet(rep_vector(1, K-1));
  
  // Likelihood for uncertain choices
  for (m in 1:M) {
    y[m] ~ categorical(chi_uncertain[m]);
  }
  
  // Likelihood for risky choices
  for (n in 1:N) {
    z[n] ~ categorical(chi_risky[n]);
  }
}

generated quantities {
  // Separate log-likelihoods
  vector[M] log_lik_uncertain;
  vector[N] log_lik_risky;
  real log_lik_total;
  
  for (m in 1:M) {
    log_lik_uncertain[m] = categorical_lpmf(y[m] | chi_uncertain[m]);
  }
  
  for (n in 1:N) {
    log_lik_risky[n] = categorical_lpmf(z[n] | chi_risky[n]);
  }
  
  log_lik_total = sum(log_lik_uncertain) + sum(log_lik_risky);
  
  // Posterior predictive samples
  array[M] int y_pred;
  array[N] int z_pred;
  
  for (m in 1:M) {
    y_pred[m] = categorical_rng(chi_uncertain[m]);
  }
  
  for (n in 1:N) {
    z_pred[n] = categorical_rng(chi_risky[n]);
  }
}
