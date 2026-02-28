/**
 * Bayesian Decision Theory Model (m_3) for Simulation Based Calibration
 * 
 * Extends m_1 SBC with proportionally related sensitivity parameters:
 *   - alpha: sensitivity for uncertain choices
 *   - kappa: association parameter (omega = kappa * alpha)
 *   - omega: sensitivity for risky choices (derived, not free)
 *
 * Following conventions required by rstan's sbc() function for combined
 * uncertain and risky choice data.
 */
data {
  // Uncertain decision problems
  int<lower=1> M;                  // number of uncertain decision problems
  int<lower=2> K;                  // number of possible consequences
  int<lower=1> D;                  // dimensions to describe alternatives
  int<lower=2> R;                  // number of distinct uncertain alternatives
  array[R] vector[D] w;            // descriptions of each uncertain alternative
  array[M,R] int<lower=0,upper=1> I; // indicator: I[m,r] = 1 if alternative r in problem m
  
  // Risky decision problems
  int<lower=1> N;                  // number of risky decision problems
  int<lower=2> S;                  // number of distinct risky alternatives
  array[S] simplex[K] x;           // objective probability simplexes for risky alternatives
  array[N,S] int<lower=0,upper=1> J; // indicator: J[n,s] = 1 if risky alt s in problem n
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
  
  // Draw "true" parameters (with trailing underscore as required)
  real<lower=0> alpha_ = lognormal_rng(0, 1);
  real<lower=0> kappa_ = lognormal_rng(0, 0.5);
  real<lower=0> omega_ = kappa_ * alpha_;  // derived sensitivity for risky choices
  
  matrix[K,D] beta_;
  for (k in 1:K) {
    for (d in 1:D) {
      beta_[k,d] = normal_rng(0, 1);
    }
  }
  
  simplex[K-1] delta_ = dirichlet_rng(rep_vector(1.0, K-1));
  
  // Construct utilities from delta_
  vector[K] upsilon_;
  upsilon_[1] = 0;
  for (k in 2:K) {
    upsilon_[k] = upsilon_[k-1] + delta_[k-1]; 
  }
  
  // === Generate uncertain choice data ===
  array[total_uncertain_alts] vector[K] psi_;
  vector[total_uncertain_alts] eta_uncertain_;
  
  // Calculate subjective probabilities
  for (i in 1:total_uncertain_alts) {
    psi_[i] = softmax(beta_ * x_uncertain[i]);
  }
  
  // Calculate expected utilities for uncertain alternatives
  for (i in 1:total_uncertain_alts) {
    eta_uncertain_[i] = dot_product(to_vector(psi_[i]), upsilon_);
  }
  
  // Generate choices for uncertain problems (using alpha_)
  array[M] int<lower=1> y;
  {
    int pos = 1;
    for (m in 1:M) {
      vector[N_uncertain[m]] problem_eta = segment(eta_uncertain_, pos, N_uncertain[m]);
      vector[N_uncertain[m]] choice_probs = softmax(alpha_ * problem_eta);
      y[m] = categorical_rng(choice_probs);
      pos += N_uncertain[m];
    }
  }
  
  // === Generate risky choice data ===
  vector[total_risky_alts] eta_risky_;
  
  // Calculate expected utilities for risky alternatives
  for (i in 1:total_risky_alts) {
    eta_risky_[i] = dot_product(to_vector(x_risky[i]), upsilon_);
  }
  
  // Generate choices for risky problems (using omega_ = kappa_ * alpha_)
  array[N] int<lower=1> z;
  {
    int pos = 1;
    for (n in 1:N) {
      vector[N_risky[n]] problem_eta = segment(eta_risky_, pos, N_risky[n]);
      vector[N_risky[n]] choice_probs = softmax(omega_ * problem_eta);
      z[n] = categorical_rng(choice_probs);
      pos += N_risky[n];
    }
  }
}

parameters {
  real<lower=0> alpha;            // sensitivity for uncertain choices
  real<lower=0> kappa;            // association parameter
  matrix[K,D] beta;               // subjective probability parameters
  simplex[K-1] delta;             // utility differences
}

transformed parameters {
  // Derived sensitivity for risky choices
  real<lower=0> omega = kappa * alpha;
  
  // Construct ordered utilities
  vector[K] upsilon;
  upsilon[1] = 0;
  for (k in 2:K) {
    upsilon[k] = upsilon[k-1] + delta[k-1];
  }
}

model {
  // Priors (same as used to generate true values)
  alpha ~ lognormal(0, 1);
  kappa ~ lognormal(0, 0.5);
  to_vector(beta) ~ std_normal();    
  delta ~ dirichlet(rep_vector(1, K-1));
  
  // Likelihood
  {
    // === Uncertain choice likelihood ===
    array[total_uncertain_alts] vector[K] psi;
    vector[total_uncertain_alts] eta_uncertain;
    
    // Calculate subjective probabilities
    for (i in 1:total_uncertain_alts) {
      psi[i] = softmax(beta * x_uncertain[i]);
    }
    
    // Calculate expected utilities
    for (i in 1:total_uncertain_alts) {
      eta_uncertain[i] = dot_product(to_vector(psi[i]), upsilon);
    }
    
    // Likelihood for uncertain choices (using alpha)
    {
      int pos = 1;
      for (m in 1:M) {
        vector[N_uncertain[m]] problem_eta = segment(eta_uncertain, pos, N_uncertain[m]);
        vector[N_uncertain[m]] choice_probs = softmax(alpha * problem_eta);
        y[m] ~ categorical(choice_probs);
        pos += N_uncertain[m];
      }
    }
    
    // === Risky choice likelihood ===
    vector[total_risky_alts] eta_risky;
    
    // Calculate expected utilities for risky alternatives
    for (i in 1:total_risky_alts) {
      eta_risky[i] = dot_product(to_vector(x_risky[i]), upsilon);
    }
    
    // Likelihood for risky choices (using omega = kappa * alpha)
    {
      int pos = 1;
      for (n in 1:N) {
        vector[N_risky[n]] problem_eta = segment(eta_risky, pos, N_risky[n]);
        vector[N_risky[n]] choice_probs = softmax(omega * problem_eta);
        z[n] ~ categorical(choice_probs);
        pos += N_risky[n];
      }
    }
  }
}

generated quantities {
  // Copy the "true" data and parameters as required by sbc
  array[M] int y_ = y;
  array[N] int z_ = z;
  
  // Store true parameter values in a vector as required by sbc
  // Parameters: alpha, kappa, beta (K*D), delta (K-1)
  vector[2 + K*D + (K-1)] pars_;
  
  // Fill the vector with all parameters
  {
    int idx = 1;
    
    // alpha (1 parameter)
    pars_[idx] = alpha_;
    idx += 1;
    
    // kappa (1 parameter)
    pars_[idx] = kappa_;
    idx += 1;
    
    // beta (K*D parameters)
    for (k in 1:K) {
      for (d in 1:D) {
        pars_[idx] = beta_[k,d];
        idx += 1;
      }
    }
    
    // delta (K-1 parameters)
    for (k in 1:(K-1)) {
      pars_[idx] = delta_[k];
      idx += 1;
    }
  }
  
  // Calculate ranks as required by sbc (binary indicators)
  vector[2 + K*D + (K-1)] ranks_;
  {
    int idx = 1;
    
    // Alpha rank
    ranks_[idx] = (alpha > alpha_) ? 1 : 0;
    idx += 1;
    
    // Kappa rank
    ranks_[idx] = (kappa > kappa_) ? 1 : 0;
    idx += 1;
    
    // Beta ranks
    for (k in 1:K) {
      for (d in 1:D) {
        ranks_[idx] = (beta[k,d] > beta_[k,d]) ? 1 : 0;
        idx += 1;
      }
    }
    
    // Delta ranks
    for (k in 1:(K-1)) {
      ranks_[idx] = (delta[k] > delta_[k]) ? 1 : 0;
      idx += 1;
    }
  }
}
