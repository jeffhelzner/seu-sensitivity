/**
 * Bayesian Decision Theory Model (m_01) for Simulation Based Calibration
 * 
 * Modifies m_0 by increasing concentration of parameter of Dirichlet prior on utility differences
 *
 * Following conventions required by rstan's sbc() function
 */
data {
  int<lower=1> M;                  // number of decision problems
  int<lower=2> K;                  // number of possible consequences
  int<lower=1> D;                  // dimensions to describe alternatives
  int<lower=2> R;                  // number of distinct alternatives
  array[R] vector[D] w;            // descriptions of each distinct alternative
  array[M,R] int<lower=0,upper=1> I; // indicator array: I[m,r] = 1 if alternative r is in problem m
}

transformed data {
  // Calculate N and total_alts
  array[M] int<lower=2> N;         // number of alternatives in each decision problem
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
  
  // Draw "true" parameters (with trailing underscore as required)
  real<lower=0> alpha_ = lognormal_rng(0, 1);
  
  matrix[K,D] beta_;
  for (k in 1:K) {
    for (d in 1:D) {
      beta_[k,d] = normal_rng(0, 1);
    }
  }
  
  simplex[K-1] delta_ = dirichlet_rng(rep_vector(5.0, K-1));
  
  // Construct utilities from delta_
  vector[K] upsilon_;
  upsilon_[1] = 0;
  for (k in 2:K) {
    upsilon_[k] = upsilon_[k-1] + delta_[k-1]; 
  }
  
  // Generate data using the "true" parameters
  array[total_alts] vector[K] psi_;
  vector[total_alts] eta_;
  
  // Calculate subjective probabilities
  for (i in 1:total_alts) {
    psi_[i] = softmax(beta_ * x[i]);
  }
  
  // Calculate expected utilities
  for (i in 1:total_alts) {
    eta_[i] = dot_product(to_vector(psi_[i]), upsilon_);
  }
  
  // Generate choices using "true" parameters
  array[M] int<lower=1> y;
  {
    int pos = 1;
    for (i in 1:M) {
      vector[N[i]] problem_eta = segment(eta_, pos, N[i]);
      vector[N[i]] choice_probs = softmax(alpha_ * problem_eta);
      
      // Generate choice from categorical distribution
      y[i] = categorical_rng(choice_probs);
      
      pos += N[i];
    }
  }
}

parameters {
  real<lower=0> alpha;            // sensitivity parameter
  matrix[K,D] beta;               // subjective probability parameters
  simplex[K-1] delta;             // utility differences
}

transformed parameters {
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
  to_vector(beta) ~ std_normal();    
  delta ~ dirichlet(rep_vector(5,K-1));
  
  // Likelihood
  {
    array[total_alts] vector[K] psi;  // Now just psi (no suffix needed)
    vector[total_alts] eta;           // Now just eta (no suffix needed)
    
    // Calculate subjective probabilities
    for (i in 1:total_alts) {
      psi[i] = softmax(beta * x[i]);
    }
    
    // Calculate expected utilities
    for (i in 1:total_alts) {
      eta[i] = dot_product(to_vector(psi[i]), upsilon);
    }
    
    // Likelihood calculation
    {
      int pos = 1;
      for (i in 1:M) {
        vector[N[i]] problem_eta = segment(eta, pos, N[i]);
        vector[N[i]] choice_probs = softmax(alpha * problem_eta);
        
        // Likelihood contribution
        y[i] ~ categorical(choice_probs);
        
        pos += N[i];
      }
    }
  }
}

generated quantities {
  // Copy the "true" data and parameters as required by sbc
  array[M] int y_ = y;
  
  // Store true parameter values in a vector as required by sbc
  // We need to flatten all parameters into a single vector
  vector[1 + K*D + (K-1)] pars_;
  
  // Fill the vector with all parameters
  {
    int idx = 1;
    
    // Add alpha
    pars_[idx] = alpha_;
    idx += 1;
    
    // Add beta elements (row-major)
    for (k in 1:K) {
      for (d in 1:D) {
        pars_[idx] = beta_[k,d];
        idx += 1;
      }
    }
    
    // Add delta elements
    for (k in 1:(K-1)) {
      pars_[idx] = delta_[k];
      idx += 1;
    }
  }
  
  // Calculate ranks as required by sbc (binary indicators)
  vector[1 + K*D + (K-1)] ranks_;
  {
    int idx = 1;
    
    // Alpha rank
    ranks_[idx] = (alpha > alpha_) ? 1 : 0;
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
  
  // Log likelihood for each observation (optional, but helpful)
  vector[M] log_lik;
  {
    array[total_alts] vector[K] psi;  // Now just psi (no suffix needed)
    vector[total_alts] eta;           // Now just eta (no suffix needed)
    
    // Calculate subjective probabilities
    for (i in 1:total_alts) {
      psi[i] = softmax(beta * x[i]);
    }
    
    // Calculate expected utilities
    for (i in 1:total_alts) {
      eta[i] = dot_product(to_vector(psi[i]), upsilon);
    }
    
    // Calculate log likelihood
    {
      int pos = 1;
      for (i in 1:M) {
        vector[N[i]] problem_eta = segment(eta, pos, N[i]);
        vector[N[i]] choice_probs = softmax(alpha * problem_eta);
        
        // Log likelihood for this observation
        log_lik[i] = categorical_lpmf(y[i] | choice_probs);
        
        pos += N[i];
      }
    }
  }
}
