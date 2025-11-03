/**
 * Bayesian Decision Theory Model (m_0)
 * 
 * This model implements an expected utility framework where:
 * - Agents have subjective probabilities over K possible consequences for each alternative
 * - These probabilities are determined by features of the alternative through a softmax transformation
 * - Utilities are ordered and parameterized as incremental differences on a unit scale
 * - Choices follow a softmax distribution of expected utilities based on the agent's sensitivity to their expected utilities
 */
data{
  int<lower=1> M; // the number of decision problems
  int<lower=2> K; // the number of possible consequences
  int<lower=1> D; // the number of dimensions to describe an alternative
  int<lower=2> R; // the number of distinct alternatives
  array[R] vector[D] w; // the descriptions of each distinct alternative
  array[M,R] int<lower=0,upper=1> I; // indicator array: I[m,r] = 1 if alternative r is in problem m
  array[M] int<lower=1> y; // selected alternative (bounds checked in transformed data)
}
transformed data {
  // Calculate N and max_N
  array[M] int<lower=2> N; // the number of alternatives in each decision problem
  int total_alternatives = 0;
  
  for (m in 1:M) {
    N[m] = sum(I[m]);
    total_alternatives += N[m];
    
    // Validate y is within bounds
    if (y[m] > N[m]) reject("y[", m, "] = ", y[m], " must be <= N[", m, "] = ", N[m]);
  }
  
  // Construct x from w and I
  array[total_alternatives] vector[D] x;
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
parameters{
  real<lower=0> alpha; // sensitivity parameter (decision noise); strictly positive
  matrix[K,D] beta; // coefficients mapping alternative descriptions to subjective probabilities
  simplex[K - 1] delta; // utility differences on unit scale (ensures utilities are ordered)
}
transformed parameters{
  array[sum(N)] simplex[K] psi; // subjective probability over possible outcomes for each alternative
  ordered[K] upsilon; // the subjective utility of each alternative
  vector[sum(N)] eta; // expected utility of each alternative on unit scale
  array[M] simplex[max(N)] chi; // choice probs with padding
  
  // Calculate subjective probabilities for each alternative using a softmax transformation
  for(i in 1:sum(N)){
    psi[i] = softmax(beta*x[i]);
  }
  
  // Construct ordered utilities - more efficient direct calculation
  upsilon = cumulative_sum(append_row(0, delta));
  
  // Calculate expected utility for each alternative
  for(i in 1:sum(N)){
    eta[i] = dot_product(psi[i],upsilon);
  }
  
  // Construct choice probabilities with padding
  {
    int pos = 1;
    for(i in 1:M){
      // Extract relevant expected utilities for this decision problem
      vector[N[i]] problem_eta = segment(eta, pos, N[i]);
      
      // Calculate choice probabilities and add padding
      chi[i] = append_row(
        softmax(alpha * problem_eta),
        rep_vector(0, max(N) - N[i])
      );
      
      pos += N[i];
    }
  }
}
model{
  // Priors
  alpha ~ lognormal(0, 1);          // Prior on choice sensitivity (now lognormal)
  to_vector(beta) ~ std_normal();    // Prior on subjective probability parameters
  delta ~ dirichlet(rep_vector(1,K-1)); // Prior ensures utilities are ordered increments on unit scale
  
  // Likelihood: categorical choice model
  for(i in 1:M){
    y[i] ~ categorical(chi[i]);
  }
}
generated quantities {
  // Log-likelihood for model comparison
  vector[M] log_lik;
  for (i in 1:M) {
    log_lik[i] = categorical_lpmf(y[i] | chi[i]);
  }
  
  // Posterior predictive checks
  array[M] int y_pred;
  for (i in 1:M) {
    y_pred[i] = categorical_rng(chi[i]);
  }
}

