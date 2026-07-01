/**
 * Bayesian Decision Theory Model (m_0_prior_sweep)
 *
 * Structurally identical to m_0 / m_01 / m_02.  The ONLY difference is that
 * the lognormal hyperparameters for the alpha prior are passed in as DATA
 * (alpha_prior_mu, alpha_prior_sigma) rather than hardcoded.  This lets a
 * single compiled program refit each application cell under alternative
 * alpha priors while holding the choice data and features fixed, for the
 * prior-sensitivity analysis of the 2x2 application (referee theme 3).
 *
 * Baseline priors reproduced by this program:
 *   insurance cells (K=3):  alpha_prior_mu=3.0, alpha_prior_sigma=0.75  (== m_01)
 *   Ellsberg  cells (K=4):  alpha_prior_mu=3.5, alpha_prior_sigma=0.75  (== m_02)
 *
 * The generated-quantities / PPC block of m_01/m_02 is omitted here because
 * the sweep only needs the alpha posterior draws.
 */
data{
  int<lower=1> M; // the number of decision problems
  int<lower=2> K; // the number of possible consequences
  int<lower=1> D; // the number of dimensions to describe an alternative
  int<lower=2> R; // the number of distinct alternatives
  array[R] vector[D] w; // the descriptions of each distinct alternative
  array[M,R] int<lower=0,upper=1> I; // indicator array: I[m,r] = 1 if alternative r is in problem m
  array[M] int<lower=1> y; // selected alternative (bounds checked in transformed data)
  real alpha_prior_mu;            // lognormal location for the alpha prior (passed as data)
  real<lower=0> alpha_prior_sigma; // lognormal scale for the alpha prior (passed as data)
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
  alpha ~ lognormal(alpha_prior_mu, alpha_prior_sigma); // data-driven alpha prior for the sweep
  to_vector(beta) ~ std_normal();                       // Prior on subjective probability parameters
  delta ~ dirichlet(rep_vector(1,K-1));                 // Prior ensures utilities are ordered increments on unit scale

  // Likelihood: categorical choice model
  for(i in 1:M){
    y[i] ~ categorical(chi[i]);
  }
}
