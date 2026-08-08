/**
 * Hierarchical Simulation Model, menu-size + PSEUDO-REPLICATION variant
 * (h_m01_size_pseudorep_sim)
 *
 * Study plan §8.5(a). The one addition over h_m01_size_sim is that observations
 * are grouped into MENUS, and repeat presentations of a menu are allowed to
 * agree more often than independent draws would.
 *
 * WHY THIS EXISTS
 * ---------------
 * h_m01_size_sim draws every one of the M_total rows independently. The real
 * design does not: it collects `num_presentations` = 2 presentations of each
 * menu (frozen at 2, §6.2), and the two presentations are the SAME items in
 * reversed order. At temperature 0 a near-deterministic model returns the same
 * item both times, so the second presentation carries almost no information.
 * Fitting the independence model to such data yields posteriors that are too
 * narrow, and a power analysis run on independent draws would therefore
 * OVERSTATE the design's power and set `num_problems` too low -- the exact
 * failure §8.5(a) says must not be discovered after the API budget is spent.
 *
 * MECHANISM
 * ---------
 * `rho_copy` is the probability that a repeat presentation reproduces the
 * menu's first presentation instead of drawing afresh:
 *
 *   rho_copy = 0  =>  identical to h_m01_size_sim (full independence)
 *   rho_copy = 1  =>  every repeat is a copy; effective N is n_menus, not M_total
 *
 * This is expressible so directly only because of how `y` is indexed. `y[m]` is
 * the rank of the chosen item among the menu's ASCENDING pool indices (see the
 * x_flat loop in h_m01.stan), NOT the chosen position. Two presentations of one
 * menu are a position permutation of the same item set, so they share the same
 * y encoding and "the same item was chosen" is exactly y_repeat = y_first.
 * Copying the POSITION would silently model something else.
 *
 * Note rho_copy is a property of the DATA-GENERATING regime, not a parameter to
 * be recovered: the inference model (h_m01_size) still assumes independence.
 * That mismatch is the thing being measured.
 *
 * `agreement_rate` is reported so the simulated regime can be calibrated
 * against the observed position-flip rate from a real smoke run (§8.8).
 */
data {
  // --- Dimensions ---
  int<lower=1> J;                         // number of cells
  int<lower=2> K;                         // number of consequences
  int<lower=1> D;                         // embedding dimensions
  int<lower=2> R;                         // number of distinct alternatives
  int<lower=1> P;                         // number of predictors

  // --- Shared alternatives ---
  array[R] vector[D] w;                   // feature vectors

  // --- Study design ---
  array[J] int<lower=1> M_per_cell;       // observations per cell
  matrix[J, P] X;                         // design matrix

  // --- Indicator arrays (stacked) ---
  int<lower=1> M_total;                   // sum(M_per_cell)
  array[M_total] int<lower=1,upper=J> cell;
  array[M_total, R] int<lower=0,upper=1> I;

  // --- Menu-size covariate (RQ6) ---
  vector[M_total] s;                      // centered menu size per observation

  // --- Pseudo-replication structure (§8.5(a)) ---
  int<lower=1> n_menus;                   // distinct menus
  array[M_total] int<lower=1> menu_id;    // which menu each observation presents
  real<lower=0,upper=1> rho_copy;         // P(a repeat presentation is a copy)

  // --- Hyperparameter controls ---
  real gamma0_mean;
  real<lower=0> gamma0_sd;
  real<lower=0> gamma_sd;
  real gamma_size_mean;
  real<lower=0> gamma_size_sd;
  real<lower=0> sigma_cell_sd;
  real<lower=0> beta_sd;
}

transformed data {
  array[M_total] int<lower=2> N_obs;
  int total_alts = 0;
  for (m in 1:M_total) {
    N_obs[m] = sum(I[m]);
    total_alts += N_obs[m];
  }

  {
    real mean_size = mean(to_vector(N_obs));
    if (abs(mean(s)) > 1e-6)
      reject("s must be centered (mean 0); got mean(s) = ", mean(s));
    for (m in 1:M_total) {
      if (abs(s[m] - (N_obs[m] - mean_size)) > 1e-6)
        reject("s[", m, "] = ", s[m], " but observation ", m, " has menu size ",
               N_obs[m], " and the mean menu size is ", mean_size);
    }
  }

  // First observation index of each menu, and a check that every presentation
  // of a menu really does carry the same item set.  If it did not, copying y
  // between presentations would be meaningless.
  array[n_menus] int first_obs = rep_array(0, n_menus);
  for (m in 1:M_total) {
    int g = menu_id[m];
    if (g > n_menus)
      reject("menu_id[", m, "] = ", g, " exceeds n_menus = ", n_menus);
    if (first_obs[g] == 0) {
      first_obs[g] = m;
    } else {
      int f = first_obs[g];
      if (cell[m] != cell[f])
        reject("menu ", g, " spans two cells (", cell[f], ", ", cell[m], ")");
      for (r in 1:R) {
        if (I[m, r] != I[f, r])
          reject("presentations of menu ", g, " differ in item set at r = ", r);
      }
    }
  }
  for (g in 1:n_menus) {
    if (first_obs[g] == 0)
      reject("menu ", g, " has no observations");
  }

  array[total_alts] vector[D] x_flat;
  {
    int pos = 1;
    for (m in 1:M_total) {
      for (r in 1:R) {
        if (I[m, r] == 1) {
          x_flat[pos] = w[r];
          pos += 1;
        }
      }
    }
  }
}

generated quantities {
  // Draw regression parameters
  real gamma0 = normal_rng(gamma0_mean, gamma0_sd);
  vector[P] gamma;
  for (p in 1:P) {
    gamma[p] = normal_rng(0, gamma_sd);
  }

  real gamma_size = (gamma_size_sd > 0)
                    ? normal_rng(gamma_size_mean, gamma_size_sd)
                    : gamma_size_mean;

  real<lower=0> sigma_cell = abs(normal_rng(0, sigma_cell_sd));

  vector[J] log_alpha_cell;
  vector[J] alpha_cell;
  for (j in 1:J) {
    real z_j = normal_rng(0, 1);
    log_alpha_cell[j] = gamma0 + X[j] * gamma + sigma_cell * z_j;
    alpha_cell[j] = exp(log_alpha_cell[j]);
  }

  array[J] matrix[K, D] beta;
  for (j in 1:J) {
    for (k in 1:K) {
      for (d in 1:D) {
        beta[j][k, d] = normal_rng(0, beta_sd);
      }
    }
  }

  simplex[K-1] delta = dirichlet_rng(rep_vector(1.0, K-1));
  vector[K] upsilon = cumulative_sum(append_row(0, delta));

  vector[M_total] alpha_obs;
  for (m in 1:M_total) {
    alpha_obs[m] = exp(log_alpha_cell[cell[m]] + gamma_size * s[m]);
  }

  array[M_total] int y;
  array[M_total] int<lower=0,upper=1> selected_seu_max;
  int<lower=0,upper=M_total> total_seu_max_selected;
  array[J] int seu_max_by_cell;
  array[M_total] int<lower=2> menu_size_out;
  // Share of repeat presentations that reproduced their menu's first
  // presentation.  Comparable to the observed position-stability rate (§8.8),
  // so a real smoke run can calibrate rho_copy instead of guessing it.
  real agreement_rate;
  int<lower=0> n_repeats;
  int<lower=0> n_agreements;

  {
    for (j in 1:J) {
      seu_max_by_cell[j] = 0;
    }
    n_repeats = 0;
    n_agreements = 0;

    int pos = 1;
    for (m in 1:M_total) {
      int j = cell[m];
      vector[N_obs[m]] problem_eta;
      for (idx in 1:N_obs[m]) {
        vector[K] psi_i = softmax(beta[j] * x_flat[pos]);
        problem_eta[idx] = dot_product(psi_i, upsilon);
        pos += 1;
      }

      // pos must advance for EVERY observation, so the draw/copy decision is
      // made only after problem_eta has been walked.
      int is_repeat = (m != first_obs[menu_id[m]]);
      int drawn = categorical_rng(softmax(alpha_obs[m] * problem_eta));
      if (is_repeat) {
        int first_y = y[first_obs[menu_id[m]]];
        int copied = (bernoulli_rng(rho_copy) == 1);
        y[m] = copied ? first_y : drawn;
        n_repeats += 1;
        if (y[m] == first_y) {
          n_agreements += 1;
        }
      } else {
        y[m] = drawn;
      }

      menu_size_out[m] = N_obs[m];

      real max_eta = max(problem_eta);
      if (abs(problem_eta[y[m]] - max_eta) < 1e-10) {
        selected_seu_max[m] = 1;
        seu_max_by_cell[j] += 1;
      } else {
        selected_seu_max[m] = 0;
      }
    }

    total_seu_max_selected = sum(selected_seu_max);
    agreement_rate = n_repeats > 0 ? (1.0 * n_agreements) / n_repeats : 0.0;
  }
}
