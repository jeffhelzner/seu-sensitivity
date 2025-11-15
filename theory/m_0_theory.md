# Theoretical Foundation for Model m_0: Sensitivity and Value Maximization

## 1. Introduction

This document establishes the theoretical foundations for our computational model of epistemic agents. We first prove three fundamental properties of the softmax choice model with respect to an arbitrary value function, then show how these properties apply to our specific case where values are subjective expected utilities (SEU). This approach clarifies that the core choice-theoretic results are independent of how values are constructed, while the SEU interpretation provides the substantive behavioral content.

## 2. General Softmax Choice Model

### 2.1 Notation and Definitions

Let:
- **A** = {1, 2, ..., K} be a finite set of alternatives
- **V: A → ℝ** be an arbitrary value function assigning real-valued utilities to alternatives
- **V(j)** ∈ ℝ denote the value of alternative j
- **α** ∈ ℝ₊ denote the sensitivity parameter

### 2.2 Softmax Choice Rule

The probability that a decision maker selects alternative k ∈ **A** is given by:

```
P(choose k | α, V) = exp(α · V(k)) / Σⱼ∈A exp(α · V(j))
```

This is the Luce choice rule or softmax function.

### 2.3 Optimal Alternatives

Define the set of value-maximizing alternatives:

```
A* = {j ∈ A : V(j) ≥ V(k) for all k ∈ A}
```

Let V* = max{V(j) : j ∈ A} denote the maximum value, and A⁻ = A \ A* denote the set of suboptimal alternatives.

## 3. Fundamental Properties of Softmax Choice

**These properties hold for ANY value function V: A → ℝ.**

### Property 1: Monotonicity in Sensitivity

**Statement:** For any value function V: A → ℝ, holding V fixed:
- For k ∈ A* (value-maximizing), P(choose k | α, V) is strictly increasing in α
- For j ∉ A* (suboptimal), P(choose j | α, V) is strictly decreasing in α

**Proof:**

*Part A: Value-maximizing alternatives (k ∈ A*)*

Let k ∈ A* such that V(k) = V*. Taking the derivative with respect to α:

```
∂P(k)/∂α = ∂/∂α [exp(α·V(k)) / Z(α)]
```

where Z(α) = Σⱼ∈A exp(α·V(j)) is the partition function.

Using the quotient rule:

```
∂P(k)/∂α = [Z(α)·V(k)·exp(α·V(k)) - exp(α·V(k))·Z'(α)] / Z(α)²
         = P(k)·[V(k) - Z'(α)/Z(α)]
```

Computing Z'(α):

```
Z'(α) = Σⱼ∈A V(j)·exp(α·V(j))
```

Therefore:

```
Z'(α)/Z(α) = Σⱼ∈A V(j)·P(j) = 𝔼[V]
```

where 𝔼[V] is the expected value under the current choice distribution.

Thus:

```
∂P(k)/∂α = P(k)·[V(k) - 𝔼[V]] = P(k)·[V* - 𝔼[V]]
```

Since V* = max{V(j)} and 𝔼[V] is a weighted average:

```
𝔼[V] = Σⱼ∈A P(j)·V(j) ≤ V*
```

with equality only when P(k) = 1 for some k ∈ A* (which occurs only as α → ∞).

For any finite α, we have 𝔼[V] < V*, so:

```
∂P(k)/∂α = P(k)·[V* - 𝔼[V]] > 0
```

*Part B: Suboptimal alternatives (j ∉ A*)*

For j ∉ A*, we have V(j) < V*. Following the same derivation:

```
∂P(j)/∂α = P(j)·[V(j) - 𝔼[V]]
```

Since j is suboptimal and A* is non-empty (P(A*) > 0 for all finite α):

```
𝔼[V] ≥ P(A*)·V* + P(j)·V(j)
     > P(A*)·V(j) + P(j)·V(j)    [since V* > V(j)]
     = V(j)
```

Therefore, V(j) - 𝔼[V] < 0, and:

```
∂P(j)/∂α = P(j)·[V(j) - 𝔼[V]] < 0
```

□

### Property 2: Perfect Optimization in the Limit (α → ∞)

**Statement:** For any value function V: A → ℝ, as α → ∞:

```
lim_{α→∞} P(choose k | α, V) = {
    1/|A*|  if k ∈ A*
    0       if k ∉ A*
}
```

**Proof:**

*Case 1: k ∈ A* (value-maximizing)*

```
P(k) = exp(α·V*) / [|A*|·exp(α·V*) + Σⱼ∈A⁻ exp(α·V(j))]
```

Dividing by exp(α·V*):

```
P(k) = 1 / [|A*| + Σⱼ∈A⁻ exp(α·[V(j) - V*])]
```

For j ∈ A⁻, we have V(j) < V*, so V(j) - V* < 0.

As α → ∞:

```
exp(α·[V(j) - V*]) → 0  for all j ∈ A⁻
```

Thus:

```
lim_{α→∞} P(k) = 1/|A*|
```

*Case 2: j ∉ A* (suboptimal)*

```
P(j) = exp(α·V(j)) / [Σₘ∈A* exp(α·V*) + Σₙ∈A⁻ exp(α·V(n))]
```

Dividing by exp(α·V*):

```
P(j) = exp(α·[V(j) - V*]) / [|A*| + Σₙ∈A⁻ exp(α·[V(n) - V*])]
```

Since V(j) - V* < 0:
- Numerator → 0
- Denominator ≥ |A*| > 0

Therefore:

```
lim_{α→∞} P(j) = 0
```

□

### Property 3: Uniform Choice in the Limit (α → 0)

**Statement:** For any value function V: A → ℝ, as α → 0:

```
lim_{α→0} P(choose k | α, V) = 1/|A|  for all k ∈ A
```

**Proof:**

Using Taylor expansion exp(x) = 1 + x + O(x²):

```
P(k) = [1 + α·V(k) + O(α²)] / [Σⱼ∈A (1 + α·V(j) + O(α²))]
     = [1 + α·V(k) + O(α²)] / [|A| + α·Σⱼ V(j) + O(α²)]
```

As α → 0:

```
lim_{α→0} P(k) = 1/|A|
```

**Alternative proof via logarithms:**

```
log P(k) = α·V(k) - log[Σⱼ∈A exp(α·V(j))]
```

Expanding the log-sum-exp:

```
log[Σⱼ∈A exp(α·V(j))] = log[|A| + α·Σⱼ V(j) + O(α²)]
                        = log|A| + (α·Σⱼ V(j))/|A| + O(α²)
```

Therefore:

```
log P(k) = α·V(k) - log|A| - (α·Σⱼ V(j))/|A| + O(α²)
         = -log|A| + α·[V(k) - (Σⱼ V(j))/|A|] + O(α²)
         → -log|A|  as α → 0
```

Thus:

```
lim_{α→0} P(k) = 1/|A|
```

□

## 4. Application to Subjective Expected Utility

### 4.1 SEU as a Value Function

We now specialize to the case where the value function V is constructed as subjective expected utility:

Let:
- **Ω** = {ω₁, ω₂, ..., ωₙ} be a finite outcome space
- **υⱼ(ωᵢ)** ∈ ℝ denote the utility of outcome ωᵢ under alternative j
- **ψⱼ(ωᵢ)** ∈ [0,1] denote the subjective probability of outcome ωᵢ given alternative j, where Σᵢ ψⱼ(ωᵢ) = 1

Define the subjective expected utility function:

```
SEU: A → ℝ
SEU(j) = Σᵢ ψⱼ(ωᵢ)·υⱼ(ωᵢ)
```

**Key observation:** SEU is simply a particular choice of value function V = SEU. Therefore, all three properties proved above apply immediately when we set V(j) = SEU(j).

### 4.2 SEU Maximization Properties

By substituting V = SEU into Properties 1-3, we obtain:

**Corollary 1 (Monotonicity for SEU):** Holding υ and ψ fixed, higher sensitivity α increases the probability of choosing alternatives that maximize SEU.

**Corollary 2 (Perfect Rationality):** As α → ∞, the decision maker chooses SEU-maximizing alternatives with probability 1.

**Corollary 3 (Random Choice):** As α → 0, the decision maker chooses uniformly at random, independent of SEU values.

### 4.3 What SEU Adds

While the mathematical properties of softmax choice hold for any value function, the SEU construction provides:

1. **Interpretability:** Values decompose into beliefs (ψ) and utilities (υ), allowing separate analysis of epistemic and preference components

2. **Normative content:** SEU maximization is a rationality criterion - Properties 1-3 characterize adherence to this normative standard

3. **Empirical predictions:** The model predicts that choices will track SEU, not other potential value functions, providing testable restrictions

4. **Parameter identification:** With sufficient choice data and variation in alternatives, we can potentially identify ψ and υ separately (not just their product)

### 4.4 Scale Invariance and Identification of Sensitivity

A fundamental property of utility functions in decision theory is that they are unique only up to positive affine transformations. This raises a critical question: how can we meaningfully identify and interpret the sensitivity parameter α?

**Theorem (Scale Invariance):** Let υ be a utility function and define a rescaled utility function:

```
υ̃(ω) = a·υ(ω) + b  where a > 0
```

Then for any alternative j:

```
SEU_υ̃(j) = a·SEU_υ(j) + b
```

**Proof:**

```
SEU_υ̃(j) = Σᵢ ψⱼ(ωᵢ)·υ̃(ωᵢ)
          = Σᵢ ψⱼ(ωᵢ)·[a·υ(ωᵢ) + b]
          = a·Σᵢ ψⱼ(ωᵢ)·υ(ωᵢ) + b·Σᵢ ψⱼ(ωᵢ)
          = a·SEU_υ(j) + b
```

**Invariance of Choice Probabilities:** Under softmax choice, this transformation leaves probabilities unchanged:

```
P(j | α, υ̃) = exp(α·SEU_υ̃(j)) / Σₖ exp(α·SEU_υ̃(k))
             = exp(α·[a·SEU_υ(j) + b]) / Σₖ exp(α·[a·SEU_υ(k) + b])
             = exp(α·a·SEU_υ(j))·exp(α·b) / [Σₖ exp(α·a·SEU_υ(k))·exp(α·b)]
             = exp(α·a·SEU_υ(j)) / Σₖ exp(α·a·SEU_υ(k))
             = P(j | α·a, υ)
```

**Key Implication:** The pair (α, υ) and (α·a, υ̃) generate identical choice probabilities for any a > 0. This means α and the scale of utility are not separately identified from choice data alone.

### 4.5 Resolving the Identification Problem

To make α interpretable as "sensitivity to subjective expected utility," we must fix the scale of utility. Model m_0 achieves this through normalization:

**Normalization Constraint:** We constrain utilities to lie in [0,1]:

```
υ₁ = 0  and  υₖ = 1
```

This is implemented in m_0 via:

```
υ = cumulative_sum([0, δ])  where δ ~ Dirichlet(1,...,1)
```

ensuring 0 = υ₁ ≤ υ₂ ≤ ... ≤ υₖ = 1.

**Identification Result:** Given this normalization, α is identified from choice data as the unique parameter governing sensitivity to differences in subjective expected utility measured on the [0,1] scale.

**Formal Statement:** Fix the utility scale by setting min(υ) = 0 and max(υ) = 1. Then:

1. The likelihood function P(y | α, ψ, υ) uniquely determines α
2. Different values of α yield different choice distributions
3. α has a clear interpretation: it measures sensitivity to expected utility differences on the unit scale

**Proof of Identification:** Under the normalization υ ∈ [0,1]:

- The range of possible SEU values is bounded: SEU(j) ∈ [0,1] for all j
- The maximum difference in SEU between any two alternatives is bounded: |SEU(j) - SEU(k)| ≤ 1
- Therefore, α directly controls the log-odds ratio between alternatives:

```
log[P(j)/P(k)] = α·[SEU(j) - SEU(k)]
```

where SEU differences are measured in standardized units.

Since log-odds ratios are directly observable in choice data (via choice frequencies), and SEU differences are determined by (ψ, υ), the parameter α is identified.

### 4.6 Interpretation of α Under Normalization

With utilities normalized to [0,1], α has a precise interpretation:

**α = 1:** A one-unit difference in SEU (the maximum possible difference) produces a log-odds ratio of 1, corresponding to:

```
P(better)/P(worse) = e ≈ 2.72
```

The better alternative is chosen with probability ≈ 73%.

**α = 2:** A one-unit SEU difference produces log-odds of 2:

```
P(better)/P(worse) = e² ≈ 7.39
```

The better alternative is chosen with probability ≈ 88%.

**α = 5:** A one-unit SEU difference produces log-odds of 5:

```
P(better)/P(worse) = e⁵ ≈ 148
```

The better alternative is chosen with probability ≈ 99%.

**General interpretation:** α measures the log-odds change per unit of standardized SEU difference. Higher α means choices become more deterministically aligned with SEU rankings.

### 4.7 Why This Matters for Model m_0

The normalization and identification results ensure that:

1. **Posterior inferences about α are meaningful:** When we infer α ≈ 3 from data, this means the decision maker's log-odds of choosing between alternatives changes by approximately 3 for each unit difference in normalized SEU.

2. **Cross-study comparability:** Two studies using the same normalization can meaningfully compare estimated α values - they measure sensitivity on the same scale.

3. **Prior specification is interpretable:** When we set `alpha ~ lognormal(0, 1)`, we're placing prior mass on interpretable sensitivity levels relative to the unit scale.

4. **Model predictions are identifiable:** The model makes sharp predictions about choice probabilities given (ψ, υ, α), and these parameters can be separately estimated from sufficiently rich choice data.

**Without normalization:** We could only identify the product α·a where a is the unknown utility scale. We couldn't separately interpret "sensitivity" from "utility scale."

**With normalization:** We fix a = 1/(max υ - min υ), making α interpretable as sensitivity per unit of standardized SEU difference.

## 5. Model m_0 Specification

### 5.1 Constructing SEU from Features

In model m_0, we parameterize the components of SEU:

**Subjective probabilities** are determined by alternative features x through:

```
ψⱼ = softmax(β · xⱼ)
```

where β ∈ ℝ^(K×D) maps D-dimensional features to K outcome probabilities.

**Utilities** are ordered with incremental differences:

```
υ = cumulative_sum([0, δ])
```

where δ is a (K-1)-simplex ensuring utilities lie in [0,1] and are strictly ordered.

**Subjective expected utility** is then:

```
SEU(j) = Σₖ ψⱼₖ · υₖ = ψⱼᵀυ
```

**Choice probabilities** follow:

```
P(choose j | α, β, δ, x) = exp(α · SEU(j)) / Σₖ exp(α · SEU(k))
```

### 5.2 Theoretical Guarantees

Properties 1-3 ensure that:

1. Posterior inference on α has a clear interpretation: higher inferred α means choices are more consistent with SEU maximization

2. The model nests both deterministic SEU maximization (α → ∞) and random choice (α → 0) as limiting cases

3. Intermediate values of α capture bounded rationality where decision makers are sensitive to SEU differences but make probabilistic choices

### 5.3 SEU Maximizer Selection

An important diagnostic for understanding model behavior is tracking whether agents select SEU-maximizing alternatives. For each decision problem m, we can define:

**SEU Maximizer Indicator:**
```
I_m = 1 if chosen alternative j* satisfies η(j*) = max_j η(j)
     0 otherwise
```

where η(j) is the expected utility of alternative j.

**Expected SEU Maximizer Selection:** Under the softmax choice model with sensitivity α, the probability of selecting an SEU maximizer for problem m is:

```
P(select SEU max | m, α) = Σ_{j ∈ A*_m} exp(α·η(j)) / Σ_{k=1}^{N_m} exp(α·η(k))
```

where A*_m is the set of SEU-maximizing alternatives in problem m.

**Theoretical Properties:**

1. **As α → ∞:** P(select SEU max | m, α) → 1 for all m
2. **As α → 0:** P(select SEU max | m, α) → |A*_m|/N_m (probability under random choice)
3. **Monotonicity:** P(select SEU max | m, α) is strictly increasing in α

**Aggregate Analysis:** The total number of SEU maximizers selected across M problems follows:

```
T = Σ_{m=1}^M I_m
```

Under prior predictive analysis, T provides a summary measure of how often the model generates "rational" choices given the prior distributions on parameters.

## 6. Implications for Rational Choice Theory

### 6.1 Generality of Results

The fact that Properties 1-3 hold for *any* value function V reveals an important insight: these properties characterize the softmax choice rule itself, not the specific theory of value.

This means:
- The monotonicity, limiting behavior, and convergence properties are **structural features** of softmax choice
- They would hold equally for risk-neutral expected value, prospect theory values, or any other value construction
- The choice of SEU as our value function is a **substantive theoretical commitment** about what drives behavior

### 6.2 SEU as a Rational Standard

By choosing V = SEU, we commit to SEU maximization as our rationality criterion. This commitment:

1. Aligns with classical Bayesian decision theory (Savage, 1954)
2. Provides a normative benchmark for evaluating choice behavior
3. Makes our parameter α interpretable as "degree of rationality" relative to this specific standard

### 6.3 Alternative Value Functions

Our framework could accommodate other value functions:
- **Expected value:** V(j) = Σᵢ ψⱼ(ωᵢ)·ωᵢ (objective outcomes, no utilities)
- **Prospect theory:** V(j) = Σᵢ w(ψⱼ(ωᵢ))·v(υⱼ(ωᵢ)) (probability weighting, reference dependence)
- **Regret theory:** V(j) = f(υⱼ, max_k υₖ) (comparative evaluation)

Each would satisfy Properties 1-3, but yield different substantive predictions about choice behavior.

## 7. Technical Notes

### 7.1 Uniqueness of Maximum

When |A*| = 1 (unique maximum), Property 2 shows deterministic optimal choice as α → ∞.

When |A*| > 1 (multiple optima), the limiting distribution is uniform over A*, representing rational indifference between equally valued alternatives.

### 7.2 Rate of Convergence

- **Property 2 (α → ∞):** Convergence is exponential with rate Δ = min{V* - V(j) : j ∉ A*}
- **Property 3 (α → 0):** Convergence is polynomial (first-order in α)

### 7.3 Numerical Implementation

For computational stability:
- Large α: Use log-sum-exp trick: log(Σⱼ exp(xⱼ)) = max(x) + log(Σⱼ exp(xⱼ - max(x)))
- Small α: Taylor expansion may provide better accuracy than direct evaluation

### 7.4 Connection to Information Theory

The softmax choice model can be derived as the maximum entropy distribution subject to the constraint 𝔼[V] = c, revealing deep connections to information theory and statistical mechanics.

## 8. References

**Softmax/Luce choice:**
- Luce, R. D. (1959). *Individual Choice Behavior: A Theoretical Analysis*
- McFadden, D. (1973). Conditional logit analysis of qualitative choice behavior

**Quantal response:**
- McKelvey, R. D., & Palfrey, T. R. (1995). Quantal response equilibria for normal form games

**Subjective expected utility:**
- Savage, L. J. (1954). *The Foundations of Statistics*
- Anscombe, F. J., & Aumann, R. J. (1963). A definition of subjective probability

**Information theory connection:**
- Jaynes, E. T. (1957). Information theory and statistical mechanics
- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*
