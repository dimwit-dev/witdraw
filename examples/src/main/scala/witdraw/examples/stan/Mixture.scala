package witdraw.examples.stan

import dimwit.*
import dimwit.stats.*
import witdraw.*

/** Univariate Gaussian mixture with `K` components, ported from the Stan
  * reference model:
  *
  * {{{
  * data {
  *   int<lower=1> K;   int<lower=1> N;   real y[N];
  * }
  * parameters {
  *   simplex[K] theta;            // mixing proportions
  *   ordered[K] mu;               // component locations
  *   vector<lower=0>[K] sigma;    // component scales
  * }
  * model {
  *   vector[K] log_theta = log(theta);
  *   sigma ~ lognormal(0, 2);
  *   mu ~ normal(0, 10);
  *   for (n in 1:N) {
  *     vector[K] lps = log_theta;
  *     for (k in 1:K) lps[k] += normal_lpdf(y[n] | mu[k], sigma[k]);
  *     target += log_sum_exp(lps);
  *   }
  * }
  * }}}
  *
  * The per-point marginalisation `log_sum_exp_k (log theta_k + N(y_n | mu_k,
  * sigma_k))` is packaged as [[MixtureLikelihood]], a single distribution over
  * the length-`N` observation vector, so `y` enters only via
  * `observe(Y := y)`. Its `logProb` is the summed log-sum-exp; its `sample`
  * draws a component per point and emits from it.
  *
  * Stan has no explicit prior on `theta`, so it gets the implicit uniform
  * (`Dirichlet(1,...,1)`) prior over the simplex. `sigma ~ lognormal(0, 2)` is
  * a small [[LogNormal]] defined here (dimwit has no built-in). The `ordered`
  * constraint on `mu` is an identifiability device against label switching; it
  * is a constrained-parameterisation concern for the sampler, so at the model
  * level `mu` just carries its `normal(0, 10)` prior.
  */
object Mixture:

  trait Comp derives Label // k = 1..K mixture components
  trait Data derives Label // n = 1..N observations

  object Theta extends Site[Tensor1[Comp, Float32]] // mixing proportions (simplex)
  object Mu extends Site[Tensor1[Comp, Float32]]    // component locations
  object Sigma extends Site[Tensor1[Comp, Float32]] // component scales (> 0)
  object Y extends Site[Tensor1[Data, Float32]]     // observations

  type Posterior = (
      theta: Tensor1[Comp, Float32],
      mu: Tensor1[Comp, Float32],
      sigma: Tensor1[Comp, Float32]
  )

  /** `lognormal(muLog, sigmaLog)`: `log x ~ Normal(muLog, sigmaLog)`. */
  class LogNormal[T <: Tuple: Labels](
      muLog: Tensor[T, Float32],
      sigmaLog: Tensor[T, Float32]
  ) extends IndependentDistribution[T, Float32]:
    private val normal = Normal(muLog, sigmaLog)
    override def elementWiseLogProb(x: Tensor[T, Float32]): Tensor[T, LogProb] =
      LogProb(normal.elementWiseLogProb(x.log).asFloat - x.log)
    override def sample(key: Key): Tensor[T, Float32] = normal.sample(key).exp

  /** log-sum-exp over `axis` of a matrix, reducing it away. */
  private def logsumexp[L: Label, K: Label](
      t: Tensor2[L, K, Float32],
      axis: Axis[L]
  ): Tensor1[K, Float32] =
    val m = t.max(axis)
    (t -! m).exp.sum(axis).log + m

  /** The marginalised mixture likelihood over a length-`n` observation vector.
    *
    *   - `logProb(y)` = `sum_n log_sum_exp_k ( log theta_k + N(y_n | mu_k,
    *     sigma_k) )`.
    *   - `sample` draws a component `z_n ~ Categorical(theta)` per point and
    *     emits `y_n ~ Normal(mu[z_n], sigma[z_n])`.
    *
    * @param n
    *   number of points to draw — only used by `sample`; `logProb` reads the
    *   length off `y`. `K` is read off `mu`.
    */
  class MixtureLikelihood(
      theta: Tensor1[Comp, Float32],
      mu: Tensor1[Comp, Float32],
      sigma: Tensor1[Comp, Float32],
      n: Int
  ) extends MultivariateDistribution[Data, Float32]:

    private val k = mu.shape(Axis[Comp])

    override def logProb(y: Tensor1[Data, Float32]): Tensor0[LogProb] =
      val nD = y.shape(Axis[Data])

      // emission log-densities  logEmit[k, n] = normal_lpdf(y[n] | mu[k], sigma[k])
      val muGrid: Tensor2[Comp, Data, Float32] =
        stack(Seq.fill(nD)(mu), Axis[Data]).transpose
      val sigmaGrid: Tensor2[Comp, Data, Float32] =
        stack(Seq.fill(nD)(sigma), Axis[Data]).transpose
      val yGrid: Tensor2[Comp, Data, Float32] =
        stack(Seq.fill(k)(y), Axis[Comp])
      val logEmit: Tensor2[Comp, Data, Float32] =
        Normal(muGrid, sigmaGrid).elementWiseLogProb(yGrid).asFloat

      // lps[k, n] = log(theta[k]) + logEmit[k, n]; sum over n of log_sum_exp_k
      val lps = logEmit +! theta.log
      LogProb(logsumexp(lps, Axis[Comp]).sum)

    override def sample(key: Key): Tensor1[Data, Float32] =
      val ys = key.split(n).map { kk =>
        val (kz, ky) = kk.split2()
        val z = Categorical(Prob(theta)).sample(kz)
        Normal(mu.slice(Axis[Comp].at(z)), sigma.slice(Axis[Comp].at(z)))
          .sample(ky)
      }
      stack(ys, Axis[Data])

  // -- the model --------------------------------------------------------

  private def samplePriors(k: Int)(using ProbContext): Posterior =
    val theta = Theta ~ Dirichlet(Tensor1(Axis[Comp] -> k).fill(1f))
    val mu = Mu ~ Normal(
      Tensor1(Axis[Comp] -> k).fill(0f),
      Tensor1(Axis[Comp] -> k).fill(10f)
    )
    val sigma = Sigma ~ LogNormal(
      Tensor1(Axis[Comp] -> k).fill(0f),
      Tensor1(Axis[Comp] -> k).fill(2f)
    )
    (theta = theta, mu = mu, sigma = sigma)

  /** The model over `k` components and `n` observations. Forward-sampling
    * draws a fresh mixture sample; observing `Y` scores the marginal.
    */
  def model(k: Int, n: Int)(using ProbContext) =
    val p = samplePriors(k)
    val y = Y ~ MixtureLikelihood(p.theta, p.mu, p.sigma, n)
    (theta = p.theta, mu = p.mu, sigma = p.sigma, y = y)

  /** The model conditioned on an observed sample (`k` = number of
    * components).
    */
  def observed(y: Tensor1[Data, Float32], k: Int)(using ProbContext) =
    observe(Y := y)(model(k, y.shape(Axis[Data])))

  def logDensity(y: Tensor1[Data, Float32])(
      params: Posterior
  ): Tensor0[Float32] =
    ModelScorer.run:
      observeAll(params)(observed(y, params.mu.shape(Axis[Comp])))

  /** Draw parameters from their priors and simulate `n` points from a
    * `k`-component mixture.
    */
  def generate(k: Int, n: Int)(
      key: Key
  ): (params: Posterior, path: Tensor1[Data, Float32]) =
    val r = ForwardSampler.run(key)(model(k, n))
    (
      params = (theta = r.theta, mu = r.mu, sigma = r.sigma),
      path = r.y
    )

  /** Simulate `n` points for fixed parameters. */
  def simulate(params: Posterior, n: Int)(key: Key): Tensor1[Data, Float32] =
    MixtureLikelihood(params.theta, params.mu, params.sigma, n).sample(key)

  // -- runnable example ----------------------------------------------

  @main
  def mixtureExample(): Unit =
    dimwit.initialize()

    // A well-separated two-component mixture (mu ordered, as Stan requires).
    val params: Posterior = (
      theta = Tensor1(Axis[Comp] -> 2).fromArray(Array(0.7f, 0.3f)),
      mu = Tensor1(Axis[Comp] -> 2).fromArray(Array(-3f, 3f)),
      sigma = Tensor1(Axis[Comp] -> 2).fromArray(Array(0.5f, 1.0f))
    )

    val y = simulate(params, n = 80)(Key(1L))
    println(s"simulated sample:\n$y")
    println(
      f"joint logDensity (marginal + priors) = ${logDensity(y)(params).item}%.3f"
    )
