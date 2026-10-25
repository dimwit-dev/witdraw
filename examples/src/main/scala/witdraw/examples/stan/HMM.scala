package witdraw.examples.stan

import dimwit.*
import dimwit.stats.*
import witdraw.*
import witdraw.inference.NUTS

/** Three-state Hidden Markov Model with Gaussian emissions, ported from the
  * Stan reference model:
  *
  * {{{
  * data {
  *   int N;
  *   array[N] real y;
  * }
  * parameters {
  *   array[3] simplex[3] gamma_arr;  // rows of the transition matrix
  *   simplex[3] rho;                 // initial state distribution
  *   vector[3] mu;                   // emission means
  *   real<lower=0> sigma;            // shared emission sd
  * }
  * transformed parameters {
  *   matrix[3, 3] gamma;
  *   for (k in 1:3) gamma[k, ] = to_row_vector(gamma_arr[k]);
  *   matrix[3, N] log_omega;
  *   for (n in 1:N) for (i in 1:3)
  *     log_omega[i, n] = normal_lpdf(y[n] | mu[i], sigma);
  * }
  * model {
  *   mu ~ normal(0, 1);
  *   sigma ~ normal(0, 1);
  *   // uniform (Dirichlet(1,1,1)) prior on each simplex, implicitly
  *   target += hmm_marginal(log_omega, gamma, rho);
  * }
  * }}}
  *
  * `hmm_marginal` — the forward algorithm that marginalises out the latent
  * state path — is packaged here as [[HMMMarginal]], a single distribution over
  * the length-`N` observation sequence. Its `logProb` is the forward recursion
  * in log space; its `sample` is the generative HMM (draw a state from `rho`,
  * emit, transition, repeat). So `y` enters the model only through
  * `observe(Y := y)`.
  *
  * `gamma_arr` / `gamma`: row-stochastic transition matrix,
  * `gamma[i, j] = P(state n+1 = j | state n = i)`. `rho`: distribution of the
  * first state. Stan leaves `sigma` on an implicit half-normal (`normal(0,1)`
  * truncated at 0); we use [[HalfNormal]] for it, and `Dirichlet(1,1,1)` for
  * the simplexes (Stan's implicit uniform-over-the-simplex prior).
  */
object HMM:

  trait Time derives Label // n = 1..N
  trait State derives Label // the 3 hidden states

  private val nStates = 3

  object Mu extends Site[Tensor1[State, Float32]] // emission means
  object Sigma extends Site[Tensor0[Float32]] // shared emission sd (> 0)
  object Rho extends Site[Tensor1[State, Float32]] // initial-state simplex
  object Gamma
      extends Site[
        Tensor2[State, Prime[State], Float32]
      ] // transition matrix, rows simplex
  object Y extends Site[Tensor1[Time, Float32]] // observed sequence

  /** Values of the four parameters, e.g. the true values to simulate from. */
  type HMMParams = (
      mu: Tensor1[State, Float32],
      sigma: Tensor0[Float32],
      rho: Tensor1[State, Float32],
      gamma: Tensor2[State, Prime[State], Float32]
  )

  // -- numerics ---------------------------------------------------------

  private def logsumexp[L: Label](t: Tensor1[L, Float32]): Tensor0[Float32] =
    val m = t.max
    (t -! m).exp.sum.log + m

  /** log-sum-exp over `axis` of a matrix, reducing it away. */
  private def logsumexp[L: Label, K: Label](
      t: Tensor2[L, K, Float32],
      axis: Axis[L]
  ): Tensor1[K, Float32] =
    val m = t.max(axis)
    (t -! m).exp.sum(axis).log + m

  // -- the hmm_marginal distribution ----------------------------------

  /** The HMM observation model over a length-`n` sequence.
    *
    *   - `logProb(y)` = `hmm_marginal(log_omega, gamma, rho)`: the forward
    *     algorithm, marginalising the latent path.
    *   - `sample` draws a state path (`rho`, then `gamma`) and emits
    *     `Normal(mu[state], sigma)` at each step.
    *
    * @param n
    *   sequence length — only used by `sample`; `logProb` reads it off `y`.
    */
  class HMMMarginal(
      mu: Tensor1[State, Float32],
      sigma: Tensor0[Float32],
      rho: Tensor1[State, Float32],
      gamma: Tensor2[State, Prime[State], Float32],
      n: Int
  ) extends MultivariateDistribution[Time, Float32]:

    override def logProb(y: Tensor1[Time, Float32]): Tensor0[LogProb] =
      val nT = y.shape(Axis[Time])

      // log_omega[i, n] = normal_lpdf(y[n] | mu[i], sigma)
      val grid = Shape(Axis[State] -> mu.shape(Axis[State]), Axis[Time] -> nT)
      val logOmega: Tensor2[State, Time, Float32] =
        Normal
          .isotropic(mu.broadcastTo(grid), sigma)
          .elementWiseLogProb(y.broadcastTo(grid))
          .asFloat

      // forward recursion, all in log space: a loop over the time steps after
      // the first, compiled once however long the sequence is
      val logGamma: Tensor2[State, Prime[State], Float32] = gamma.log
      val logAlpha0: Tensor1[State, Float32] =
        rho.log + logOmega.select(Axis[Time].at(0))
      val (logAlpha, _) =
        scan(Axis[Time])(logAlpha0, logOmega.select(Axis[Time].at(1 until nT))):
          (logAlpha, logOmegaT) =>
            // logPred[j] = logsumexp_i ( logAlpha[i] + logGamma[i, j] )
            val logPred: Tensor1[State, Float32] =
              logsumexp(logGamma +! logAlpha, Axis[State]).relabelTo(Axis[State])
            (logPred + logOmegaT, ())

      LogProb(logsumexp(logAlpha))

    override def sample(key: Key): Tensor1[Time, Float32] =
      val (kInit, kSeq) = key.split2()
      var state: Tensor0[Int32] = Categorical(Prob(rho)).sample(kInit)
      val ys = kSeq.split(n).map { k =>
        val (kEmit, kTrans) = k.split2()
        val yt = Normal(mu.select(Axis[State].at(state)), sigma).sample(kEmit)
        val row = gamma.select(Axis[State].at(state)) // Tensor1[Prime[State], _]
        state = Categorical(Prob(row)).sample(kTrans)
        yt
      }
      stack(ys, Axis[Time])

  /** `Tensor2[State, Prime[State]]` whose rows are i.i.d. `Dirichlet(1,1,1)` —
    * Stan's implicit uniform prior over each `simplex[3]` row of the transition
    * matrix.
    */
  class RowwiseUniformSimplex
      extends Distribution[(State, Prime[State]), Float32]:
    private val row =
      Dirichlet[Prime[State], Float32](
        Tensor1(Axis[Prime[State]] -> nStates).fill(1f)
      )
    override def sample(key: Key): Tensor2[State, Prime[State], Float32] =
      key.splitvmap(Axis[State] -> nStates)(k => row.sample(k))
    override def logProb(
        x: Tensor2[State, Prime[State], Float32]
    ): Tensor0[LogProb] =
      x.vmap(Axis[State])(r => row.logProb(r)).sum

  object RowwiseUniformSimplex:
    /** Every row on the simplex, so `Latent` samples each row by
      * stick-breaking.
      */
    given SupportOf[RowwiseUniformSimplex, (State, Prime[State]), Float32] =
      _ => Support.Rowwise(Support.Simplex())

  // -- the model ------------------------------------------------------

  private def samplePriors(using ProbContext): HMMParams =
    val mu = Mu ~ Normal(
      Tensor1(Axis[State] -> nStates).fill(0f),
      Tensor1(Axis[State] -> nStates).fill(1f)
    )
    val sigma = Sigma ~ HalfNormal(Tensor0(0f), Tensor0(1f))
    val rho = Rho ~ Dirichlet(Tensor1(Axis[State] -> nStates).fill(1f))
    val gamma = Gamma ~ RowwiseUniformSimplex()
    (mu = mu, sigma = sigma, rho = rho, gamma = gamma)

  /** The model over a length-`n` sequence. Forward-sampling it draws a fresh
    * HMM path; observing `Y` scores `hmm_marginal`.
    */
  def model(n: Int)(using ProbContext) =
    val p = samplePriors
    val y = Y ~ HMMMarginal(p.mu, p.sigma, p.rho, p.gamma, n)
    (mu = p.mu, sigma = p.sigma, rho = p.rho, gamma = p.gamma, y = y)

  /** The model conditioned on an observed sequence — the entry point for
    * inference.
    */
  def observed(y: Tensor1[Time, Float32])(using ProbContext) =
    observe(Y := y)(model(y.shape(Axis[Time])))

  /** The latent sites of the model observed on `y`, with the constraints of the
    * Stan model: `sigma` positive, `rho` and every row of `gamma` on the
    * simplex. `Latent` reads them off the distributions (HalfNormal, Dirichlet,
    * RowwiseUniformSimplex).
    */
  def latent(y: Tensor1[Time, Float32]): Latent =
    Latent.discover()(observed(y))

  /** Draw parameters from their priors and simulate a length-`n` sequence. */
  def generate(n: Int)(
      key: Key
  ): (params: HMMParams, path: Tensor1[Time, Float32]) =
    val r = ForwardSampler.run(key)(model(n))
    (
      params = (mu = r.mu, sigma = r.sigma, rho = r.rho, gamma = r.gamma),
      path = r.y
    )

  /** Simulate a length-`n` sequence for fixed parameters. */
  def simulate(params: HMMParams, n: Int)(key: Key): Tensor1[Time, Float32] =
    HMMMarginal(params.mu, params.sigma, params.rho, params.gamma, n).sample(
      key
    )

  // -- runnable example ---------------------------------------------------

  @main
  def hmmExample(): Unit =
    dimwit.initialize()

    // Well-separated components with sticky transitions, so the latent
    // regime is visible in the simulated series.
    val params: HMMParams = (
      mu = Tensor1(Axis[State] -> nStates).fromArray(Array(-4f, 0f, 4f)),
      sigma = Tensor0(0.7f),
      rho = Tensor1(Axis[State] -> nStates).fromArray(Array(0.8f, 0.1f, 0.1f)),
      gamma = Tensor2(Axis[State] -> nStates, Axis[Prime[State]] -> nStates)
        .fromArray(
          // row-stochastic: 0.9 stay, 0.05 to each other state
          Array(0.9f, 0.05f, 0.05f, 0.05f, 0.9f, 0.05f, 0.05f, 0.05f, 0.9f)
        )
    )

    val y = simulate(params, n = 60)(Key(1L))

    val posterior = latent(y)
    println(s"latent sites: ${posterior.sites.map(_.name).mkString(", ")}")

    val atTruth = posterior.initWith(
      Mu := params.mu,
      Sigma := params.sigma,
      Rho := params.rho,
      Gamma := params.gamma
    )
    println(
      f"joint log-density (hmm_marginal + priors) at the true parameters = ${posterior.modelLogDensity(atTruth).item}%.3f"
    )

    // The warmup tunes the step size and the mass matrix.
    val (warmupKey, sampleKey) = Key(3L).split2()
    val adapted =
      NUTS.adapt(posterior, posterior.init, warmupKey, numSteps = 500)
    println(f"NUTS adapted step size: ${adapted.stepSize}%.3f")
    val (samples, infos) = adapted.sampler
      .run(adapted.position, sampleKey)
      .take(500)
      .toList
      .unzip
    println(
      f"NUTS divergences: ${infos.count(_.isDivergent)} / ${samples.size}"
    )

    trait Draw derives Label
    val muMean = stack(samples.map(_(Mu)), Axis[Draw]).mean(Axis[Draw])
    val sigmaMean = stack(samples.map(_(Sigma)), Axis[Draw]).mean
    println(
      s"mu:    $muMean   (true ${params.mu}, up to the order of the states)"
    )
    println(f"sigma: ${sigmaMean.item}%.3f   (true ${params.sigma.item})")
