package witdraw.examples.stan

import dimwit.*
import dimwit.stats.*
import witdraw.*
import witdraw.mcmc.OptimizerKernel
import witdraw.mcmc.NUTSAlgorithm

/** Bayesian linear regression with **missing predictor values**, ported from
  * the recipe in the Stan User's Guide chapter on missing data:
  *
  * {{{
  * data {
  *   int<lower=0> N_obs;                  // rows with x observed
  *   int<lower=0> N_mis;                  // rows with x missing
  *   vector[N_obs] x_obs;
  *   vector[N_obs + N_mis] y;             // outcome fully observed, rows ordered obs-then-mis
  * }
  * parameters {
  *   real alpha;
  *   real beta;
  *   real<lower=0> sigma;
  *   real mu_x;                           // predictor population mean
  *   real<lower=0> sigma_x;               // predictor population sd
  *   vector[N_mis] x_mis;                 // the imputed predictor values
  * }
  * model {
  *   vector[N_obs + N_mis] x = append_row(x_obs, x_mis);
  *   x_obs ~ normal(mu_x, sigma_x);       // observed entries inform (mu_x, sigma_x)
  *   x_mis ~ normal(mu_x, sigma_x);       // missing entries drawn from the same population
  *   y     ~ normal(alpha + beta * x, sigma);
  *   alpha ~ normal(0, 10);  beta ~ normal(0, 10);
  *   sigma ~ cauchy(0, 5);   mu_x ~ normal(0, 10);  sigma_x ~ cauchy(0, 5);
  * }
  * }}}
  *
  * The idea: the missing `x` entries are ordinary model parameters
  * ([[XMis]]). They are given a small **imputation model**
  * `x_n ~ Normal(mu_x, sigma_x)`; the observed entries pin down
  * `(mu_x, sigma_x)`, and the missing ones are then sampled jointly with the
  * regression coefficients. So the imputation uncertainty propagates into the
  * posterior for `alpha` / `beta` instead of being thrown away (as it would be
  * by mean-imputation or complete-case analysis).
  *
  * Row-order convention: the caller arranges the design so the `N_obs` rows
  * whose predictor is known come first, followed by the `N_mis` rows whose
  * predictor is missing. `y` is the full length-`N` vector in that same order.
  * The model rebuilds the predictor column with
  * `concatenate(xObs, xMis)` along [[Data]].
  *
  * Missing **outcomes** are strictly easier: a missing-at-random `y_n` that
  * you don't care to track can simply be dropped from the likelihood, and one
  * you do want in the trace is just another latent site concatenated onto `y`
  * the same way `x_mis` is concatenated onto `x` here.
  */
object MissingData:

  trait Data derives Label    // N = N_obs + N_mis, rows ordered observed-then-missing
  trait Missing derives Label // the N_mis rows whose predictor is missing

  object Alpha extends Site[Tensor0[Float32]]  // intercept
  object Beta extends Site[Tensor0[Float32]]   // slope
  object Sigma extends Site[Tensor0[Float32]]  // residual scale (> 0)
  object MuX extends Site[Tensor0[Float32]]    // predictor population mean
  object SigmaX extends Site[Tensor0[Float32]] // predictor population sd (> 0)

  object XObs extends Site[Tensor1[Data, Float32]]    // observed predictor entries (conditioned)
  object XMis extends Site[Tensor1[Missing, Float32]] // imputed predictor entries (latent)
  object Y extends Site[Tensor1[Data, Float32]]       // outcome, full length N

  type Posterior = (
      alpha: Tensor0[Float32],
      beta: Tensor0[Float32],
      sigma: Tensor0[Float32],
      muX: Tensor0[Float32],
      sigmaX: Tensor0[Float32],
      xMis: Tensor1[Missing, Float32] // the imputed values are part of the posterior
  )

  /** @param nObs number of rows with the predictor observed
    * @param nMissing number of rows with the predictor missing
    */
  def model(nObs: Int, nMissing: Int) = ProbProg:
    val alpha = Alpha ~ Normal(Tensor0(0f), Tensor0(10f))
    val beta = Beta ~ Normal(Tensor0(0f), Tensor0(10f))
    val sigma = Sigma ~ HalfCauchy(Tensor0(0f), Tensor0(5f))

    // -- imputation model for the predictor -------------------------------
    val muX = MuX ~ Normal(Tensor0(0f), Tensor0(10f))
    val sigmaX = SigmaX ~ HalfCauchy(Tensor0(0f), Tensor0(5f))

    val obsMean = Tensor1(Axis[Data] -> nObs).fill(0f) +! muX
    val misMean = Tensor1(Axis[Missing] -> nMissing).fill(0f) +! muX

    val xObs = XObs ~ Normal.isotropic(obsMean, sigmaX)
    val xMis = XMis ~ Normal.isotropic(misMean, sigmaX)

    // completed predictor column: observed rows first, then imputed rows
    val x: Tensor1[Data, Float32] =
      concatenate(xObs, xMis.relabelTo(Axis[Data]), Axis[Data])

    // -- regression ------------------------------------------------------
    val mu = alpha +! (beta *! x)
    val y = Y ~ Normal.isotropic(mu, sigma)

    (
      alpha = alpha,
      beta = beta,
      sigma = sigma,
      muX = muX,
      sigmaX = sigmaX,
      xMis = xMis,
      y = y
    )

  /** The model conditioned on the observed predictor entries and the full
    * outcome vector — the entry point for inference.
    */
  def observed(
      xObs: Tensor1[Data, Float32],
      y: Tensor1[Data, Float32],
      nMissing: Int
  ): ProbProg[?] =
    model(xObs.shape(Axis[Data]), nMissing)
      .observe(XObs := xObs)
      .observe(Y := y)

  def logDensity(
      xObs: Tensor1[Data, Float32],
      y: Tensor1[Data, Float32],
      nMissing: Int
  )(params: Posterior): Tensor0[Float32] =
    ModelScorer.run(observed(xObs, y, nMissing).observeAll(params))

  // -- runnable example -------------------------------------------------

  @main
  def missingDataExample(): Unit =
    dimwit.initialize()

    // ---- ground truth -------------------------------------------------
    val nObs = 80
    val nMissing = 20
    val n = nObs + nMissing

    val alphaTrue = 1.0f
    val betaTrue = 2.0f
    val sigmaTrue = 0.5f
    val muXTrue = 3.0f
    val sigmaXTrue = 1.5f

    val (kx, ky) = Key(0L).split2()
    val xAll = Normal
      .isotropic(Tensor1(Axis[Data] -> n).fill(muXTrue), Tensor0(sigmaXTrue))
      .sample(kx)
    val yAll = Normal
      .isotropic(alphaTrue +! (betaTrue *! xAll), Tensor0(sigmaTrue))
      .sample(ky)

    // Rows are already in "observed then missing" order: hide x on the tail.
    val xObs = xAll.slice(Axis[Data].at(0 until nObs))
    val xMisTrue = xAll.slice(Axis[Data].at(nObs until n)).relabelTo(Axis[Missing])

    // ---- inference --------------------------------------------------
    val target = logDensity(xObs, yAll, nMissing)

    val xBar = xObs.mean.item
    val init: Posterior = (
      alpha = Tensor0(0f),
      beta = Tensor0(0f),
      sigma = Tensor0(1f),
      muX = Tensor0(xBar),
      sigmaX = Tensor0(1f),
      xMis = Tensor1(Axis[Missing] -> nMissing).fill(xBar) // mean-impute as a start
    )

    // quick gradient ascent to a MAP-ish point, then NUTS from there
    val ok = OptimizerKernel.toSamplingAlgorithm(
      optimizer = dimwit.optimizer.Adam(learningRate = -0.05f),
      gradFn = (p: Posterior) => Autodiff.grad(target)(p)
    )
    val start = ok.run(init, Key(1L)).take(400).toList.last._1

    val nuts = NUTSAlgorithm[Posterior](stepSize = 0.02)(target)
    trait Sample derives Label
    val (samples, infos) = nuts
      .run(start, Key(2L))
      .drop(500)
      .take(1500)
      .toList
      .unzip

    println(
      f"NUTS divergences: ${infos.count(_.isDivergent)} / ${samples.length}"
    )

    def postMean(f: Posterior => Tensor0[Float32]): Float =
      stack(samples.map(f), Axis[Sample]).mean.item

    println(f"alpha : ${postMean(_.alpha)}%6.3f   (true $alphaTrue)")
    println(f"beta  : ${postMean(_.beta)}%6.3f   (true $betaTrue)")
    println(f"sigma : ${postMean(_.sigma)}%6.3f   (true $sigmaTrue)")
    println(f"muX   : ${postMean(_.muX)}%6.3f   (true $muXTrue)")
    println(f"sigmaX: ${postMean(_.sigmaX)}%6.3f   (true $sigmaXTrue)")

    // how well were the 20 hidden predictor values recovered?
    val xMisPost: Tensor1[Missing, Float32] =
      stack(samples.map(_.xMis), Axis[Sample]).mean(Axis[Sample])
    val rmse = (xMisPost - xMisTrue).pow(Tensor0(2f)).mean.sqrt.item
    println(
      f"imputed x  RMSE vs truth: $rmse%.3f  " +
        f"(vs sigma_x = $sigmaXTrue, the spread with no outcome to lean on)"
    )
