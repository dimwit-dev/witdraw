package witdraw
package examples.pymc

import dimwit.*
import dimwit.linalg.LinearAlgebra
import dimwit.stats.{Gamma, HalfNormal, MVNormal, Normal}
import witdraw.*
import witdraw.inference.NUTS

import plotwit.*
import plotwit.plots.{linePlot, scatterPlot}
import plotwit.PlotTargets.desktopBrowser
import witdraw.ForwardSampler
import plotwit.plots.histogramPlot

/** Gaussian process regression with a squared-exponential kernel */
object GPRegression:

  trait Point derives Label // training inputs
  trait NewPoint derives Label // inputs to predict at

  object Ell extends Site[Tensor0[Float32]] // length scale
  object Eta extends Site[Tensor0[Float32]] // amplitude
  object Sigma extends Site[Tensor0[Float32]] // observation noise
  object Y extends Site[Tensor1[Point, Float32]] // observed values

  /** `k(a, b) = eta² · exp(-(a - b)² / (2 ell²))` at all pairs of `xa` and
    * `xb`. The axes may differ, which gives cross-covariances, e.g. between the
    * new and the training inputs.
    */
  def expQuad[A: Label, B: Label](eta: Tensor0[Float32], ell: Tensor0[Float32])(
      xa: Tensor1[A, Float32],
      xb: Tensor1[B, Float32]
  ): Tensor2[A, B, Float32] =
    val scale = Tensor0(-0.5f) / (ell * ell)
    xa.vmap(Axis[A]): a =>
      ((xb -! a).pow(2f) *! scale).exp *! (eta * eta)

  /** `K + d·I`, for a square matrix `K`. */
  def addDiagonal[L: Label](
      k: Tensor2[L, Prime[L], Float32],
      d: Tensor0[Float32]
  ): Tensor2[L, Prime[L], Float32] =
    val identity = Tensor2(k.extent(Axis[L])).eye
    k + (identity *! d)

  /** The kernel matrix of `x` with itself, `K[i, j] = k(x_i, x_j)`; the columns
    * are on the primed axis, as a covariance matrix needs.
    */
  def gram[L: Label](eta: Tensor0[Float32], ell: Tensor0[Float32])(
      x: Tensor1[L, Float32]
  ): Tensor2[L, Prime[L], Float32] =
    expQuad(eta, ell)(x, x.relabelTo(Axis[Prime[L]]))

  /** The GP at `xNew` conditioned on the observations `y` at `x`, whose
    * covariance is `ky = K + sigma² I`:
    * `f* | y ~ MVNormal(K*ᵀ Ky⁻¹ y, K** - K*ᵀ Ky⁻¹ K*)`, with `K* = k(x, xNew)`
    * and `K** = k(xNew, xNew)` (`gp.conditional` in PyMC).
    *
    * Computed through the Cholesky factor `L` of `ky`, as `inv(ky)` is too
    * imprecise in Float32: with `V = L⁻¹ K*`, the mean is `Vᵀ L⁻¹ y` and the
    * covariance `K** - Vᵀ V`.
    */
  def conditional(eta: Tensor0[Float32], ell: Tensor0[Float32])(
      x: Tensor1[Point, Float32],
      ky: Tensor2[Point, Prime[Point], Float32],
      y: Tensor1[Point, Float32],
      xNew: Tensor1[NewPoint, Float32]
  ): (
      mean: Tensor1[NewPoint, Float32],
      cov: Tensor2[NewPoint, Prime[NewPoint], Float32]
  ) =
    val l = LinearAlgebra.cholesky(ky) // Tensor2[Point, Prime[Point]]
    // L⁻¹ k, column by column, as dimwit's solve takes a vector
    def whiten[A: Label](k: Tensor2[Point, A, Float32]) =
      k.vmap(Axis[A])(LinearAlgebra.solve(l, _)) // Tensor2[A, Prime[Point]]
    val v = whiten(expQuad(eta, ell)(x, xNew))
    val vPrime = whiten(
      expQuad(eta, ell)(x, xNew.relabelTo(Axis[Prime[NewPoint]]))
    )
    (
      mean = v.dot(Axis[Prime[Point]])(LinearAlgebra.solve(l, y)),
      cov = gram(eta, ell)(xNew) - v.dot(Axis[Prime[Point]])(vPrime)
    )

  /** A draw of `MVNormal(mean, cov)`, written as `mean + chol(cov) z` with
    * `z ~ N(0, I)` at the site `z`. Same distribution, but an inference
    * algorithm samples the independent `z` instead of a correlated vector whose
    * shape changes with the other parameters, which is much easier for NUTS.
    */
  def nonCentered[L: Label](z: SiteDef[Tuple1[Prime[L]], Float32])(
      mean: Tensor1[L, Float32],
      cov: Tensor2[L, Prime[L], Float32]
  )(using ProbContext): Tensor1[L, Float32] =
    val zeros = Tensor1(Axis[Prime[L]] -> mean.shape(Axis[L])).fill(0f)
    val ones = Tensor1(Axis[Prime[L]] -> mean.shape(Axis[L])).fill(1f)
    mean + LinearAlgebra
      .cholesky(cov)
      .dot(Axis[Prime[L]])(z ~ Normal(zeros, ones))

  object ZNew extends Site[Tensor1[Prime[NewPoint], Float32]] // whitened f*

  /** Marginal GP */
  def model(x: Tensor1[Point, Float32], xNew: Tensor1[NewPoint, Float32])(using
      ProbContext
  ) =
    val ell = Ell ~ Gamma(Tensor0(2f), Tensor0(1f))
    val eta = Eta ~ HalfNormal(0f, 2f)
    val sigma = Sigma ~ HalfNormal(0f, 1f)

    // y = f + noise, with f ~ GP(0, k) integrated out
    val ky = addDiagonal(gram(eta, ell)(x), sigma * sigma + Tensor0(1e-6f))
    val y = Y ~ MVNormal(Tensor1(x.extent(Axis[Point])).fill(0f), ky)

    // f* | y. Float32 rounding in cov grows with eta², so the jitter that keeps
    // it positive definite does too.
    val (mean, cov) = conditional(eta, ell)(x, ky, y, xNew)
    val fNew =
      nonCentered(ZNew)(mean, addDiagonal(cov, eta * eta * Tensor0(1e-3f)))
    (ell = ell, eta = eta, sigma = sigma, fNew = fNew)

  trait Function derives Label // one sampled function per index

  @main
  def runGPRegression(): Unit =
    dimwit.initialize()

    // Synthetic data: y = sin(x) + noise
    val n = 30
    val (dataKey, rest) = Random.Key(1).split2()
    val xs = Array.tabulate(n)(i => 10f * i / (n - 1))
    val x = Tensor1(Axis[Point]).fromArray(xs)
    val noise = Normal(
      Tensor1(x.extent(Axis[Point])).fill(0f),
      Tensor1(x.extent(Axis[Point])).fill(0.3f)
    ).sample(dataKey)
    val y = x.sin + noise
    val xNew = Tensor1(Axis[NewPoint]).fromArray(
      Array.tabulate(50)(i => 12f * i / 49 - 1f)
    )

    val xCond = Tensor1(Axis[Point]).fromArray(Array(1f, 8f))
    val yCond = Tensor1(Axis[Point]).fromArray(Array(1f, 1f))

    // some prior samples
    val samples =
      ForwardSampler.sample(Key(42), 10):
        observe(Eta := 1, Y := yCond, Sigma := 0.1f):
          model(xCond, x.relabelTo(Axis[NewPoint]))
    val pplots =
      for sample <- samples
      yield linePlot(x.relabelTo(Axis[NewPoint]), sample.fNew)
    display(
      grid(
        Seq(
          Seq(
            overlay(pplots*),
            histogramPlot(stack(samples.map(_.ell), Axis[Point]))
          )
        )
      )
    )

    System.exit(0)

    val posterior =
      Latent.discover:
        observe(Y := y)(model(x, xNew))
    println(s"latent sites: ${posterior.sites.map(_.name).mkString(", ")}")

    val (initKey, sampleKey) = rest.split2()
    val (warmupKey, drawKey) = sampleKey.split2()
    val adapted =
      NUTS.adapt(
        posterior,
        warmupKey,
        posterior.initRandom(initKey),
        numSteps = 500
      )
    val chain = adapted.sample(drawKey, 500)
    println(s"divergences: ${chain.numDivergent}")

    trait Draw derives Label
    val draws = chain.samples
    println(s"ell   ≈ ${stack(draws.map(_.ell), Axis[Draw]).mean.item}")
    println(s"eta   ≈ ${stack(draws.map(_.eta), Axis[Draw]).mean.item}")
    println(s"sigma ≈ ${stack(draws.map(_.sigma), Axis[Draw]).mean.item}")
    val fMean = stack(draws.map(_.fNew), Axis[Draw]).mean(Axis[Draw])

    val numFunctions = 8
    val priorKey = Random.Key(7)
    val data = scatterPlot(x, y, _.mark.color := "black")

    // Prior: the model run forward, without data. Each run draws its own
    // hyperparameters, so the functions vary in length scale and amplitude.
    val priorFunctions = stack(
      ForwardSampler.sample(priorKey, numFunctions)(model(x, xNew)).map(_.fNew),
      Axis[Function]
    )
    display(
      linePlot(
        xNew,
        priorFunctions,
        _.title := "GP prior functions",
        _.width := 500,
        _.height := 300
      )
    )

    // Posterior: f* is a site of the model, so every NUTS draw is a function
    // from the posterior; take a few, spread over the chain
    val picked = draws.indices.by(draws.size / numFunctions).take(numFunctions)
    val posteriorFunctions = stack(picked.map(draws(_).fNew), Axis[Function])
    display(
      overlay(
        linePlot(
          xNew,
          posteriorFunctions,
          _.title := "GP posterior functions and data",
          _.width := 500,
          _.height := 300
        ),
        linePlot(xNew, fMean, _.mark.color := "black", _.mark.strokeWidth := 3),
        data
      )
    )
