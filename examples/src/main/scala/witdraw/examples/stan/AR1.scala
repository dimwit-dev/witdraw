package witdraw.examples.stan

import dimwit.*
import dimwit.stats.*
import witdraw.*
import plotwit.*
import plotwit.PlotTargets.websocket
import io.circe.Json

object AR1:

  trait Time derives Label // the index n = 1..N

  object Alpha extends Site[Tensor0[Float32]] // intercept
  object Beta extends Site[Tensor0[Float32]] // autoregressive coefficient
  object Sigma extends Site[Tensor0[Float32]] // innovation scale (> 0)
  object Y extends Site[Tensor1[Time, Float32]] // the full observed series

  /** Values of the three parameters, e.g. the true values to simulate from. */
  type AR1Params = (
      alpha: Tensor0[Float32],
      beta: Tensor0[Float32],
      sigma: Tensor0[Float32]
  )

  class AR1Likelihood(
      alpha: Tensor0[Float32],
      beta: Tensor0[Float32],
      sigma: Tensor0[Float32],
      n: Int,
      y0: Tensor0[Float32] = Tensor0(0f)
  ) extends MultivariateDistribution[Time, Float32]:

    override def logProb(y: Tensor1[Time, Float32]): Tensor0[LogProb] =
      val len = y.shape(Axis[Time])
      val yLag = y.slice(Axis[Time].at(0 until len - 1)) // y[1 .. N-1]
      val yLead = y.slice(Axis[Time].at(1 until len)) // y[2 .. N]
      Normal.isotropic(alpha +! (beta *! yLag), sigma).logProb(yLead)

    override def sample(key: Key): Tensor1[Time, Float32] =
      val values = key
        .split(n - 1)
        .scanLeft(y0) { (yPrev, k) =>
          // all scalars here, so plain + / * (not the broadcasting +! / *!)
          Normal.isotropic(alpha + beta * yPrev, sigma).sample(k)
        }
      stack(values, Axis[Time]) // length 1 + (n - 1) = n

  private def samplePriors(using ProbContext) =
    val alpha = Alpha ~ Normal(Tensor0(0f), Tensor0(100f))
    val beta = Beta ~ Normal(Tensor0(0f), Tensor0(100f))
    val sigma = Sigma ~ HalfCauchy(Tensor0(0f), Tensor0(25f))
    (alpha = alpha, beta = beta, sigma = sigma)

  def model(n: Int, y0: Tensor0[Float32] = Tensor0(0f))(using ProbContext) =
    val ps = samplePriors
    val y = Y ~ AR1Likelihood(ps.alpha, ps.beta, ps.sigma, n, y0)
    (alpha = ps.alpha, beta = ps.beta, sigma = ps.sigma, y = y)

  /** The model conditioned on an observed series — the idiomatic entry point
    * for inference.
    */
  def observed(y: Tensor1[Time, Float32])(using ProbContext) =
    observe(Y := y)(model(y.shape(Axis[Time])))

  /** The latent sites of the model observed on `y`, with `sigma` positive. */
  def latent(y: Tensor1[Time, Float32]): Latent =
    Latent.discover(Constraint(Sigma, Bijector.exp))(observed(y))

  // -- Generative helpers -------------------------------------------------

  /** Roll the recursion forward for fixed parameters (e.g. posterior draws) —
    * useful for forecasting from the tail of an observed series.
    */
  def simulate(params: AR1Params, n: Int, y0: Tensor0[Float32])(
      key: Key
  ): Tensor1[Time, Float32] =
    AR1Likelihood(params.alpha, params.beta, params.sigma, n, y0).sample(key)

  def generate(n: Int, y0: Tensor0[Float32] = Tensor0(0f))(
      key: Key
  ): (params: AR1Params, path: Tensor1[Time, Float32]) =
    val r = ForwardSampler.run(key)(model(n, y0))
    (params = (alpha = r.alpha, beta = r.beta, sigma = r.sigma), path = r.y)

  // -- Runnable example --------------------------------------------------

  /** Plot a handful of simulated AR(1) paths:
    *
    *   - four "free" draws from the process — no data, `Y` left free so
    *     [[ForwardSampler]] rolls the recursion forward from the process mean;
    *   - one path whose initial value `y[1]` is *observed* far from the mean
    *     (the Stan model likewise conditions on `y[1]`), which then relaxes
    *     back toward it.
    *
    * Needs a plotwit websocket viewer listening on the default port.
    */
  @main
  def ar1Simulations(): Unit =
    dimwit.initialize()

    val n = 160
    val alpha = 1f
    val beta = 0.7f // |beta| < 1 → stationary, so the paths stay comparable
    val sigma = 1f
    val processMean = alpha / (1f - beta) // ≈ 3.33

    val params: AR1Params =
      (alpha = Tensor0(alpha), beta = Tensor0(beta), sigma = Tensor0(sigma))

    // Four series with no observation: pure draws from the process, each
    // started at the process mean so they are directly comparable.
    val nFree = 4
    val freePaths =
      Key(1L)
        .split(nFree)
        .map(k => simulate(params, n, Tensor0(processMean))(k))

    // One series where y[1] is observed to be 15 (well off the mean); the
    // path then relaxes back toward the process mean.
    val observedPath = simulate(params, n, Tensor0(15f))(Key(99L))

    // Show the inference entry point runs on that "observed" series.
    val posterior = latent(observedPath)
    val atTruth = posterior.initWith(
      Alpha := params.alpha,
      Beta := params.beta,
      Sigma := params.sigma
    )
    println(
      f"log density of the observed series under the model: ${posterior.modelLogDensity(atTruth).item}%.1f"
    )

    val named =
      freePaths.zipWithIndex.map((p, i) => s"free draw $i" -> p.toArray) :+
        ("observed y[1] = 15" -> observedPath.toArray)

    display(
      timeSeriesPlot(named, f"AR(1): alpha=$alpha, beta=$beta, sigma=$sigma")
    )

  /** Minimal multi-series Vega-Lite line spec, built straight from arrays. */
  private def timeSeriesPlot(
      series: Seq[(String, Array[Float])],
      title: String
  ): UnitSpec =
    val values = series.flatMap: (name, ys) =>
      ys.indices.map: i =>
        Json.obj(
          "series" -> Json.fromString(name),
          "n" -> Json.fromInt(i),
          "y" -> Json.fromFloatOrNull(ys(i))
        )
    UnitSpec(
      Json.obj(
        "$schema" -> Json.fromString(
          "https://vega.github.io/schema/vega-lite/v6.json"
        ),
        "title" -> Json.fromString(title),
        "width" -> Json.fromInt(640),
        "height" -> Json.fromInt(360),
        "data" -> Json.obj("values" -> Json.fromValues(values)),
        "mark" -> Json.obj("type" -> Json.fromString("line")),
        "encoding" -> Json.obj(
          "x" -> Json.obj(
            "field" -> Json.fromString("n"),
            "type" -> Json.fromString("quantitative")
          ),
          "y" -> Json.obj(
            "field" -> Json.fromString("y"),
            "type" -> Json.fromString("quantitative")
          ),
          "color" -> Json.obj(
            "field" -> Json.fromString("series"),
            "type" -> Json.fromString("nominal"),
            "legend" -> Json.obj("title" -> Json.fromString("series"))
          )
        )
      )
    )
