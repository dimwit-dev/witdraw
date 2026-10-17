package witdraw.examples.stan

import dimwit.*
import dimwit.stats.*
import witdraw.*

object LinearRegression:

  trait Data derives Label // N — one row per observation
  trait Predictor derives Label // K — one entry per regression coefficient

  object Alpha extends Site[Tensor0[Float32]] // intercept
  object Beta
      extends Site[Tensor1[Predictor, Float32]] // coefficients for predictors
  object Sigma extends Site[Tensor0[Float32]] // error scale (> 0)
  object Y extends Site[Tensor1[Data, Float32]] // outcome vector

  def model(x: Tensor2[Data, Predictor, Float32])(using ProbContext) =
    val nPredictors = x.shape(Axis[Predictor])

    val alpha = Alpha ~ Normal(Tensor0(0f), Tensor0(100f))
    val beta = Beta ~ Normal(
      Tensor1(Axis[Predictor] -> nPredictors).fill(0f),
      Tensor1(Axis[Predictor] -> nPredictors).fill(100f)
    )
    val sigma = Sigma ~ HalfCauchy(Tensor0(0f), Tensor0(25f))

    val mu: Tensor1[Data, Float32] = x.dot(Axis[Predictor])(beta) +! alpha
    val y = Y ~ Normal.isotropic(mu, sigma)

    (alpha = alpha, beta = beta, sigma = sigma, y = y)

  /** The free parameters, for conditioning the model on observed `x` / `y` and
    * scoring the joint log-density (e.g. to hand to an MCMC kernel).
    */
  type Posterior = (
      alpha: Tensor0[Float32],
      beta: Tensor1[Predictor, Float32],
      sigma: Tensor0[Float32]
  )

  def logDensity(
      x: Tensor2[Data, Predictor, Float32],
      y: Tensor1[Data, Float32]
  )(params: Posterior): Tensor0[Float32] =
    ModelScorer.run:
      observe(Y := y):
        observeAll(params):
          model(x)
