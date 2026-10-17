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

  /** The latent sites of the model observed on `x` / `y`, with `sigma`
    * positive. `latent.logDensity` is the joint log-density to hand to an MCMC
    * kernel, and `latent.init` a starting point.
    */
  def latent(
      x: Tensor2[Data, Predictor, Float32],
      y: Tensor1[Data, Float32]
  ): Latent =
    Latent.discover(Constraint(Sigma, Bijector.exp)):
      observe(Y := y)(model(x))
