package witdraw

import dimwit.*
import dimwit.stats.*

/** A small normal model shared by the test suites: mu and sigma, and three
  * observations y.
  */
object NormalModel:
  trait Sample derives Label

  object Mu extends Site[Tensor0[Float32]]
  object Sigma extends Site[Tensor0[Float32]]
  object Y extends Site[Tensor1[Sample, Float32]]

  val n = 3

  def prior(using ProbContext) =
    (mu = Mu ~ Normal(0f, 10f), sigma = Sigma ~ HalfNormal(0f, 1f))

  def likelihood(mu: Tensor0[Float32], sigma: Tensor0[Float32])(using
      ProbContext
  ) =
    Y ~ Normal.isotropic(Tensor1(Axis[Sample] -> n).fill(0f) +! mu, sigma)

  def model(using ProbContext) =
    val p = prior
    val y = likelihood(p.mu, p.sigma)
    (mu = p.mu, sigma = p.sigma, y = y)

  lazy val yData = Tensor1(Axis[Sample]).fromArray(Array(0.5f, -1f, 2f))

  def manualLogProb(mu: Float, sigma: Float): Float =
    val muT = Tensor0(mu)
    val sigmaT = Tensor0(sigma)
    (Normal(Tensor0(0f), Tensor0(10f)).logProb(muT).asFloat +
      HalfNormal(Tensor0(0f), Tensor0(1f)).logProb(sigmaT).asFloat +
      Normal
        .isotropic(Tensor1(Axis[Sample] -> n).fill(0f) +! muT, sigmaT)
        .logProb(yData)
        .asFloat).item
