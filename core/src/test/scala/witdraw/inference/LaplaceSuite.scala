package witdraw
package inference

import dimwit.*
import dimwit.stats.*

class LaplaceSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  // sampling thousands of draws step by step takes close to munit's default 30 s
  override val munitTimeout = scala.concurrent.duration.Duration(120, "s")

  trait Sample derives Label

  object Mu extends Site[Tensor0[Float32]]
  object Y extends Site[Tensor1[Sample, Float32]]

  test("the Laplace approximation is exact for a Gaussian posterior"):
    // mu ~ N(0, 1), y_i ~ N(mu, 1): the posterior is N(sum(y) / 4, 1 / 4)
    val yData = Tensor1(Axis[Sample]).fromArray(Array(0.5f, -1f, 2f))
    def model(using ProbContext) =
      val mu = Mu ~ Normal(0f, 1f)
      Y ~ Normal.isotropic(Tensor1(Axis[Sample] -> 3).fill(0f) +! mu, 1f)
    val latent = Latent.discover()(observe(Y := yData)(model))
    // starts away from the mode, which init finds itself
    val laplace = Laplace(latent)
    val start = latent.initWith(Mu := Tensor0(-2f))
    val draws = laplace.sample(start, Key(7), 4000).map(_._1(Mu).item)
    // the compiled loop gives the draws of the step-by-step one
    val looped = laplace.run(start, Key(7)).take(20).map(_._1(Mu).item).toList
    for (a, b) <- looped.zip(draws) do assertEqualsFloat(a, b, 1e-5f)
    val mean = draws.sum / draws.size
    val variance = draws.map(d => (d - mean) * (d - mean)).sum / draws.size
    assertEqualsFloat(mean, 0.375f, 0.03f)
    assertEqualsFloat(variance, 0.25f, 0.03f)
