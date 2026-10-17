package witdraw
package inference

import dimwit.*
import dimwit.stats.*

class MAPSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  object P extends Site[Tensor0[Float32]]
  object Successes extends Site[Tensor0[Int32]]

  // p ~ Uniform(0, 1), with 6 successes in 9 trials, on the logit scale
  lazy val latent = Latent.discover(
    Constraint(P, Bijector.interval(Tensor0(0f), Tensor0(1f)))
  ):
    observe(Successes := Tensor0(6)):
      Successes ~ Binomial(9, Prob(P ~ Uniform(0f, 1f)))

  test("MAP maximises the model's density, whatever the constraint"):
    // the MAP is 6/9; with the Jacobian of the logit scale, it would be 7/11
    assertEqualsFloat(MAP(latent)(P).item, 6f / 9f, 1e-3f)

  test("MAP starts from given values"):
    val map = MAP(latent, start = Seq(P := Tensor0(0.1f)))
    assertEqualsFloat(map(P).item, 6f / 9f, 1e-3f)

  test("MAP that has not converged is an error"):
    val e = intercept[IllegalStateException](
      MAP(latent, start = Seq(P := Tensor0(0.01f)), maxSteps = 100, learningRate = 0.001f)
    )
    assert(e.getMessage.contains("did not converge"), e.getMessage)
