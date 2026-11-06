package witdraw
package distributions

import dimwit.*
import dimwit.stats.*
import me.shadaj.scalapy.py

class InverseSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  trait N derives Label

  lazy val ys = Tensor1(Axis[N]).fromArray(Array(0.1f, 0.7f, 1.5f, 4f))

  // the reference implementation of the inverse-gamma distribution
  lazy val invgamma = py.module("scipy.stats").invgamma

  test("Inverse(Gamma(a, b)) is SciPy's inverse-gamma with shape a and scale b"):
    for (a, b) <- Seq((2f, 3f), (0.5f, 1f), (3.7f, 0.2f)) do
      val inv = Inverse(Gamma(Tensor.like(ys).fill(a), Tensor.like(ys).fill(b)))
      inv.elementWiseLogProb(ys).asFloat.toArray.zip(ys.toArray).foreach: (lp, y) =>
        val expected =
          invgamma.logpdf(y.toDouble, a.toDouble, scale = b.toDouble).as[Double]
        assertEqualsFloat(lp, expected.toFloat, 1e-4f, s"a = $a, b = $b, y = $y")

  test("the density is zero at and below zero"):
    val inv: Inverse[EmptyTuple, Float32] = Inverse(Gamma(Tensor0(2f), Tensor0(1f)))
    assertEquals(inv.logProb(Tensor0(-0.5f)).asFloat.item, Float.NegativeInfinity)
    assertEquals(inv.logProb(Tensor0(0f)).asFloat.item, Float.NegativeInfinity)

  test("samples are positive"):
    val inv = Inverse(Exponential(Tensor.like(ys).fill(2f)))
    val keys = Random.Key(0).split(20)
    for k <- keys do assert(inv.sample(k).toArray.forall(_ > 0f))

  object Sigma2 extends Site[Tensor0[Float32]]

  test("an Inverse site is sampled on the log scale"):
    val latent = Latent.discover()(Sigma2 ~ Inverse(Gamma(Tensor0(2f), Tensor0(1f))))
    val p = latent.initWith(Sigma2 := Tensor0(1.5f))
    assertEqualsFloat(p(Sigma2).item, 1.5f, 1e-5f)
    assertEqualsFloat(
      (latent.logDensity(p) - latent.modelLogDensity(p)).item,
      math.log(1.5).toFloat,
      1e-4f
    )
