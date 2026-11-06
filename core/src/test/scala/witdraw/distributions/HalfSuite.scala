package witdraw
package distributions

import dimwit.*
import dimwit.stats.*

class HalfSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  trait N derives Label

  lazy val xs = Tensor1(Axis[N]).fromArray(Array(0.1f, 0.7f, 1.5f, 4f))
  lazy val loc = Tensor.like(xs).fill(0f)
  lazy val scale = Tensor.like(xs).fill(2f)

  def assertClose(a: Tensor1[N, LogProb], b: Tensor1[N, LogProb]): Unit =
    a.asFloat.toArray.zip(b.asFloat.toArray).foreach: (x, y) =>
      assertEqualsFloat(x, y, 1e-5f)

  test("Half(Normal) is HalfNormal"):
    assertClose(
      Half(Normal(loc, scale)).elementWiseLogProb(xs),
      HalfNormal(loc, scale).elementWiseLogProb(xs)
    )

  test("Half(Cauchy) is HalfCauchy"):
    assertClose(
      Half(Cauchy(loc, scale)).elementWiseLogProb(xs),
      HalfCauchy(loc, scale).elementWiseLogProb(xs)
    )

  test("Half(StudentT) has twice the density of StudentT above loc"):
    val t = StudentT(Tensor0(5f), loc, scale)
    val half = Half(t).elementWiseLogProb(xs).asFloat
    val full = t.elementWiseLogProb(xs).asFloat
    half.toArray.zip(full.toArray).foreach: (h, f) =>
      assertEqualsFloat(h, f + math.log(2).toFloat, 1e-5f)

  test("the density is zero below loc"):
    val half: Half[EmptyTuple, Float32] = Half(StudentT(5f, 1f, 1f))
    assertEquals(half.logProb(Tensor0(0.5f)).asFloat.item, Float.NegativeInfinity)

  test("samples are at or above loc"):
    val shifted = Tensor.like(xs).fill(3f)
    val half = Half(StudentT(Tensor0(5f), shifted, scale))
    val keys = Random.Key(0).split(20)
    for k <- keys do assert(half.sample(k).toArray.forall(_ >= 3f))

  object Tau extends Site[Tensor0[Float32]]

  test("a Half site is sampled on the log scale"):
    val latent = Latent.discover()(Tau ~ Half(StudentT(5f, 0f, 1f)))
    val p = latent.initWith(Tau := Tensor0(1.5f))
    assertEqualsFloat(p(Tau).item, 1.5f, 1e-5f)
    // tau = exp(z), so log |d tau / dz| = z = log tau
    assertEqualsFloat(
      (latent.logDensity(p) - latent.modelLogDensity(p)).item,
      math.log(1.5).toFloat,
      1e-4f
    )
