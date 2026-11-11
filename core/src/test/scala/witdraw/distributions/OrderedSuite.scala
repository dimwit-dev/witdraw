package witdraw
package distributions

import dimwit.*
import dimwit.stats.*

class OrderedSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  trait Comp derives Label
  object Mu extends Site[Tensor1[Comp, Float32]]

  lazy val ordered = Ordered(Normal(0f, 1f), Axis[Comp] -> 3)

  def increasing(x: Tensor1[Comp, Float32]) =
    x.toArray.sliding(2).forall(p => p(0) < p(1))

  test("forward samples are increasing"):
    for i <- 0 until 5 do
      val x = ForwardSampler.run(Key(i))(Mu ~ ordered)
      assert(increasing(x), x.toArray.mkString(", "))

  test("the density is k! times that of the draws, and zero when not increasing"):
    val x = Tensor1(Axis[Comp]).fromArray(Array(-1f, 0.5f, 2f))
    val draws = Normal(Tensor.like(x).fill(0f), Tensor.like(x).fill(1f)).logProb(x)
    val expected = draws.asFloat.item + math.log(6).toFloat
    assertEqualsFloat(ordered.logProb(x).asFloat.item, expected, 1e-5f)
    val unordered = Tensor1(Axis[Comp]).fromArray(Array(0.5f, -1f, 2f))
    assertEquals(ordered.logProb(unordered).asFloat.item, Float.NegativeInfinity)

  test("Latent samples an Ordered site as increasing values"):
    val latent = Latent.discover(Mu ~ ordered)
    for i <- 0 until 5 do
      val x = latent.result(latent.initRandom(Key(i)))
      assert(increasing(x), x.toArray.mkString(", "))

  test("the base distribution must take any real value"):
    val errors = compileErrors("Ordered(HalfNormal(0f, 1f), Axis[Comp] -> 3)")
    assert(errors.contains("is not known to take any real value"), errors)
    val widened = compileErrors(
      "Ordered(Normal(0f, 1f): Distribution[EmptyTuple, Float32], Axis[Comp] -> 3)"
    )
    assert(widened.contains("is not known to take any real value"), widened)
