package witdraw
package distributions

import dimwit.*

class LKJSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  trait D derives Label

  def lkj(eta: Float, d: Int) =
    LKJ(Tensor0(eta), Shape(Axis[D] -> d, Axis[Prime[D]] -> d))

  def correlation(entries: Array[Array[Float]]) =
    Tensor2(Axis[D], Axis[Prime[D]]).fromArray(entries)

  def logProbAt(eta: Float, entries: Array[Array[Float]]): Float =
    lkj(eta, entries.length).logProb(correlation(entries)).asFloat32.item

  test("2×2, η = 1: uniform on the correlation r, density 1/2"):
    for r <- Seq(-0.8f, 0f, 0.5f) do
      assertEqualsFloat(logProbAt(1f, Array(Array(1f, r), Array(r, 1f))), math.log(0.5).toFloat, 1e-4f)

  test("2×2, η = 2: density 3/4 · (1 - r²)"):
    val r = 0.3f
    val expected = math.log(0.75 * (1 - r * r)).toFloat
    assertEqualsFloat(logProbAt(2f, Array(Array(1f, r), Array(r, 1f))), expected, 1e-4f)

  test("3×3, η = 1: uniform over the correlation matrices, which have volume π²/2"):
    val R = Array(Array(1f, 0.2f, -0.3f), Array(0.2f, 1f, 0.4f), Array(-0.3f, 0.4f, 1f))
    assertEqualsFloat(logProbAt(1f, R), math.log(2 / (math.Pi * math.Pi)).toFloat, 1e-4f)
