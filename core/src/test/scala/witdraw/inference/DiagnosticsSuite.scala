package witdraw
package inference

import dimwit.*
import dimwit.stats.*

class DiagnosticsSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  trait K derives Label

  /** `numChains` chains of independent standard normal draws, shifted by
    * `offset(c)` for chain c.
    */
  def normalChains(numChains: Int, numDraws: Int, key: Key)(offset: Int => Float) =
    key.split(numChains).zipWithIndex.map: (k, c) =>
      val draws = Normal.standardNormal(Shape(Axis[K] -> numDraws)).sample(k).toArray
      draws.toSeq.map(d => Tensor0(d + offset(c)))

  test("rhat is close to 1 for chains of the same distribution"):
    val chains = normalChains(4, 1000, Key(1))(_ => 0f)
    assertEqualsFloat(Diagnostics.rhat(chains).item, 1f, 0.01f)

  test("rhat is large for chains that disagree"):
    val chains = normalChains(4, 1000, Key(2))(c => 3f * c)
    val rhat = Diagnostics.rhat(chains).item
    assert(rhat > 2, s"rhat $rhat")

  test("rhat detects a single chain that drifts"):
    val drifting = (0 until 1000).map(i => Tensor0(i / 100f))
    val rhat = Diagnostics.rhat(Seq(drifting)).item
    assert(rhat > 1.5, s"rhat $rhat")

  test("ess is about the number of independent draws"):
    val chains = normalChains(4, 1000, Key(3))(_ => 0f)
    val ess = Diagnostics.ess(chains).item
    assert(ess > 3000 && ess < 5000, s"ess $ess")

  test("ess is small for strongly correlated draws"):
    // a random walk: each draw is the last one plus a little noise
    val noise = Normal.standardNormal(Shape(Axis[K] -> 1000)).sample(Key(4)).toArray
    val walk = noise.scanLeft(0f)(_ + _).tail.toSeq.map(Tensor0(_))
    val ess = Diagnostics.ess(Seq(walk)).item
    assert(ess < 50, s"ess $ess")

  test("the diagnostics keep the shape of the quantity, also for axes of size 1"):
    val chains = Seq.fill(2)(
      (0 until 10).map(i => Tensor1(Axis[K]).fromArray(Array(i.toFloat)))
    )
    assertEquals(Diagnostics.rhat(chains).shape, Shape(Axis[K] -> 1))
    assertEquals(Diagnostics.ess(chains).shape, Shape(Axis[K] -> 1))
