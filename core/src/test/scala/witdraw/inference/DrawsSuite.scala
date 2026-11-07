package witdraw
package inference

import dimwit.*

class DrawsSuite extends munit.FunSuite:

  // positions are plain floats here; the info says whether a draw diverged
  case class Info(divergent: Boolean)
  given SamplerInfo[Info] = _.divergent

  val draws = Draws(
    Seq(
      Seq(1f -> Info(false), 2f -> Info(true), 3f -> Info(false)),
      Seq(4f -> Info(false), 5f -> Info(false), 6f -> Info(true))
    )
  )

  test("sizes, positions and values per chain"):
    assertEquals((draws.numChains, draws.numDraws), (2, 3))
    assertEquals(draws.positions, Seq(Seq(1f, 2f, 3f), Seq(4f, 5f, 6f)))
    assertEquals(draws.all, Seq(1f, 2f, 3f, 4f, 5f, 6f))
    assertEquals(draws.values(_ * 2), Seq(Seq(2f, 4f, 6f), Seq(8f, 10f, 12f)))

  test("divergences come from the sampler's info"):
    assertEquals(draws.divergent, Seq(Seq(false, true, false), Seq(false, false, true)))
    assertEquals(draws.numDivergent, 2)

  test("draws without info have no divergences"):
    assertEquals(Draws.single(Seq(1f -> (), 2f -> ())).numDivergent, 0)

  test("chains must be given, non-empty, and of equal length"):
    intercept[IllegalArgumentException](Draws(Seq.empty[Seq[(Float, Unit)]]))
    intercept[IllegalArgumentException](Draws(Seq(Seq.empty[(Float, Unit)])))
    intercept[IllegalArgumentException](Draws(Seq(Seq(1f -> ()), Seq(1f -> (), 2f -> ()))))
