package witdraw
package inference

class ChainSuite extends munit.FunSuite:

  // positions are plain floats here; the info says whether a draw diverged
  case class Info(divergent: Boolean)
  given SamplerInfo[Info] = _.divergent

  val chain = Chain(Seq(1f -> Info(false), 2f -> Info(true), 3f -> Info(true)))

  test("size, positions and values"):
    assertEquals(chain.size, 3)
    assertEquals(chain.positions, Seq(1f, 2f, 3f))
    assertEquals(chain.values(_ * 2), Seq(2f, 4f, 6f))

  test("divergences come from the sampler's info"):
    assertEquals(chain.divergent, Seq(false, true, true))
    assertEquals(chain.numDivergent, 2)

  test("draws without info have no divergences"):
    assertEquals(Chain(Seq(1f -> (), 2f -> ())).numDivergent, 0)
