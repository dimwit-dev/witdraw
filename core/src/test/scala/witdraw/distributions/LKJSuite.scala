package witdraw
package distributions

import dimwit.*

class LKJSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  // the NUTS test runs a warmup and draws, longer than munit's default 30 s
  override val munitTimeout = scala.concurrent.duration.Duration(180, "s")

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

  test("NUTS samples LKJ through Bijector.correlationMatrix"):
    // the marginal of each correlation r is (r + 1)/2 ~ Beta(a, a), with
    // a = η - 1 + d/2, so E[r] = 0 and Var[r] = 1 / (2a + 1)
    object Corr extends Site[Tensor2[D, Prime[D], Float32]]
    trait DZ derives Label
    val (eta, d) = (2f, 3)
    val latent = Latent.discover(
      Constraint(Corr, Bijector.correlationMatrix[DZ, D])
    )(Corr ~ lkj(eta, d))
    val (warmupKey, sampleKey) = Key(7).split2()
    val adapted =
      inference.NUTS.adapt(latent, latent.init, warmupKey, numSteps = 500)
    val (draws, infos) = adapted.sampler
      .run(adapted.position, sampleKey)
      .take(1000)
      .toList
      .unzip
    assertEquals(infos.count(_.isDivergent), 0)
    val a = eta - 1 + d / 2f
    trait Draw derives Label
    val stacked = stack(draws.map(_(Corr)), Axis[Draw])
    val means = stacked.mean(Axis[Draw])
    val variances = (stacked -! means).pow(2f).mean(Axis[Draw])
    for (i, j) <- Seq((1, 0), (2, 0), (2, 1)) do
      def at(m: Tensor2[D, Prime[D], Float32]) =
        m(Axis[D].at(i))(Axis[Prime[D]].at(j)).item
      assertEqualsFloat(at(means), 0f, 0.05f, s"mean of r($i, $j)")
      assertEqualsFloat(at(variances), 1f / (2 * a + 1), 0.03f, s"variance of r($i, $j)")
