package witdraw
package inference

import dimwit.*
import dimwit.stats.*

class NUTSSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  // sampling thousands of draws step by step takes close to munit's default 30 s
  override val munitTimeout = scala.concurrent.duration.Duration(120, "s")

  object Narrow extends Site[Tensor0[Float32]]
  object Wide extends Site[Tensor0[Float32]]

  // two independent normals whose scales differ by a factor of 100
  def model(using ProbContext) =
    (narrow = Narrow ~ Normal(1f, 0.1f), wide = Wide ~ Normal(-5f, 10f))

  lazy val latent = Latent.discover()(model)

  def moments(draws: Seq[Float]): (Float, Float) =
    val mean = draws.sum / draws.size
    (mean, draws.map(d => (d - mean) * (d - mean)).sum / draws.size)

  test("the adapted diagonal mass matrix estimates the variances"):
    val adapted = NUTS.adapt(latent, latent.init, Key(1), numSteps = 1000)
    val variances = adapted.inverseMassMatrix match
      case InverseMassMatrix.Diagonal(v) => v
      case other                         => fail(s"not diagonal: $other")
    // the warmup shrinks the estimate slightly towards 1e-3
    assertEqualsFloat(variances(Narrow).item, 0.01f, 0.004f)
    assertEqualsFloat(variances(Wide).item, 100f, 40f)
    assert(adapted.stepSize > 0.1, s"step size ${adapted.stepSize}")

  test("NUTS samples the target with the adapted parameters"):
    val (warmupKey, sampleKey) = Key(2).split2()
    val adapted = NUTS.adapt(latent, latent.init, warmupKey)
    val (draws, infos) = adapted.sampler
      .sample(adapted.position, sampleKey, 2000)
      .unzip
    val (narrowMean, narrowVar) = moments(draws.map(_(Narrow).item))
    val (wideMean, wideVar) = moments(draws.map(_(Wide).item))
    assertEqualsFloat(narrowMean, 1f, 0.02f)
    assertEqualsFloat(narrowVar, 0.01f, 0.002f)
    assertEqualsFloat(wideMean, -5f, 2f)
    assertEqualsFloat(wideVar, 100f, 20f)
    val acceptance = infos.map(_.acceptanceRate).sum / infos.size
    // dual averaging ends on the averaged step size, which tends to be a little
    // smaller than the target needs: the acceptance rate ends above it
    assert(acceptance > 0.7, s"acceptance rate $acceptance")
    assertEquals(infos.count(_.isDivergent), 0)

  test("a dense mass matrix can be adapted"):
    val adapted = NUTS.adapt(latent, latent.init, Key(3), dense = true)
    assert(adapted.inverseMassMatrix.isInstanceOf[InverseMassMatrix.Dense[?]])
    val draws = adapted.sampler
      .run(adapted.position, Key(4))
      .take(1000)
      .map(_._1(Narrow).item)
      .toList
    assertEqualsFloat(moments(draws)._1, 1f, 0.03f)

  test("a given diagonal mass matrix is used by the sampler"):
    val variances = latent.initWith(
      Narrow := Tensor0(0.01f),
      Wide := Tensor0(100f)
    )
    val nuts = NUTS(latent, 0.5, InverseMassMatrix.Diagonal(variances))
    val infos = nuts.run(latent.init, Key(5)).take(500).map(_._2).toList
    assertEquals(infos.count(_.isDivergent), 0)

  test("sample gives the same draws as run, in one compiled loop"):
    val nuts = NUTS(latent, 0.1)
    val looped = nuts.run(latent.init, Key(6)).take(30).toList
    val compiled = nuts.sample(latent.init, Key(6), 30)
    assertEquals(compiled.size, 30)
    for ((p1, i1), (p2, i2)) <- looped.zip(compiled) do
      assertEqualsFloat(p1(Narrow).item, p2(Narrow).item, 1e-4f)
      assertEqualsFloat(p1(Wide).item, p2(Wide).item, 1e-3f)
      assertEquals(i1.numIntegrationSteps, i2.numIntegrationSteps)

  // starting positions spread out around the target
  def starts(n: Int) = (0 until n).map: i =>
    latent.initWith(Narrow := Tensor0(0.5f + 0.3f * i), Wide := Tensor0(-20f + 10f * i))

  test("sampleChains runs one chain per starting position"):
    val nuts = NUTS(latent, 0.1)
    val chains = nuts.sampleChains(starts(3), Key(7), 50)
    assertEquals(chains.map(_.size), Seq(50, 50, 50))
    val firsts = chains.map(_.head._1(Wide).item)
    assertEquals(firsts.distinct.size, 3, s"chains start alike: $firsts")

  test("sampleChains matches sample for each chain's key"):
    val nuts = NUTS(latent, 0.1)
    val key = Key(8)
    val chains = nuts.sampleChains(starts(2), key, 20)
    val keys = key.split(2)
    for c <- 0 until 2 do
      val single = nuts.sample(starts(2)(c), keys(c), 20)
      for ((p1, i1), (p2, i2)) <- single.zip(chains(c)) do
        assertEqualsFloat(p1(Narrow).item, p2(Narrow).item, 1e-4f)
        assertEqualsFloat(p1(Wide).item, p2(Wide).item, 1e-3f)
        assertEquals(i1.numIntegrationSteps, i2.numIntegrationSteps)

  test("adaptChains tunes each chain, and the chains agree"):
    val (warmupKey, sampleKey) = Key(9).split2()
    val adapted = NUTS.adaptChains(latent, starts(4), warmupKey)
    assertEquals(adapted.chains.size, 4)
    val stepSizes = adapted.chains.map(_.stepSize)
    assertEquals(stepSizes.distinct.size, 4, s"step sizes $stepSizes")
    val chains = adapted.sample(sampleKey, 1000)
    assertEquals(chains.map(_.size), Seq.fill(4)(1000))
    val positions = chains.map(_.map(_._1))
    for site <- Seq(Narrow, Wide) do
      val draws = positions.map(_.map(_(site)))
      val rhat = Diagnostics.rhat(draws).item
      val ess = Diagnostics.ess(draws).item
      assert(rhat < 1.02, s"rhat of ${site.name}: $rhat")
      assert(ess > 1000, s"ess of ${site.name}: $ess")
    val (narrowMean, narrowVar) = moments(positions.flatten.map(_(Narrow).item))
    assertEqualsFloat(narrowMean, 1f, 0.02f)
    assertEqualsFloat(narrowVar, 0.01f, 0.002f)
    assertEquals(chains.flatten.count(_._2.isDivergent), 0)

  test("adaptChains works with a dense mass matrix"):
    val adapted = NUTS.adaptChains(latent, starts(2), Key(10), dense = true)
    assert(adapted.chains.forall(_.inverseMassMatrix.isInstanceOf[InverseMassMatrix.Dense[?]]))
    val chains = adapted.sample(Key(11), 500)
    val narrow = chains.flatten.map(_._1(Narrow).item)
    assertEqualsFloat(moments(narrow)._1, 1f, 0.03f)

  test("a chains warmup compiles once, and later runs reuse it"):
    val warmup = NUTS.chainsWarmup(latent, numSteps = 200)
    def run() =
      val adapted = warmup.run(starts(2), Key(12))
      val draws = adapted.sample(Key(13), 50)
      (adapted.chains.map(_.stepSize), draws.map(_.last._1(Wide).item))
    val t0 = System.nanoTime()
    val first = run()
    val t1 = System.nanoTime()
    val second = run()
    val t2 = System.nanoTime()
    assertEquals(second, first)
    // the first run compiles the warmup and the loop, the second does not
    assert(t2 - t1 < (t1 - t0) / 2, s"first ${(t1 - t0) / 1e9}s, second ${(t2 - t1) / 1e9}s")
