package witdraw
package inference

import dimwit.*
import dimwit.stats.*
import scala.concurrent.{Await, Future, blocking}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration

class NUTSSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  // sampling thousands of draws step by step takes close to munit's default 30 s
  override val munitTimeout = scala.concurrent.duration.Duration(120, "s")

  object Narrow extends Site[Tensor0[Float32]]
  object Wide extends Site[Tensor0[Float32]]

  // two independent normals whose scales differ by a factor of 100
  def model(using ProbContext) =
    (narrow = Narrow ~ Normal(1f, 0.1f), wide = Wide ~ Normal(-5f, 10f))

  final lazy val latent = Latent.discover(model)

  def moments(draws: Seq[Float]): (Float, Float) =
    val mean = draws.sum / draws.size
    (mean, draws.map(d => (d - mean) * (d - mean)).sum / draws.size)

  test("the adapted diagonal mass matrix estimates the variances"):
    val adapted = NUTS.adapt(latent, Key(1), latent.init, numSteps = 1000)
    val variances = adapted.inverseMassMatrix match
      case InverseMassMatrix.Diagonal(v) => v
      case other                         => fail(s"not diagonal: $other")
    // the warmup shrinks the estimate slightly towards 1e-3
    assertEqualsFloat(variances(Narrow).item, 0.01f, 0.004f)
    assertEqualsFloat(variances(Wide).item, 100f, 40f)
    assert(adapted.stepSize > 0.1, s"step size ${adapted.stepSize}")

  test("NUTS samples the target with the adapted parameters"):
    val (warmupKey, sampleKey) = Key(2).split2()
    val adapted = NUTS.adapt(latent, warmupKey, latent.init)
    val chain = adapted.sample(sampleKey, 2000)
    val (draws, infos) = (chain.positions, chain.infos)
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
    val adapted = NUTS.adapt(latent, Key(3), latent.init, dense = true)
    assert(adapted.inverseMassMatrix.isInstanceOf[InverseMassMatrix.Dense[?]])
    val draws = adapted.sample(Key(4), 1000)(Narrow).map(_.item)
    assertEqualsFloat(moments(draws)._1, 1f, 0.03f)

  test("a given diagonal mass matrix is used by the sampler"):
    val variances = latent.initWith(
      Narrow := Tensor0(0.01f),
      Wide := Tensor0(100f)
    )
    val nuts = NUTS(latent, 0.5, InverseMassMatrix.Diagonal(variances))
    val infos = nuts.iterate(Key(5), latent.init).take(500).map(_._2).toList
    assertEquals(infos.count(_.isDivergent), 0)

  test("sample gives the same draws as run, in one compiled loop"):
    val nuts = NUTS(latent, 0.1)
    val looped = nuts.iterate(Key(6), latent.init).take(30).toList
    val compiled = nuts.sample(Key(6), latent.init, 30)
    assertEquals(compiled.size, 30)
    for ((p1, i1), (p2, i2)) <- looped.zip(compiled.draws) do
      assertEqualsFloat(p1(Narrow).item, p2(Narrow).item, 1e-4f)
      assertEqualsFloat(p1(Wide).item, p2(Wide).item, 1e-3f)
      assertEquals(i1.numIntegrationSteps, i2.numIntegrationSteps)

  // starting positions spread out around the target
  def starts(n: Int) = (0 until n).map: i =>
    latent.initWith(Narrow := Tensor0(0.5f + 0.3f * i), Wide := Tensor0(-20f + 10f * i))

  // one chain: a warmup, then sampling
  def adaptedChain(numDraws: Int, dense: Boolean = false)(key: Key, start: latent.Params) =
    val (warmupKey, sampleKey) = key.split2()
    val adapted = NUTS.adapt(latent, warmupKey, start, dense = dense)
    (adapted, adapted.sample(sampleKey, numDraws))

  // runs `f` for each input in a Future of its own, as users run chains
  def inFutures[A, B](inputs: Seq[A])(f: A => B): Seq[B] =
    Await.result(Future.sequence(inputs.map(a => Future(blocking(f(a))))), Duration.Inf)

  test("chains in Futures run one chain per input"):
    val nuts = NUTS(latent, 0.1)
    val chains = inFutures(Key(7).split(3).zip(starts(3)))((k, s) => nuts.sample(k, s, 50))
    assertEquals(chains.map(_.size), Seq(50, 50, 50))
    val firsts = chains.map(_(Wide).head.item)
    assertEquals(firsts.distinct.size, 3, s"chains start alike: $firsts")

  test("chains in Futures give the same draws as chains run one after another"):
    val inputs = Key(8).split(3).zip(starts(3))
    def chain(k: Key, s: latent.Params) = adaptedChain(20)(k, s)._2
    val parallel = inFutures(inputs)(chain)
    val sequential = inputs.map(chain)
    for (p, q) <- parallel.flatMap(_.draws).zip(sequential.flatMap(_.draws)) do
      assertEqualsFloat(p._1(Narrow).item, q._1(Narrow).item, 1e-6f)
      assertEqualsFloat(p._1(Wide).item, q._1(Wide).item, 1e-5f)

  test("adapted chains are tuned each on its own, and agree"):
    val runs = inFutures(Key(9).split(4).zip(starts(4)))(adaptedChain(1000))
    val stepSizes = runs.map(_._1.stepSize)
    assertEquals(stepSizes.distinct.size, 4, s"step sizes $stepSizes")
    val chains = runs.map(_._2)
    assertEquals(chains.map(_.size), Seq(1000, 1000, 1000, 1000))
    for site <- Seq(Narrow, Wide) do
      val rhat = Diagnostics.rhat(chains, site).item
      val ess = Diagnostics.ess(chains, site).item
      assert(rhat < 1.02, s"rhat of ${site.name}: $rhat")
      assert(ess > 1000, s"ess of ${site.name}: $ess")
    val (narrowMean, narrowVar) = moments(chains.flatMap(_(Narrow)).map(_.item))
    assertEqualsFloat(narrowMean, 1f, 0.02f)
    assertEqualsFloat(narrowVar, 0.01f, 0.002f)
    assertEquals(chains.map(_.numDivergent).sum, 0)

  test("adapted chains work with a dense mass matrix"):
    val runs = inFutures(Key(10).split(2).zip(starts(2)))(adaptedChain(500, dense = true))
    assert(runs.forall(_._1.inverseMassMatrix.isInstanceOf[InverseMassMatrix.Dense[?]]))
    val narrow = runs.flatMap(_._2(Narrow)).map(_.item)
    assertEqualsFloat(moments(narrow)._1, 1f, 0.03f)

  test("the chains of a latent share the compiled warmup and sampling loop"):
    // a model of its own, so that nothing is compiled for it yet
    val fresh = Latent.discover(model)
    def chain(key: Key, start: fresh.Params) =
      val (warmupKey, sampleKey) = key.split2()
      val adapted = NUTS.adapt(fresh, warmupKey, start, numSteps = 200)
      (adapted.stepSize, adapted.sample(sampleKey, 50)(Wide).last.item)
    val t0 = System.nanoTime()
    val first = chain(Key(12), fresh.init)
    val t1 = System.nanoTime()
    val second = chain(Key(12), fresh.init)
    val t2 = System.nanoTime()
    assertEquals(second, first)
    // the first chain compiles the warmup and the loop, the second does not
    assert(t2 - t1 < (t1 - t0) / 2, s"first ${(t1 - t0) / 1e9}s, second ${(t2 - t1) / 1e9}s")
