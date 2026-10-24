package witdraw
package inference

import dimwit.*
import dimwit.stats.*

class NUTSSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  object Narrow extends Site[Tensor0[Float32]]
  object Wide extends Site[Tensor0[Float32]]

  // two independent normals whose scales differ by a factor of 100
  def model(using ProbContext) =
    (narrow = Narrow ~ Normal(1f, 0.1f), wide = Wide ~ Normal(-5f, 10f))

  lazy val latent = Latent.discover()(model)

  def moments(draws: List[Float]): (Float, Float) =
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
      .run(adapted.position, sampleKey)
      .take(2000)
      .toList
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

