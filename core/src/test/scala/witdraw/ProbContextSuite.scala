package witdraw

import dimwit.*
import dimwit.stats.*
import scala.collection.mutable

import NormalModel.*

class ProbContextSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  test("ForwardSampler is deterministic in the key and composes sub-models"):
    val a = ForwardSampler.run(Key(1))(model)
    val b = ForwardSampler.run(Key(1))(model)
    val c = ForwardSampler.run(Key(2))(model)
    assertEquals(a.mu.item, b.mu.item)
    assertNotEquals(a.mu.item, c.mu.item)
    assert(a.sigma.item > 0f)

  test("a model can be stored as a ProbProg value and run later"):
    val stored: ProbProg[Tensor0[Float32]] = model.mu
    assertEquals(
      ForwardSampler.run(Key(1))(stored).item,
      ForwardSampler.run(Key(1))(model).mu.item
    )

  test("observe fixes a site for the forward sampler"):
    val r = ForwardSampler.run(Key(1))(observe(Sigma := Tensor0(0.25f))(model))
    assertEquals(r.sigma.item, 0.25f)

  test("ModelScorer sums the log-densities of all sites"):
    val lp = ModelScorer.run:
      observe(Mu := Tensor0(0.3f), Sigma := Tensor0(1.5f), Y := yData):
        model
    assertEqualsFloat(lp.item, manualLogProb(0.3f, 1.5f), 1e-4f)

  test("observe nests, also around sub-models"):
    val lp = ModelScorer.run:
      observe(Mu := Tensor0(0.3f), Sigma := Tensor0(1.5f)):
        val p = prior
        observe(Y := yData)(likelihood(p.mu, p.sigma))
    assertEqualsFloat(lp.item, manualLogProb(0.3f, 1.5f), 1e-4f)

  test("the log-density is differentiable through handlers"):
    val logDensity = (p: (mu: Tensor0[Float32], sigma: Tensor0[Float32])) =>
      ModelScorer.run:
        observe(Y := yData, Mu := p.mu, Sigma := p.sigma)(model)
    val grad =
      Autodiff.grad(logDensity)((mu = Tensor0(0f), sigma = Tensor0(1f)))
    // d/dmu at mu = 0, sigma = 1: sum(y) - mu / 100
    assertEqualsFloat(grad.value.mu.item, 1.5f, 1e-4f)

  test("sites are identified by object, not by name"):
    object Other:
      object Mu extends Site[Tensor0[Float32]]
    val r = ForwardSampler.run(Key(1)):
      observe(Mu := Tensor0(5f)):
        (mine = Mu ~ Normal(0f, 1f), other = Other.Mu ~ Normal(0f, 1f))
    assertEquals(r.mine.item, 5f)
    assertNotEquals(r.other.item, 5f)

  test("a custom handler sees every statement of its block"):
    val seen = mutable.ListBuffer.empty[String]
    def recordSites[A](block: ProbProg[A])(using outer: ProbContext): A =
      block(using
        new ProbContext:
          def sample[T <: Tuple, V](
              s: SampleStatement[T, V]
          ): Tensor[T, V] =
            seen += s"${s.site.name}${
                if s.observed.isDefined then "=obs" else ""
              }"
            outer.sample(s)
      )
    ForwardSampler.run(Key(1))(recordSites(observe(Y := yData)(model)))
    assertEquals(seen.toList, List("Mu", "Sigma", "Y=obs"))

  test("errors: unknown observation, double observe"):
    intercept[IllegalArgumentException]:
      ForwardSampler.run(Key(1))(observe(Y := yData)(prior))
    intercept[IllegalArgumentException]:
      ForwardSampler.run(Key(1))(
        observe(Y := yData)(observe(Y := yData)(model))
      )
    intercept[IllegalArgumentException]:
      ForwardSampler.run(Key(1))(
        observe(Mu := Tensor0(0f), Mu := Tensor0(1f))(model)
      )

  test("errors: free site in ModelScorer, site sampled twice"):
    intercept[IllegalArgumentException]:
      ModelScorer.run(observe(Y := yData)(model))
    def twice(using ProbContext) = (Mu ~ Normal(0f, 1f), Mu ~ Normal(0f, 1f))
    intercept[IllegalArgumentException]:
      ForwardSampler.run(Key(1))(observe(Mu := Tensor0(0f))(twice))
    intercept[IllegalArgumentException]:
      ForwardSampler.run(Key(1))(withTrace(twice))

  test("withTrace records every site, its value and whether it was observed"):
    val (r, tr) =
      ForwardSampler.run(Key(1))(withTrace(observe(Y := yData)(model)))
    assertEquals(tr.sites, Seq(Mu, Sigma, Y))
    assertEquals(tr.latentSites, Seq(Mu, Sigma))
    assertEquals(tr.observedSites, Seq(Y))
    assertEquals(tr(Mu).item, r.mu.item)
    val total = tr.entries.map(_.logProb).reduce(_ + _)
    assertEqualsFloat(total.item, manualLogProb(r.mu.item, r.sigma.item), 1e-3f)
