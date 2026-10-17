package witdraw

import dimwit.*
import dimwit.stats.*
import scala.collection.mutable
import witdraw.reparam.Bijector
import dimwit.tensortree.TensorTree

class ProbContextSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  trait Sample derives Label

  object Mu extends Site[Tensor0[Float32]]
  object Sigma extends Site[Tensor0[Float32]]
  object Y extends Site[Tensor1[Sample, Float32]]

  val n = 3

  def prior(using ProbContext) =
    (mu = Mu ~ Normal(0f, 10f), sigma = Sigma ~ HalfNormal(0f, 1f))

  def likelihood(mu: Tensor0[Float32], sigma: Tensor0[Float32])(using
      ProbContext
  ) =
    Y ~ Normal.isotropic(Tensor1(Axis[Sample] -> n).fill(0f) +! mu, sigma)

  def model(using ProbContext) =
    val p = prior
    val y = likelihood(p.mu, p.sigma)
    (mu = p.mu, sigma = p.sigma, y = y)

  lazy val yData = Tensor1(Axis[Sample]).fromArray(Array(0.5f, -1f, 2f))

  def manualLogProb(mu: Float, sigma: Float): Float =
    val muT = Tensor0(mu)
    val sigmaT = Tensor0(sigma)
    (Normal(Tensor0(0f), Tensor0(10f)).logProb(muT).asFloat +
      HalfNormal(Tensor0(0f), Tensor0(1f)).logProb(sigmaT).asFloat +
      Normal
        .isotropic(Tensor1(Axis[Sample] -> n).fill(0f) +! muT, sigmaT)
        .logProb(yData)
        .asFloat).item

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

  test("ScoredSampler returns the same log-density as ModelScorer"):
    val (r, lp) = ScoredSampler.run(Key(3))(observe(Y := yData)(model))
    assertEqualsFloat(lp.item, manualLogProb(r.mu.item, r.sigma.item), 1e-3f)

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
    def recordSites[A](block: ProbProg[A])(using ctx: ProbContext): A =
      block(using
        new Handler(ctx):
          override def sample[Ten <: Tensor[?, ?]](
              s: SampleStatement[Ten]
          ): Ten =
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
    intercept[IllegalArgumentException]:
      ForwardSampler.run(Key(1)):
        (Mu ~ Normal(0f, 1f), Mu ~ Normal(0f, 1f))

  test("withTrace records every site, its value and whether it was observed"):
    val (r, tr) =
      ForwardSampler.run(Key(1))(withTrace(observe(Y := yData)(model)))
    assertEquals(tr.sites, Seq(Mu, Sigma, Y))
    assertEquals(tr.latentSites, Seq(Mu, Sigma))
    assertEquals(tr.observedSites, Seq(Y))
    assertEquals(tr(Mu).item, r.mu.item)
    val total = tr.entries.map(_.logProb).reduce(_ + _)
    assertEqualsFloat(total.item, manualLogProb(r.mu.item, r.sigma.item), 1e-3f)

  lazy val latent = Latent.discover(Key(1), Constraint(Sigma, Bijector.exp)):
    observe(Y := yData)(model)

  test("Latent finds the latent sites and starts from the traced values"):
    assertEquals(latent.sites, Seq(Mu, Sigma))
    val (_, tr) =
      ForwardSampler.run(Key(1))(withTrace(observe(Y := yData)(model)))
    assertEqualsFloat(latent.init(Mu).item, tr(Mu).item, 1e-6f)
    assertEqualsFloat(latent.init(Sigma).item, tr(Sigma).item, 1e-5f)

  test("Latent log-densities: model density, plus the Jacobian for sampling"):
    val p = latent.initWith(Mu := Tensor0(0.3f), Sigma := Tensor0(1.5f))
    assertEqualsFloat(p(Sigma).item, 1.5f, 1e-5f)
    assertEqualsFloat(
      latent.modelLogDensity(p).item,
      manualLogProb(0.3f, 1.5f),
      1e-4f
    )
    // sigma = exp(z), so log |d sigma / dz| = z = log sigma
    assertEqualsFloat(
      latent.logDensity(p).item,
      manualLogProb(0.3f, 1.5f) + math.log(1.5).toFloat,
      1e-4f
    )

  test("Latent.Params is a tensor tree, so it can be differentiated"):
    val p = latent.initWith(Mu := Tensor0(0f), Sigma := Tensor0(1f))
    val grad = Autodiff.grad(latent.modelLogDensity)(p).value
    // d/dmu at mu = 0, sigma = 1: sum(y) - mu / 100
    assertEqualsFloat(grad(Mu).item, 1.5f, 1e-4f)

  test(
    "Latent errors: unknown site, constraint on observed site, discrete site"
  ):
    intercept[NoSuchElementException](latent.init(Y))
    intercept[IllegalArgumentException]:
      Latent.discover(Key(1), Constraint(Y, Bijector.exp[Tuple1[Sample]])):
        observe(Y := yData)(model)
    // Binomial samples are float32 at runtime, so the check must use the
    // site's declared type
    object Count extends Site[Tensor0[Int32]]
    val e = intercept[IllegalArgumentException]:
      Latent.discover(Key(1))(Count ~ Binomial(3, Prob(Tensor0(0.5f))))
    assert(e.getMessage.contains("'Count' is Int32"), e.getMessage)

  trait Comp derives Label
  trait CompZ derives Label
  object Weights extends Site[Tensor1[Comp, Float32]]
  object Locs extends Site[Tensor1[Comp, Float32]]

  def mixturePrior(using ProbContext) =
    (
      weights = Weights ~ Dirichlet(Tensor1(Axis[Comp] -> 3).fill(1f)),
      locs = Locs ~ Normal(
        Tensor1(Axis[Comp] -> 3).fill(0f),
        Tensor1(Axis[Comp] -> 3).fill(10f)
      )
    )

  final lazy val mixture = Latent.discover(
    Key(5),
    Constraint(Weights, Bijector.simplexStickBreaking[CompZ, Comp](3)),
    Constraint(Locs, Bijector.ordered[Comp](3))
  )(mixturePrior)

  test("a constraint may change the shape: simplex from K-1 coordinates"):
    val weights = Tensor1(Axis[Comp]).fromArray(Array(0.2f, 0.5f, 0.3f))
    val locs = Tensor1(Axis[Comp]).fromArray(Array(-1f, 0f, 2f))
    val p = mixture.initWith(Weights := weights, Locs := locs)
    assertEqualsFloat(p(Weights).sum.item, 1f, 1e-5f)
    assertEqualsFloat(p(Weights).slice(Axis[Comp].at(1)).item, 0.5f, 1e-5f)
    val expected = ModelScorer.run:
      observe(Weights := weights, Locs := locs)(mixturePrior)
    assertEqualsFloat(mixture.modelLogDensity(p).item, expected.item, 1e-3f)
    // the unconstrained parameters have K-1 coordinates for the weights
    val leafSizes = TensorTree[mixture.Params]
      .mapLeaves(
        p,
        [T <: Tuple, V] =>
          (_: Labels[T]) ?=> (t: Tensor[T, V]) => t.shape.dimensions.product
      )
      .toList
    assertEquals(leafSizes, List(2, 3))

  test("a start value outside its constraint is an error, with a hint"):
    val unordered = Tensor1(Axis[Comp]).fromArray(Array(1f, -1f, 0f))
    val e =
      intercept[IllegalArgumentException](mixture.initWith(Locs := unordered))
    assert(e.getMessage.contains("given value of 'Locs'"), e.getMessage)
    assert(e.getMessage.contains("latent.initWith(Locs := ...)"), e.getMessage)
