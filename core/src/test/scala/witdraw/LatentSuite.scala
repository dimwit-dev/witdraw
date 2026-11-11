package witdraw

import dimwit.*
import dimwit.stats.*
import dimwit.tensortree.TensorTree
import witdraw.distributions.Ordered

import NormalModel.*

class LatentSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  lazy val latent = Latent.discover:
    observe(Y := yData)(model)

  test("Latent finds the latent sites and starts from the traced values"):
    assertEquals(latent.sites, Seq(Mu, Sigma))
    // discover traces the model with the fixed key Key(0)
    val (_, tr) =
      ForwardSampler.run(Key(0))(withTrace(observe(Y := yData)(model)))
    assertEqualsFloat(latent.result(latent.init).mu.item, tr(Mu).item, 1e-6f)
    assertEqualsFloat(latent.result(latent.init).sigma.item, tr(Sigma).item, 1e-5f)

  test("initRandom draws unconstrained values between -radius and radius"):
    val starts = Key(1).split(20).map(latent.initRandom(_))
    val mus = starts.map(latent.result(_).mu.item)
    val sigmas = starts.map(latent.result(_).sigma.item)
    assertEquals(mus.distinct.size, 20)
    assert(mus.forall(m => m > -2f && m < 2f), s"mu $mus")
    // sigma is exp of its unconstrained value
    assert(sigmas.forall(s => s > math.exp(-2).toFloat && s < math.exp(2).toFloat), s"sigma $sigmas")
    val narrow = latent.initRandom(Key(2), radius = 0.1f)
    assert(math.abs(latent.result(narrow).mu.item) < 0.1f)

  test("Latent log-densities: model density, plus the Jacobian for sampling"):
    val p = latent.initWith(Mu := Tensor0(0.3f), Sigma := Tensor0(1.5f))
    assertEqualsFloat(latent.result(p).sigma.item, 1.5f, 1e-5f)
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
    // the gradient is not a point of the posterior: read its entry for Mu by site
    assertEqualsFloat(latent.values(grad)(Mu).item, 1.5f, 1e-4f)

  test("result is the model's own result at the parameters, with the data"):
    val p = latent.initWith(Mu := Tensor0(0.3f), Sigma := Tensor0(1.5f))
    val r = latent.result(p)
    assertEqualsFloat(r.mu.item, 0.3f, 1e-5f)
    assertEqualsFloat(r.sigma.item, 1.5f, 1e-5f)
    assertEquals(r.y.toArray.toSeq, yData.toArray.toSeq)

  test("the samples of a chain are the model's results at each draw"):
    val starts = Key(4).split(5).map(latent.initRandom(_))
    val chain = inference.Chain(starts.map(_ -> ()))
    assertEquals(chain.samples.map(_.mu.item), starts.map(latent.result(_).mu.item))
    assertEquals(chain.samples.map(_.y.toArray.toSeq).distinct, Seq(yData.toArray.toSeq))

  test("a result that cannot be compiled is computed eagerly"):
    // `.item` needs a concrete value, so the model cannot be traced by jit
    val eager = Latent.discover:
      val mu = Mu ~ Normal(0f, 1f)
      (mu = mu, positive = Tensor0(if mu.item > 0 then 1f else 0f))
    val p = eager.initWith(Mu := Tensor0(0.5f))
    assertEquals(eager.result(p).positive.item, 1f)
    assertEquals(eager.result(eager.initWith(Mu := Tensor0(-0.5f))).positive.item, 0f)

  test("a site that is not latent is an error"):
    intercept[NoSuchElementException](latent.initWith(Y := yData))

  trait Comp derives Label
  object Weights extends Site[Tensor1[Comp, Float32]]
  object Locs extends Site[Tensor1[Comp, Float32]]

  def mixturePrior(using ProbContext) =
    (
      weights = Weights ~ Dirichlet(Tensor1(Axis[Comp] -> 3).fill(1f)),
      locs = Locs ~ Ordered(Normal(0f, 10f), Axis[Comp] -> 3)
    )

  final lazy val mixture = Latent.discover(mixturePrior)

  test("a bijector may change the shape: simplex from K-1 coordinates"):
    val weights = Tensor1(Axis[Comp]).fromArray(Array(0.2f, 0.5f, 0.3f))
    val locs = Tensor1(Axis[Comp]).fromArray(Array(-1f, 0f, 2f))
    val p = mixture.initWith(Weights := weights, Locs := locs)
    assertEqualsFloat(mixture.result(p).weights.sum.item, 1f, 1e-5f)
    assertEqualsFloat(mixture.result(p).weights.select(Axis[Comp].at(1)).item, 0.5f, 1e-5f)
    val expected = ModelScorer.run:
      observe(Weights := weights, Locs := locs)(mixturePrior)
    assertEqualsFloat(mixture.modelLogDensity(p).item, expected.item, 1e-3f)
    // the unconstrained parameters are one vector, with K-1 = 2 coordinates
    // for the weights and 3 for the locations
    val leafSizes = TensorTree[mixture.Params]
      .mapLeaves(
        p,
        [T <: Tuple, V] =>
          (_: Labels[T]) ?=> (t: Tensor[T, V]) => t.shape.dimensions.product
      )
      .toList
    assertEquals(leafSizes, List(5))

  test("a start value outside its support is an error, with a hint"):
    val unordered = Tensor1(Axis[Comp]).fromArray(Array(1f, -1f, 0f))
    val e =
      intercept[IllegalArgumentException](mixture.initWith(Locs := unordered))
    assert(e.getMessage.contains("given value of 'Locs'"), e.getMessage)
    assert(e.getMessage.contains("latent.initWith(Locs := ...)"), e.getMessage)
