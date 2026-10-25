package witdraw

import dimwit.*
import dimwit.stats.*
import dimwit.tensortree.TensorTree

import NormalModel.*

class LatentSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  lazy val latent = Latent.discover(Constraint(Sigma, Bijector.exp)):
    observe(Y := yData)(model)

  test("Latent finds the latent sites and starts from the traced values"):
    assertEquals(latent.sites, Seq(Mu, Sigma))
    // discover traces the model with the fixed key Key(0)
    val (_, tr) =
      ForwardSampler.run(Key(0))(withTrace(observe(Y := yData)(model)))
    assertEqualsFloat(latent.init(Mu).item, tr(Mu).item, 1e-6f)
    assertEqualsFloat(latent.init(Sigma).item, tr(Sigma).item, 1e-5f)

  test("initRandom draws unconstrained values between -radius and radius"):
    val starts = Key(1).split(20).map(latent.initRandom(_))
    val mus = starts.map(_(Mu).item)
    val sigmas = starts.map(_(Sigma).item)
    assertEquals(mus.distinct.size, 20)
    assert(mus.forall(m => m > -2f && m < 2f), s"mu $mus")
    // sigma is exp of its unconstrained value
    assert(sigmas.forall(s => s > math.exp(-2).toFloat && s < math.exp(2).toFloat), s"sigma $sigmas")
    val narrow = latent.initRandom(Key(2), radius = 0.1f)
    assert(math.abs(narrow(Mu).item) < 0.1f)

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

  test("Latent errors: unknown site, constraint on observed site"):
    intercept[NoSuchElementException](latent.init(Y))
    intercept[IllegalArgumentException]:
      Latent.discover(Constraint(Y, Bijector.exp[Tuple1[Sample]])):
        observe(Y := yData)(model)

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
    Constraint(Weights, Bijector.simplexStickBreaking[CompZ, Comp]),
    Constraint(Locs, Bijector.ordered[Comp])
  )(mixturePrior)

  test("a constraint may change the shape: simplex from K-1 coordinates"):
    val weights = Tensor1(Axis[Comp]).fromArray(Array(0.2f, 0.5f, 0.3f))
    val locs = Tensor1(Axis[Comp]).fromArray(Array(-1f, 0f, 2f))
    val p = mixture.initWith(Weights := weights, Locs := locs)
    assertEqualsFloat(p(Weights).sum.item, 1f, 1e-5f)
    assertEqualsFloat(p(Weights).select(Axis[Comp].at(1)).item, 0.5f, 1e-5f)
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

  test("initRandom works when the traced value lies outside its constraint"):
    // the traced value is about -10, which the exp constraint cannot reach
    object Neg extends Site[Tensor0[Float32]]
    val constrained = Latent.discover(Constraint(Neg, Bijector.exp)):
      Neg ~ Normal(-10f, 0.1f)
    intercept[IllegalArgumentException](constrained.init)
    val start = constrained.initRandom(Key(3))
    assert(start(Neg).item > 0f)

  test("a start value outside its constraint is an error, with a hint"):
    val unordered = Tensor1(Axis[Comp]).fromArray(Array(1f, -1f, 0f))
    val e =
      intercept[IllegalArgumentException](mixture.initWith(Locs := unordered))
    assert(e.getMessage.contains("given value of 'Locs'"), e.getMessage)
    assert(e.getMessage.contains("latent.initWith(Locs := ...)"), e.getMessage)
