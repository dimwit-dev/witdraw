package witdraw

import dimwit.*
import dimwit.stats.*
import dimwit.tensortree.TensorTree
import witdraw.distributions.LKJ

import NormalModel.*

/** A distribution of positive values, defined outside dimwit, with its support
  * given in its companion object.
  */
class Positive(scale: Float) extends Distribution[EmptyTuple, Float32]:
  private val inner = HalfNormal(Tensor0(0f), Tensor0(scale))
  def sample(k: Key) = inner.sample(k)
  def logProb(x: Tensor0[Float32]) = inner.logProb(x)

object Positive:
  given SupportOf[Positive, EmptyTuple, Float32] =
    _ => Support.GreaterThan(Tensor0(0f))

/** A subclass of HalfNormal without an instance of its own. */
class MyHalfNormal
    extends HalfNormal[EmptyTuple, Float32](Tensor0(0f), Tensor0(1f))

/** A subclass of HalfNormal with another support, given in its companion. */
class CappedHalfNormal
    extends HalfNormal[EmptyTuple, Float32](Tensor0(0f), Tensor0(1f))

object CappedHalfNormal:
  given SupportOf[CappedHalfNormal, EmptyTuple, Float32] =
    _ => Support.Interval(Tensor0(0f), Tensor0(5f))

trait Row derives Label
trait Col derives Label

/** Matrices with independent, uniform rows on the simplex. */
class StochasticMatrix(rows: Int, cols: Int)
    extends Distribution[(Row, Col), Float32]:
  private val row =
    Dirichlet[Col, Float32](Tensor1(Axis[Col] -> cols).fill(1f))
  def sample(k: Key) = k.splitvmap(Axis[Row] -> rows)(row.sample)
  def logProb(x: Tensor2[Row, Col, Float32]) =
    x.vmap(Axis[Row])(row.logProb).sum

object StochasticMatrix:
  given SupportOf[StochasticMatrix, (Row, Col), Float32] =
    _ => Support.Rowwise(Support.Simplex())

trait K derives Label

/** Increasing vectors, with a support that brings its own bijector. */
class Increasing(k: Int) extends Distribution[Tuple1[K], Float32]:
  private val steps = Normal.standardNormal(Shape(Axis[K] -> k))
  def sample(key: Key) = steps.sample(key).exp.cumsum(Axis[K])
  def logProb(x: Tensor1[K, Float32]) = steps.logProb(x)

object Increasing:
  given SupportOf[Increasing, Tuple1[K], Float32] =
    _ => Support.Custom("increasing", Bijector.ordered[K])

class SupportSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  /** The number of unconstrained values. */
  def numParams(latent: Latent)(p: latent.Params): Int =
    TensorTree[latent.Params]
      .mapLeaves(
        p,
        [T <: Tuple, V] =>
          (_: Labels[T]) ?=> (t: Tensor[T, V]) => t.shape.dimensions.product
      )
      .toList
      .sum

  def supportOf(site: SiteDef[?, ?])(model: ProbProg[Any]): Support[?, ?] =
    val (_, tr) = ForwardSampler.run(Key(0))(withTrace(model))
    tr.entry(site).get.support

  test("~ records the support of the distribution's class in the trace"):
    assertEquals(supportOf(Mu)(model), Support.Real())
    assert(supportOf(Sigma)(model).isInstanceOf[Support.GreaterThan[?, ?]])
    assertEquals(supportOf(Y)(model), Support.Real())

  test("a distribution typed as a plain Distribution has an unknown support"):
    val widened: Distribution[EmptyTuple, Float32] = HalfNormal(0f, 1f)
    assertEquals(supportOf(Sigma)(Sigma ~ widened), Support.Unknown())

  test("a given SupportOf in the companion of a distribution is used"):
    assert(
      supportOf(Sigma)(Sigma ~ Positive(1f))
        .isInstanceOf[Support.GreaterThan[?, ?]]
    )

  test("without constraints, a HalfNormal site is sampled on the log scale"):
    val latent = Latent.discover()(observe(Y := yData)(model))
    val p = latent.initWith(Mu := Tensor0(0.3f), Sigma := Tensor0(1.5f))
    assertEqualsFloat(p(Sigma).item, 1.5f, 1e-5f)
    // sigma = exp(z), so log |d sigma / dz| = z = log sigma
    assertEqualsFloat(
      (latent.logDensity(p) - latent.modelLogDensity(p)).item,
      math.log(1.5).toFloat,
      1e-4f
    )

  object P extends Site[Tensor0[Float32]]

  test("a Uniform site is sampled on the logit scale of its interval"):
    val latent = Latent.discover()(P ~ Uniform(Tensor0(0f), Tensor0(2f)))
    val p = latent.initWith(P := Tensor0(0.5f))
    assertEqualsFloat(p(P).item, 0.5f, 1e-5f)
    // x = 2·σ(z), so log |dx/dz| = log(2·u·(1-u)) with u = x / 2
    assertEqualsFloat(
      (latent.logDensity(p) - latent.modelLogDensity(p)).item,
      math.log(2 * 0.25 * 0.75).toFloat,
      1e-4f
    )

  trait Comp derives Label
  object Weights extends Site[Tensor1[Comp, Float32]]
  object Locs extends Site[Tensor1[Comp, Float32]]

  def mixturePrior(using ProbContext) =
    Weights ~ Dirichlet(Tensor1(Axis[Comp] -> 3).fill(1f))
    Locs ~ Normal(
      Tensor1(Axis[Comp] -> 3).fill(0f),
      Tensor1(Axis[Comp] -> 3).fill(10f)
    )

  test("a Dirichlet site is a simplex, and a constraint overrides a default"):
    val latent =
      Latent.discover(Constraint(Locs, Bijector.ordered[Comp]))(mixturePrior)
    val weights = Tensor1(Axis[Comp]).fromArray(Array(0.2f, 0.5f, 0.3f))
    val locs = Tensor1(Axis[Comp]).fromArray(Array(-1f, 0f, 2f))
    val p = latent.initWith(Weights := weights, Locs := locs)
    assertEqualsFloat(p(Weights).sum.item, 1f, 1e-5f)
    // K-1 = 2 coordinates for the simplex, 3 for the locations
    assertEquals(numParams(latent)(p), 5)
    // ordered: an unordered start is outside the constraint
    intercept[IllegalArgumentException]:
      latent.initWith(Locs := Tensor1(Axis[Comp]).fromArray(Array(1f, 0f, 2f)))

  trait AB derives Label
  object R extends Site[Tensor2[AB, Prime[AB], Float32]]

  test("an LKJ site is a correlation matrix"):
    val latent = Latent.discover():
      R ~ LKJ(Tensor0(2f), Shape(Axis[AB] -> 3, Axis[Prime[AB]] -> 3))
    // one unconstrained value per entry below the diagonal
    assertEquals(numParams(latent)(latent.init), 3)
    val r = latent.init(R)
    for i <- 0 until 3 do
      assertEqualsFloat(
        r(Axis[AB].at(i))(Axis[Prime[AB]].at(i)).item,
        1f,
        1e-5f
      )

  test("a latent site with an unknown support needs a constraint"):
    def widenedModel(using ProbContext) =
      val widened: Distribution[EmptyTuple, Float32] = HalfNormal(0f, 1f)
      Sigma ~ widened
    val e = intercept[IllegalArgumentException](Latent.discover()(widenedModel))
    assert(e.getMessage.contains("'Sigma' is latent"), e.getMessage)
    assert(e.getMessage.contains("Constraint(Sigma"), e.getMessage)
    // with a constraint, or observed, the unknown support does not matter
    Latent.discover(Constraint(Sigma, Bijector.exp))(widenedModel)
    ModelScorer.run(observe(Sigma := Tensor0(1f))(widenedModel))

  object X extends Site[Tensor0[Float32]]

  def boundedBySigma(using ProbContext) =
    val sigma = Sigma ~ HalfNormal(0f, 1f)
    X ~ Uniform(Tensor0(0f), sigma)

  test("a support may depend on another latent site"):
    val latent = Latent.discover()(boundedBySigma)
    // X = sigma·σ(z) lies in (0, sigma) for every sigma
    val p = latent.initWith(Sigma := Tensor0(3f), X := Tensor0(2.5f))
    assertEqualsFloat(p(Sigma).item, 3f, 1e-5f)
    assertEqualsFloat(p(X).item, 2.5f, 1e-4f)
    val v = latent.values(p)
    assertEqualsFloat(v(X).item, 2.5f, 1e-4f)
    // log-Jacobian: log sigma for Sigma = exp(z), and log(sigma·u·(1-u)) for
    // X, with u = x / sigma
    val u = 2.5 / 3
    assertEqualsFloat(
      (latent.logDensity(p) - latent.modelLogDensity(p)).item,
      (math.log(3) + math.log(3 * u * (1 - u))).toFloat,
      1e-3f
    )
    // the model's own density, at the values
    val expected = ModelScorer.run:
      observe(Sigma := Tensor0(3f), X := Tensor0(2.5f))(boundedBySigma)
    assertEqualsFloat(latent.modelLogDensity(p).item, expected.item, 1e-4f)

  test("a value outside a support that moved with a given value is an error"):
    val latent = Latent.discover()(boundedBySigma)
    // X traced inside (0, traced sigma), but not inside (0, 1e-3)
    val e = intercept[IllegalArgumentException]:
      latent.initWith(Sigma := Tensor0(1e-3f))
    assert(e.getMessage.contains("traced value of 'X'"), e.getMessage)

  test("a subclass has the support of its parent, unless it gives its own"):
    val inherited = supportOf(Sigma)(Sigma ~ MyHalfNormal())
    assert(inherited.isInstanceOf[Support.GreaterThan[?, ?]], inherited)
    assert(
      supportOf(Sigma)(Sigma ~ CappedHalfNormal())
        .isInstanceOf[Support.Interval[?, ?]]
    )

  object M extends Site[Tensor2[Row, Col, Float32]]

  test("a Rowwise(Simplex) site has a simplex in every row"):
    val latent = Latent.discover()(M ~ StochasticMatrix(2, 3))
    val rowSums = latent.init(M).sum(Axis[Col])
    for i <- 0 until 2 do
      assertEqualsFloat(rowSums.select(Axis[Row].at(i)).item, 1f, 1e-5f)
    // 2 rows of K-1 = 2 coordinates
    assertEquals(numParams(latent)(latent.init), 4)
    // the log-Jacobian is the sum of the rows' log-Jacobians
    val p = latent.init
    val row = Bijector.simplexStickBreaking[Support.Unconstrained, Col]
    val rowsLogDet = (0 until 2)
      .map(i => row.logDetJac(row.inverse(p(M).select(Axis[Row].at(i)))))
      .reduce(_ + _)
    assertEqualsFloat(
      (latent.logDensity(p) - latent.modelLogDensity(p)).item,
      rowsLogDet.item,
      1e-4f
    )

  test("rows without a bijector are an error"):
    val unknownRows: SupportOf[StochasticMatrix, (Row, Col), Float32] =
      _ => Support.Rowwise(Support.Unknown())
    val e = intercept[IllegalArgumentException]:
      Latent.discover():
        M.~(StochasticMatrix(2, 3))(using summon[ProbContext], unknownRows)
    assert(e.getMessage.contains("no bijector"), e.getMessage)

  object O extends Site[Tensor1[K, Float32]]

  test("a Custom support brings its own bijector"):
    val latent = Latent.discover()(O ~ Increasing(3))
    val xs = latent.init(O).toArray
    assert(xs.sliding(2).forall(p => p(0) < p(1)), xs.mkString(", "))
    intercept[IllegalArgumentException]:
      latent.initWith(O := Tensor1(Axis[K]).fromArray(Array(1f, 0f, 2f)))
