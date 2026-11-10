package witdraw.plots

import dimwit.*
import dimwit.stats.Normal
import witdraw.*
import witdraw.inference.Chain

class SummarySuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  trait K derives Label
  object Mu extends Site[Tensor0[Float32]]
  object Beta extends Site[Tensor1[K, Float32]]

  def model(using ProbContext) =
    val mu = Mu ~ Normal(0f, 1f)
    val beta = Beta ~ Normal.isotropic(Tensor1(Axis[K] -> 2).fill(0f), 1f)
    (mu = mu, beta = beta)

  lazy val latent = Latent.discover(model)

  /** A chain whose draws have the given values of mu, and beta = (mu, -mu). */
  def chain(mus: Float*) =
    Chain(mus.map: m =>
      latent.initWith(Mu := Tensor0(m), Beta := Tensor1(Axis[K]).fromArray(Array(m, -m))) -> ())

  lazy val chains = Seq(chain(1f, 2f, 3f, 4f), chain(5f, 6f, 7f, 8f))

  test("one row per element, named by the fields and the axis names"):
    val table = Summary.summary(chains, Axis[K].names(Seq("a", "b")))(s => (mu = s.mu, beta = s.beta))
    assertEquals(table.rows.map(_.name), Seq("mu", "beta[a]", "beta[b]"))
    // the draws of all chains are pooled: 1, ..., 8
    assertEqualsDouble(table.rows(0).mean, 4.5, 1e-5)
    assertEqualsDouble(table.rows(0).sd, math.sqrt(5.25), 1e-5)
    assertEqualsDouble(table.rows(2).mean, -4.5, 1e-5)
    // the chains disagree, so R-hat is far above 1
    assert(table.rows(0).rhat > 1.5, s"rhat ${table.rows(0).rhat}")
    assert(table.toString.linesIterator.next().startsWith("parameter"), table.toString)

  test("entries without names are numbered, and wrong names are an error"):
    val table = Summary.summary(chains)(s => (beta = s.beta))
    assertEquals(table.rows.map(_.name), Seq("beta[0]", "beta[1]"))
    val e = intercept[IllegalArgumentException]:
      Summary.summary(chains, Axis[K].names(Seq("a")))(s => (beta = s.beta))
    assert(e.getMessage.contains("1 names for axis K of beta"), e.getMessage)
