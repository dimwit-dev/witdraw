package witdraw

import dimwit.*
import dimwit.stats.*
import witdraw.inference.Chain

class MarginalizeSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  trait N derives Label

  lazy val xs = Tensor1(Axis[N]).fromArray(Array(0.3f, -1.2f, 2.5f, 1.9f))

  object Mu extends Site[Tensor0[Float32]]
  object Z extends Site[Tensor0[Int32]]
  object B extends Site[Tensor0[Bool]]
  object X extends Site[Tensor1[N, Float32]]

  def normal(loc: Tensor0[Float32]) =
    Normal(Tensor.like(xs).fill(0f) +! loc, Tensor.like(xs).fill(1f))

  /** A change of the mean by `z`, uniform over 0 to 2. */
  def model(using ProbContext) =
    val mu = Mu ~ Normal(0f, 1f)
    val z = Z ~ DiscreteUniform(0, 3)
    val x = X ~ normal(mu + z.asFloat32)
    (mu = mu, z = z, x = x)

  lazy val latent = Latent.discover(marginalize(Z)(observe(X := xs)(model)))

  def logSumExp(ls: Seq[Double]): Double =
    val m = ls.max
    m + math.log(ls.map(l => math.exp(l - m)).sum)

  def logLik(loc: Float): Double = normal(Tensor0(loc)).logProb(xs).asFloat.item.toDouble

  /** `log p(mu, x)`, with `z` summed out by hand. */
  def byHand(mu: Float): Double =
    val logPrior = Normal(Tensor0(0f), Tensor0(1f)).logProb(Tensor0(mu)).asFloat.item
    logPrior + logSumExp((0 to 2).map(z => math.log(1.0 / 3) + logLik(mu + z)))

  test("the sites summed out are not parameters"):
    assertEquals(latent.sites.map(_.name), Seq("Mu"))
    assertEquals(latent.marginalizedSites.map(_.name), Seq("Z"))

  test("the log-density is the one with the site summed out by hand"):
    for mu <- Seq(-1f, 0.4f, 2f) do
      val p = latent.initWith(Mu := Tensor0(mu))
      // Mu is unconstrained: no Jacobian
      assertEqualsDouble(latent.logDensity(p).item.toDouble, byHand(mu), 1e-3)
      assertEqualsDouble(latent.modelLogDensity(p).item.toDouble, byHand(mu), 1e-3)

  test("posteriorOf is the exact posterior of the site given the parameters"):
    val p = latent.initWith(Mu := Tensor0(0.4f))
    val joint = (0 to 2).map(z => logLik(0.4f + z))
    val byHandPosterior = joint.map(l => math.exp(l - logSumExp(joint)))
    assertEquals(latent.valuesOf(Z).toArray.toSeq, Seq(0, 1, 2))
    latent.posteriorOf(Z)(p).toArray.zip(byHandPosterior).foreach: (a, b) =>
      assertEqualsDouble(a.toDouble, b, 1e-5)

  test("result is an error, pointing to recover"):
    val error = intercept[IllegalStateException](latent.result(latent.init))
    assert(error.getMessage.contains("recover"), error.getMessage)

  test("recover draws the site from its posterior given the parameters"):
    val p = latent.initWith(Mu := Tensor0(0.4f))
    val n = 2000
    val chain = Chain(Seq.fill(n)((p, ())))
    val recovered = latent.recover(chain, Random.Key(3))
    val samples = recovered.samples
    assert(samples.forall(_.mu.item == 0.4f))
    val counts = samples.groupMapReduce(_.z.item)(_ => 1)(_ + _)
    latent.posteriorOf(Z)(p).toArray.zipWithIndex.foreach: (prob, z) =>
      val frequency = counts.getOrElse(z, 0).toFloat / n
      assertEqualsFloat(frequency, prob, 0.04f, s"z = $z")

  test("recover is an error without sites summed out"):
    val complete = Latent.discover(observe(X := xs, Z := Tensor0(1))(model))
    intercept[IllegalStateException](complete.recover(Chain(Seq((complete.init, ()))), Random.Key(0)))

  test("several sites are summed out over all combinations of their values"):
    def twoSites(using ProbContext) =
      val z = Z ~ DiscreteUniform(0, 3)
      val b = B ~ Bernoulli(Prob(Tensor0(0.3f)))
      val mu = Mu ~ Normal(0f, 1f)
      X ~ normal(mu + z.asFloat32 + where(b, Tensor0(5f), Tensor0(0f)))
      (z = z, b = b)
    val both = Latent.discover(marginalize(Z, B)(observe(X := xs)(twoSites)))
    val p = both.initWith(Mu := Tensor0(0.2f))
    val terms =
      for z <- 0 to 2; (b, pb) <- Seq((0f, 0.7), (5f, 0.3))
      yield math.log(pb / 3) + logLik(0.2f + z + b)
    val m = terms.max
    val logPrior = Normal(Tensor0(0f), Tensor0(1f)).logProb(Tensor0(0.2f)).asFloat.item
    val expected = logPrior + m + math.log(terms.map(t => math.exp(t - m)).sum)
    assertEqualsDouble(both.modelLogDensity(p).item.toDouble, expected, 1e-3)
    assertEqualsFloat(both.posteriorOf(B)(p).sum.item, 1f, 1e-5f)
    assertEqualsFloat(both.posteriorOf(Z)(p).sum.item, 1f, 1e-5f)

  test("a forward run still draws the sites summed out"):
    val zs = Random.Key(0).split(30).map(k => ForwardSampler.run(k)(marginalize(Z)(model)).z.item)
    assert(zs.forall(z => z >= 0 && z <= 2), zs)
    assert(zs.distinct.size > 1, zs)

  test("summing out an observed site is an error"):
    intercept[IllegalArgumentException]:
      Latent.discover(marginalize(Z)(observe(X := xs, Z := Tensor0(1))(model)))

  test("summing out a site whose values cannot be listed is an error"):
    object Count extends Site[Tensor0[Int32]]
    val error = intercept[IllegalArgumentException]:
      Latent.discover(marginalize(Count)(Count ~ Poisson(Tensor0(2f))))
    assert(error.getMessage.contains("cannot be listed"), error.getMessage)

  test("summing out a site whose values depend on other sites is an error"):
    object Upper extends Site[Tensor0[Int32]]
    def dependent(using ProbContext) =
      val upper = Upper ~ DiscreteUniform(2, 1000)
      Z ~ DiscreteUniform(Tensor0(0), upper)
    val error = intercept[IllegalArgumentException]:
      Latent.discover(marginalize(Upper, Z)(dependent))
    assert(error.getMessage.contains("depend on other sites"), error.getMessage)

  test("recover keeps the chains apart, and posteriorOf over chains averages the posteriors"):
    val (p, q) = (latent.initWith(Mu := Tensor0(-0.5f)), latent.initWith(Mu := Tensor0(1f)))
    val chains = Seq(Chain(Seq.fill(3)((p, ()))), Chain(Seq.fill(5)((q, ()))))
    val recovered = latent.recover(chains, Random.Key(1))
    assertEquals(recovered.map(_.size), Seq(3, 5))
    assertEquals(recovered.map(_.samples.map(_.mu.item).distinct), Seq(Seq(-0.5f), Seq(1f)))
    val expected = latent.posteriorOf(Z)(p).toArray.zip(latent.posteriorOf(Z)(q).toArray)
      .map((a, b) => (3 * a + 5 * b) / 8)
    latent.posteriorOf(Z, chains).toArray.zip(expected).foreach((a, b) => assertEqualsFloat(a, b, 1e-5f))
