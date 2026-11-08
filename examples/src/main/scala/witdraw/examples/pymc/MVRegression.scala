package witdraw
package examples.pymc

import dimwit.*
import dimwit.stats.{Gamma, HalfNormal, Normal, StudentT}
import io.github.quafadas.table.*
import scala.NamedTuple.{Names, DropNames}
import scala.concurrent.{Await, Future, blocking}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration
import witdraw.*
import witdraw.distributions.{Half, Inverse}
import witdraw.inference.{Diagnostics, NUTS}

/** Linear regression with a regularized horseshoe prior on the coefficients,
  * after the PyMC example "Multivariate regression with a regularized
  * horseshoe" on the test scores of children with hearing loss.
  *
  * The horseshoe shrinks most coefficients strongly towards zero, while letting
  * a few large ones through; `c2` caps how large they can get.
  */
object MVRegression:

  type TestScore = (
      id: Int,
      score: Int,
      male: Int,
      siblings: Double,
      family_inv: Option[Double],
      non_english: Boolean,
      prev_disab: Option[Double],
      age_test: Int,
      non_severe_hl: Double,
      mother_hs: Option[Double],
      early_ident: Boolean,
      non_white: Boolean
  )

  val predictorNames = Seq(
    "male",
    "siblings",
    "family_inv",
    "non_english",
    "prev_disab",
    "age_test",
    "non_severe_hl",
    "mother_hs",
    "early_ident",
    "non_white"
  )

  trait Predictor derives Label // D: one entry per predictor
  trait Sample derives Label // N: one row per child

  object Sigma extends Site[Tensor0[Float32]] // error scale
  object Tau extends Site[Tensor0[Float32]] // global shrinkage
  object Lambda extends Site[Tensor1[Predictor, Float32]] // local shrinkage
  object C2 extends Site[Tensor0[Float32]] // slab width, squared
  object Z extends Site[Tensor1[Predictor, Float32]] // unscaled coefficients
  object Beta0 extends Site[Tensor0[Float32]] // intercept
  object Scores extends Site[Tensor1[Sample, Float32]] // observed scores

  /** The shrunken coefficients, `pm.Deterministic("beta", ...)` in PyMC:
    * `z * tau * lam * sqrt(c2 / (c2 + tau^2 * lam^2))`.
    */
  def shrunkCoefficients(
      z: Tensor1[Predictor, Float32],
      tau: Tensor0[Float32],
      lambda: Tensor1[Predictor, Float32]
  )(c2: Tensor0[Float32]): Tensor1[Predictor, Float32] =
    val tauLambda = lambda *! tau
    val shrinkage = (((tauLambda * tauLambda) +! c2).pow(-1f) *! c2).sqrt
    z * tauLambda * shrinkage

  def model(x: Tensor2[Sample, Predictor, Float32])(using ProbContext) =
    val n = x.shape(Axis[Sample]) // N: number of observations
    val d = x.shape(Axis[Predictor]) // D: number of predictors
    val d0 = d / 2 // D0: expected number of relevant predictors
    val zeros = Tensor1(x.extent(Axis[Predictor])).fill(0f)
    val ones = Tensor1(x.extent(Axis[Predictor])).fill(1f)

    val sigma = Sigma ~ HalfNormal(0f, 25f)

    // Global shrinkage, scaled by the expected fraction of relevant predictors
    val tauScale = sigma * Tensor0(d0.toFloat / (d - d0) / math.sqrt(n).toFloat)
    val tau = Tau ~ Half(StudentT(2f, 0f, tauScale))
    // Local shrinkage, one per predictor
    val lambda = Lambda ~ Half(StudentT(Tensor0(5f), zeros, ones))
    val c2 = C2 ~ Inverse(Gamma(Tensor0(1f), Tensor0(1f)))
    val z = Z ~ Normal(zeros, ones)
    val beta = shrunkCoefficients(z, tau, lambda)(c2)

    // No shrinkage on the intercept
    val beta0 = Beta0 ~ Normal(100f, 25f)

    val mu = x.dot(Axis[Predictor])(beta) +! beta0
    val scores = Scores ~ Normal.isotropic(mu, sigma)
    (beta = beta, beta0 = beta0, sigma = sigma, scores = scores)

  /** The children with all predictors known: their predictors, standardized to
    * mean 0 and standard deviation 1 as in the PyMC example, and scores.
    */
  def loadData()
      : (Tensor2[Sample, Predictor, Float32], Tensor1[Sample, Float32]) =
    val rows = CSV
      .fromTyped[Names[TestScore], DropNames[TestScore]](
        os.pwd / "data" / "test_scores.csv"
      )
      .toVector
      .filter(r =>
        r.family_inv.isDefined && r.prev_disab.isDefined && r.mother_hs.isDefined
      )
    def flag(b: Boolean) = if b then 1.0 else 0.0
    val predictors = rows.map(r =>
      Array(
        r.male.toDouble,
        r.siblings,
        r.family_inv.get,
        flag(r.non_english),
        r.prev_disab.get,
        r.age_test.toDouble,
        r.non_severe_hl,
        r.mother_hs.get,
        flag(r.early_ident),
        flag(r.non_white)
      )
    )
    val columns = predictors.transpose
    val standardized = columns.map: c =>
      val mean = c.sum / c.size
      // sample standard deviation, as pandas computes it
      val sd = math.sqrt(c.map(v => (v - mean) * (v - mean)).sum / (c.size - 1))
      c.map(v => ((v - mean) / sd).toFloat)
    val x = Tensor2(Axis[Sample], Axis[Predictor])
      .fromArray(standardized.transpose.map(_.toArray).toArray)
    val y = Tensor1(Axis[Sample]).fromArray(rows.map(_.score.toFloat).toArray)
    (x, y)

  @main
  def runMVRegression(): Unit =
    dimwit.initialize()

    val (x, y) = loadData()
    println(
      s"${x.shape(Axis[Sample])} children, ${x.shape(Axis[Predictor])} predictors"
    )

    // sigma, tau, lambda and c2 are positive, so Latent samples them on the
    // log scale.
    val posterior =
      Latent.discover:
        observe(Scores := y)(model(x))
    println(s"latent sites: ${posterior.sites.map(_.name).mkString(", ")}")

    val numChains = 4
    val numDraws = 1000
    val Seq(initKey, chainsKey) = Random.Key(42).split(2)
    val starts = initKey.split(numChains).map(posterior.initRandom(_))
    // One chain: a warmup of its own, then sampling. The horseshoe has a
    // funnel-shaped posterior: a higher target acceptance rate, as in the PyMC
    // example, gives smaller steps and fewer divergences.
    def chain(key: Random.Key, start: posterior.Params) =
      val (warmupKey, sampleKey) = key.split2()
      val adapted = NUTS.adapt(
        posterior,
        warmupKey,
        start,
        numSteps = 1000,
        targetAcceptanceRate = 0.99
      )
      adapted.sample(sampleKey, numDraws)

    // the chains run at the same time, one per starting position
    val runs = chainsKey
      .split(numChains)
      .zip(starts)
      .map((key, start) => Future(blocking(chain(key, start))))
    val chains = Await.result(Future.sequence(runs), Duration.Inf)
    println(
      s"NUTS divergences: ${chains.map(_.numDivergent).sum} / ${numChains * numDraws}"
    )

    val betaChains =
      chains.map(
        _.values(p => shrunkCoefficients(p(Z), p(Tau), p(Lambda))(p(C2)))
      )

    type Row = (name: String, mean: Float, sd: Float, rhat: Float, ess: Float)

    def scalarRow(name: String, perChain: Seq[Seq[Tensor0[Float32]]]): Row =
      val all = stack(perChain.flatten, Axis[Sample])
      (
        name = name,
        mean = all.mean.item,
        sd = all.std.item,
        rhat = Diagnostics.rhat(perChain).item,
        ess = Diagnostics.ess(perChain).item
      )

    def vectorRows(
        name: String,
        perChain: Seq[Seq[Tensor1[Predictor, Float32]]]
    ): Seq[Row] =
      val all = stack(perChain.flatten, Axis[Sample])
      val means = all.mean(Axis[Sample]).toArray
      val sds = all.std(Axis[Sample]).toArray
      val rhat = Diagnostics.rhat(perChain).toArray
      val ess = Diagnostics.ess(perChain).toArray
      predictorNames.indices.map: i =>
        (
          name = s"$name[${predictorNames(i)}]",
          mean = means(i),
          sd = sds(i),
          rhat = rhat(i),
          ess = ess(i)
        )

    val rows =
      Seq(
        scalarRow("sigma", chains.map(_(Sigma))),
        scalarRow("tau", chains.map(_(Tau))),
        scalarRow("c2", chains.map(_(C2))),
        scalarRow("beta0", chains.map(_(Beta0)))
      ) ++ vectorRows("beta", betaChains)

    println(
      f"${"parameter"}%-22s ${"mean"}%8s ${"sd"}%7s ${"rhat"}%6s ${"ess"}%7s"
    )
    for r <- rows do
      println(
        f"${r.name}%-22s ${r.mean}%8.3f ${r.sd}%7.3f ${r.rhat}%6.3f ${r.ess}%7.0f"
      )
