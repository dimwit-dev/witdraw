package witdraw.examples.rethinking

import dimwit.*
import dimwit.tensor.Labels
import io.github.quafadas.table.*
import scala.NamedTuple.{Names, DropNames}
import dimwit.stats.HalfCauchy
import dimwit.stats.Distribution
import witdraw.{Site, Support, SupportOf}
import dimwit.stats.Normal
import dimwit.stats.MVNormal
import dimwit.stats.Bernoulli
import witdraw.ForwardSampler
import plotwit.PlotTargets.websocket
import plotwit.{display, grid}
import witdraw.distributions.LKJ
import witdraw.Latent
import witdraw.ProbProg
import witdraw.observe
import witdraw.inference.NUTS
import dimwit.jax.Jax

/** `extent.size` independent draws of a multivariate normal, one per value of
  * `LBatch`.
  */
class IndependentMultivariateDist[LBatch: Label, L: Label](
    extent: AxisExtent[LBatch],
    dist: MVNormal[L, Float32]
) extends Distribution[(LBatch, L), Float32]:

  override def sample(key: Key): Tensor[(LBatch, L), Float32] =
    key.splitvmap(Axis[LBatch] -> extent.size)(k => dist.sample(k))

  override def logProb(value: Tensor[(LBatch, L), Float32]): Tensor0[LogProb] =
    val z = value.vmap(Axis[LBatch])(v => dist.logProb(v))
    z.sum

object IndependentMultivariateDist:
  /** Normal draws take any value. */
  given [LBatch, L]: SupportOf[
    IndependentMultivariateDist[LBatch, L],
    (LBatch, L),
    Float32
  ] = _ => Support.Real()

object AdmissionEx:

  @main
  def admission(): Unit =
    dimwit.initialize()

    type UcbAdmit = (
        rownames: Int,
        Admit: String,
        Gender: String,
        Dept: String,
        Freq: Int
    )

    val df = CSV
      .fromTyped[Names[UcbAdmit], DropNames[UcbAdmit]](
        os.pwd / "data" / "ucbadmit.csv"
      )
      .toVector
    val dfCleaned = df
      .map(row =>
        (
          Admit = if row.Admit == "Admitted" then 1 else 0,
          Gender = if row.Gender == "Male" then 1 else 0,
          Dept = row.Dept match
            case "A" => 0
            case "B" => 1
            case "C" => 2
            case "D" => 3
            case "E" => 4
            case "F" => 5
          ,
          Freq = row.Freq
        )
      )
      .toColumnOriented

    // Each row counts the applications with one outcome, gender and
    // department; expand them into one case per application.
    val rows = dfCleaned.Freq.toArray.indices.flatMap: i =>
      Seq.fill(dfCleaned.Freq.toArray(i))(i)
    def perCase(col: Array[Int]): Seq[Int] = rows.map(col)
    val admitted = perCase(dfCleaned.Admit.toArray)
    val isMale = perCase(dfCleaned.Gender.toArray)
    val deptIndex = perCase(dfCleaned.Dept.toArray)
    val depts = Seq("A", "B", "C", "D", "E", "F")
    val nDepts = depts.size
    println(s"${rows.size} applications in ${nDepts} departments")

    trait Case derives Label // one application
    trait Dept derives Label // a department
    trait AB derives Label // the intercept (0) and the slope for male (1)

    object Alpha extends Site[Tensor0[Float32]] // the mean of the intercepts
    object Beta extends Site[Tensor0[Float32]] // the mean of the slopes
    object AlphaDept
        extends Site[Tensor1[Dept, Float32]] // the intercept per department
    object BetaDept
        extends Site[Tensor1[Dept, Float32]] // the slope per department
    object Sigma
        extends Site[
          Tensor1[AB, Float32]
        ] // the spread of the intercepts and of the slopes
    object R
        extends Site[
          Tensor2[AB, Prime[AB], Float32]
        ] // the correlation of intercept and slope
    object DeptAB
        extends Site[
          Tensor2[Dept, AB, Float32]
        ] // intercept and slope per department, from a bivariate normal
    object Admission extends Site[Tensor1[Case, Bool]]

    val male = Tensor1(Axis[Case]).fromArray(isMale.map(_.toFloat).toArray)

    case class Result(
        alphaDept: AlphaDept.Value,
        betaDept: BetaDept.Value
    )

    // McElreath's varying intercepts and slopes model of the admissions:
    // logit p = alphaDept + betaDept * male, where the pairs
    // (alphaDept, betaDept) are bivariate normal around (alpha, beta).
    val model: ProbProg[Result] =
      val alpha = Alpha ~ Normal(0f, 10f)
      val beta = Beta ~ Normal(0f, 1f)
      val sigma = Sigma ~ HalfCauchy(
        Tensor1(Axis[AB] -> 2).fill(0f),
        Tensor1(Axis[AB] -> 2).fill(2f)
      )
      val r = R ~ LKJ(Tensor0(2f), Shape(Axis[AB] -> 2, Axis[Prime[AB]] -> 2))

      val muAB = stack(Seq(alpha, beta), Axis[AB])
      val covAB = sigma.outerProduct(sigma) * r
      val perDept = new IndependentMultivariateDist[Dept, AB](
        Axis[Dept] -> nDepts,
        MVNormal(muAB, covAB)
      )
      val abPerDept = DeptAB ~ perDept

      val alphaDeptValues = abPerDept.select(Axis[AB].at(0))
      val betaDeptValues = abPerDept.select(Axis[AB].at(1))

      val alphaPerCase: Tensor1[Case, Float32] =
        alphaDeptValues.relabelTo(Axis[Case]).select(Axis[Case].at(deptIndex))
      val betaPerCase: Tensor1[Case, Float32] =
        betaDeptValues.relabelTo(Axis[Case]).select(Axis[Case].at(deptIndex))
      val admissionProb =
        Prob(Tensor.sigmoid(alphaPerCase + betaPerCase * male))
      val admission = Admission ~ Bernoulli(admissionProb)
      Result(alphaDeptValues, betaDeptValues)

    val admission = Tensor1(Axis[Case]).fromArray(admitted.map(_ == 1).toArray)

    // The latent sites are Alpha, Beta, Sigma, R and DeptAB; the intercepts
    // and slopes per department are the columns of DeptAB. Latent reads the
    // supports off the distributions: the spreads (HalfCauchy) are sampled on
    // the log scale, and R (LKJ) through the correlation bijector, so it stays
    // a correlation matrix.
    val posterior = Latent.discover():
      observe(Admission := admission)(model)
    println(s"latent sites: ${posterior.sites.map(_.name).mkString(", ")}")

    val numWarmup = 1000
    val nSamples = 1000
    // The warmup tunes the step size and the mass matrix.
    val (warmupKey, sampleKey) = Key(42).split2()
    val adapted =
      NUTS.adapt(posterior, posterior.init, warmupKey, numSteps = numWarmup)
    println(f"NUTS adapted step size: ${adapted.stepSize}%.3f")
    val (nutsSamples, nutsInfos) = adapted.sampler
      .run(adapted.position, sampleKey)
      .take(nSamples)
      .toList
      .unzip
    println(s"NUTS divergences: ${nutsInfos.count(_.isDivergent)} / $nSamples")

    trait Sample derives Label
    def summary[T <: Tuple: Labels](draws: List[Tensor[T, Float32]]) =
      val stacked = stack(draws, Axis[Sample])
      (stacked.mean(Axis[Sample]), stacked.std(Axis[Sample]))

    val (alphaMean, alphaSd) = summary(nutsSamples.map(_(Alpha)))
    val (betaMean, betaSd) = summary(nutsSamples.map(_(Beta)))
    val (sigmaMean, _) = summary(nutsSamples.map(_(Sigma)))
    val (rMean, _) = summary(nutsSamples.map(_(R)))
    val (deptMean, deptSd) = summary(nutsSamples.map(_(DeptAB)))
    println(f"alpha: mean ${alphaMean.item}%6.2f  sd ${alphaSd.item}%5.2f")
    println(f"beta:  mean ${betaMean.item}%6.2f  sd ${betaSd.item}%5.2f")
    println(s"sigma (intercept, slope): $sigmaMean")
    println(
      f"correlation of intercept and slope: ${rMean(Axis[AB].at(0))(Axis[Prime[AB]].at(1)).item}%.2f"
    )
    println("per department   intercept        slope for male")
    for (dept, d) <- depts.zipWithIndex do
      def at(m: Tensor2[Dept, AB, Float32], ab: Int) =
        m(Axis[Dept].at(d))(Axis[AB].at(ab)).item
      println(
        f"  $dept             ${at(deptMean, 0)}%6.2f (sd ${at(deptSd, 0)}%4.2f)   ${at(deptMean, 1)}%6.2f (sd ${at(deptSd, 1)}%4.2f)"
      )
