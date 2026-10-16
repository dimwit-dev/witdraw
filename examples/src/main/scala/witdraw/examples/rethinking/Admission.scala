package witdraw.examples.rethinking

import dimwit.*
import witling.DataFrame
import java.io.File
import dimwit.stats.HalfCauchy
import dimwit.stats.Distribution
import witdraw.Site
import witdraw.RandomVar
import dimwit.stats.Normal
import dimwit.stats.MVNormal
import dimwit.stats.Bernoulli
import witdraw.ForwardSampler
import plotwit.PlotTargets.websocket
import plotwit.{display, grid}
import witdraw.distributions.LKJ
import witdraw.ModelScorer
import witdraw.mcmc.NUTSAlgorithm
import dimwit.jax.Jax
import witling.updated
import witling.Col
import witdraw.~

class IndependentMultivariateDist[LBatch: Label, L: Label](
    extent: AxisExtent[LBatch],
    dist: MultivariateDistribution[L, Float32]
) extends Distribution[(LBatch, L), Float32]:

  override def sample(key: Key): Tensor[(LBatch, L), Float32] =
    key.splitvmap(Axis[LBatch] -> extent.size)(k => dist.sample(k))

  override def logProb(value: Tensor[(LBatch, L), Float32]): Tensor0[LogProb] =
    val z = value.vmap(Axis[LBatch])(v => dist.logProb(v))
    z.sum

object AdmissionEx:

  @main
  def admission(): Unit =
    dimwit.initialize()

    type Schema = (
        rownames: Int,
        Admit: String,
        Gender: String,
        Dept: String,
        Freq: Int
    )

    val dataDir = new File(System.getProperty("user.dir"))
    val df =
      DataFrame.fromCSV[Schema](new File(dataDir, "data/ucbadmit.csv"))
    val dfCleaned =
      df.mapRows(row =>
        row
          .updated(Col["Admit"])(if row.Admit == "Admitted" then 1 else 0)
          .updated(Col["Gender"])(if row.Gender == "Male" then 1 else 0)
          .updated(Col["Dept"])(row.Dept match
            case "A" => 0
            case "B" => 1
            case "C" => 2
            case "D" => 3
            case "E" => 4
            case "F" => 5)
      )

    val nDepts = dfCleaned.cols.Dept.toSet.size

    trait Case derives Label // Each row in the dataset is a case
    trait Dept derives Label // Each department is a category
    trait AB derives Label // For the two parameters alpha and beta

    object Beta extends Site[Tensor0[Float32]] // The global slope parameter
    object BetaDept
        extends Site[
          Tensor1[Dept, Float32]
        ] // the slope parameter per department
    object Alpha
        extends Site[Tensor0[Float32]] // The global intercept parameter
    object AlphaDept
        extends Site[
          Tensor1[Dept, Float32]
        ] // the intercept parameter per department
    object R
        extends Site[
          Tensor2[AB, Prime[AB], Float32]
        ] // The correlation matrix for the multivariate normal distribution
    object DeptAB
        extends Site[
          Tensor2[Dept, AB, Float32]
        ] // The bivariate normal distribution connecting the alpha and beta parameters per department

    // object AdmissionProb extends Site[Tensor1[Case, Prob]]
    object Admission extends Site[Tensor1[Case, Bool]]

    val depts = dfCleaned.cols.Dept.toSet.toList.sorted
    val deptIndex = dfCleaned.cols.Dept.map(dept => depts.indexOf(dept))
    val male = Tensor1(Axis[Case]).fromArray(dfCleaned.cols.Gender.toArray)

    case class Result(
        alphaDept: AlphaDept.Value,
        betaDept: BetaDept.Value
    )

    val model = witdraw.ProbProg[Result]:

      val alpha = Alpha ~ Normal(0f, 10f)
      val beta = Beta ~ Normal(0f, 1f)

      // this is wrong
      val mu_ab = stack(Seq(Tensor0(0f), Tensor0(0f)), Axis[AB])
      val sigma_ab = stack(Seq(Tensor0(1f), Tensor0(1f)), Axis[AB])
      val r = R ~ LKJ(Tensor0(2f), Shape(Axis[AB] -> 2, Axis[Prime[AB]] -> 2))

      val cov_ab = sigma_ab.outerProduct(sigma_ab) * r
      val mvDist = MVNormal(mu_ab, cov_ab)

      val nDept = 10
      val indepMVNormal =
        new IndependentMultivariateDist[Dept, AB](Axis[Dept] -> nDepts, mvDist)

      // Sample per department a length-2 vector [alphaDept, betaDept]
      // Shape: Tensor2[Dept, AB, Float32]

      val abPerDept = DeptAB ~ indepMVNormal

      // Split that tensor into separate alphaDept and betaDept along AB
      val alphaDeptValues =
        abPerDept.slice(Axis[AB].at(0)) // Tensor1[Dept, Float32]
      val betaDeptValues =
        abPerDept.slice(Axis[AB].at(1)) // Tensor1[Dept, Float32]

      val alphaPerCase: Tensor1[Case, Float32] =
        alphaDeptValues.relabelTo(Axis[Case]).slice(Axis[Case].at(deptIndex))
      val betaPerCase: Tensor1[Case, Float32] =
        betaDeptValues.relabelTo(Axis[Case]).slice(Axis[Case].at(deptIndex))
      val admissionProb = Prob(
        Tensor.sigmoid(
          alpha +! alphaPerCase + betaPerCase * male.asType(VType[Float32])
        )
      )
      val admission = RandomVar(Admission, Bernoulli(admissionProb))
      Result(alphaDeptValues, betaDeptValues)

    trait Sample derives Label
    // val pp = Key(42L)
    //   .splitvmap(Axis[Sample] -> 1000)(k => ForwardSampler.run(k)(model))
    //   .flatten
    //   .relabelTo(Axis[Sample])
    //   .asType(VType[Float32])

    // Plotting.display(Plotting.plotHistogram(pp, Some("Admission Probability")))

    case class Posterior(
        alpha: Alpha.Value,
        beta: Beta.Value,
        r: R.Value,
        alphaDept: AlphaDept.Value,
        betaDept: BetaDept.Value,
        deptAB: DeptAB.Value
    ) derives TensorTree

    def logDensity(admission: Admission.Value)(params: Posterior) =
      val observedModel = model.observe(
        Alpha := params.alpha,
        Beta := params.beta,
        R := params.r,
        DeptAB := params.deptAB,
        AlphaDept := params.alphaDept,
        BetaDept := params.betaDept,
        Admission := admission
      )

      ModelScorer.run(observedModel)

    val initialParams = Posterior(
      alpha = Tensor0(0f),
      beta = Tensor0(0f),
      r = Tensor2(Axis[AB] -> 2).eye,
      alphaDept = Tensor1(Axis[Dept] -> nDepts).fill(0f),
      betaDept = Tensor1(Axis[Dept] -> nDepts).fill(0f),
      deptAB = Tensor2(Axis[Dept] -> nDepts, Axis[AB] -> 2).fill(1f)
    )

    val admission = Tensor1(Axis[Case] -> dfCleaned.size).fromArray(
      dfCleaned.cols.Admit.toArray.map(x => if x == 1 then true else false)
    )
    val burnIn = 200
    val nSamples = 200
    println("starting sampling")
    val nuts =
      NUTSAlgorithm[Posterior](stepSize = 0.1)(jit(logDensity(admission)))
    val (nutsSamples, nutsInfos) = nuts
      .run(
        initialParams,
        Key(42)
      )
      .drop(burnIn)
      .take(nSamples)
      .toList
      .unzip
    println(" done sampling")
    val alphaDepts = stack(nutsSamples.map(_.alphaDept), Axis[Sample])
    val betaDepts = stack(nutsSamples.map(_.betaDept), Axis[Sample])

    println(f"mean alpha: ${alphaDepts.mean(Axis[Sample])}")
    println(f"mean beta ${betaDepts.mean(Axis[Sample])}")
