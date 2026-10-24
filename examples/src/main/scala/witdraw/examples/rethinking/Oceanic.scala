package witdraw.examples.rethinking

import dimwit.*
import dimwit.stats.*
import witdraw.*
import witling.*
import java.io.File
import witdraw.inference.NUTS
import plotwit.plots.histogramPlot

object OceanicExample:

  trait Society derives Label

  object NumTools extends Site[Tensor1[Society, Int32]]
  object SocietyIntercept extends Site[Tensor1[Society, Float32]]
  object GlobalIntercept extends Site[Tensor0[Float32]]
  object SocietySlope extends Site[Tensor0[Float32]]
  object Eta2 extends Site[Tensor0[Float32]]
  object Rho2 extends Site[Tensor0[Float32]]

  def model(
      population: Tensor1[Society, Float32],
      distances: Tensor2[Society, Prime[Society], Float32]
  )(using ProbContext) =
    val numSocieties = 10
    val globalIntercept =
      GlobalIntercept ~ Normal(Tensor0(0.0f), Tensor0(10.0f))

    val eta2: Tensor0[Float32] =
      Eta2 ~ HalfCauchy(Tensor0(0.0f), Tensor0(1.0f))

    val rho2: Tensor0[Float32] =
      Rho2 ~ HalfCauchy(Tensor0(0.0f), Tensor0(1.0f))

    val squaredists = distances * distances
    val K: Tensor2[Society, Prime[Society], Float32] =
      eta2 *! (-rho2 *! squaredists).exp + Tensor2(
        Axis[Society] -> numSocieties
      ).eye *! Tensor0(0.01f)

    val societyIntercept = SocietyIntercept ~ MVNormal(
      mean = Tensor1(Axis[Society] -> numSocieties).fill(0f),
      covariance = K
    )

    val societySlope = SocietySlope ~ Normal(Tensor0(0f), Tensor0(1f))

    val logToolRate =
      globalIntercept +! societyIntercept + societySlope *! population.log
    val toolRate = logToolRate.exp

    val numTools = NumTools ~ Poisson(toolRate)

    (
      globalIntercept = globalIntercept,
      societyIntercept = societyIntercept,
      societySlope = societySlope,
      eta2 = eta2,
      rho2 = rho2,
      numTools = numTools
    )

  @main
  def runOceanicExample(): Unit =
    dimwit.initialize()

    val dataDir = new File(System.getProperty("user.dir"))
    val df =
      DynamicDataFrame.fromCSV(File(dataDir, "data/islandsDistMatrix.csv"))

    val typed = df.narrowTo[
      (
          Malekula: Float,
          Tikopia: Float,
          Santa_Cruz: Float,
          Yap: Float,
          Lau_Fiji: Float,
          Trobriand: Float,
          Chuuk: Float,
          Manus: Float,
          Tonga: Float,
          Hawaii: Float
      )
    ]
    println(typed.show())

    val kline2 = DynamicDataFrame.fromCSV(File(dataDir, "data/Kline2.csv"))
    val klineTyped = kline2.narrowTo[
      (
          culture: String,
          population: Int,
          total_tools: Int
      )
    ]
    val distances = Tensor2(Axis[Society] -> 10, Axis[Prime[Society]] -> 10)
      .fromArray(
        typed.cols.Malekula.toArray ++
          typed.cols.Tikopia.toArray ++
          typed.cols.Santa_Cruz.toArray ++
          typed.cols.Yap.toArray ++
          typed.cols.Lau_Fiji.toArray ++
          typed.cols.Trobriand.toArray ++
          typed.cols.Chuuk.toArray ++
          typed.cols.Manus.toArray ++
          typed.cols.Tonga.toArray ++
          typed.cols.Hawaii.toArray
      )

    val population = Tensor1(Axis[Society] -> 10)
      .fromArray(klineTyped.cols.population.toArray.map(_.toFloat))
    val numTools = Tensor1(Axis[Society]).fromArray(
      klineTyped.cols.total_tools.toArray.map(_.toInt)
    )

    // eta2 and rho2 are the scale and decay rate of the covariance kernel,
    // so both are sampled on the log scale.
    val posterior = Latent.discover(
      Constraint(Eta2, Bijector.exp),
      Constraint(Rho2, Bijector.exp)
    ):
      observe(NumTools := numTools)(model(population, distances))
    println(s"latent sites: ${posterior.sites.map(_.name).mkString(", ")}")

    val numWarmup = 200
    val nSamples = 500
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

    trait Sample derives Label
    println(s"NUTS divergences: ${nutsInfos.count(_.isDivergent)} / $nSamples")
    for site <- List(GlobalIntercept, SocietySlope, Eta2, Rho2) do
      val draws = stack(nutsSamples.map(_(site)), Axis[Sample])
      println(
        f"${site.name}%-16s mean ${draws.mean.item}%7.3f  sd ${draws.std.item}%6.3f"
      )
    val p1 = histogramPlot(
      stack(nutsSamples.map(_(GlobalIntercept)), Axis[Sample]),
      _.title := "Posterior of globalIntercept"
    )

    val p2 = histogramPlot(
      stack(nutsSamples.map(_(SocietySlope)), Axis[Sample]),
      _.title := "Posterior of societySlope"
    )

    val p3 = histogramPlot(
      stack(
        nutsSamples.map(_(SocietyIntercept).select(Axis[Society].at(0))),
        Axis[Sample]
      ),
      _.title := "Posterior of societyIntercept 0"
    )

    val p4 = histogramPlot(
      stack(
        nutsSamples.map(_(SocietyIntercept).select(Axis[Society].at(1))),
        Axis[Sample]
      ),
      _.title := "Posterior of societyIntercept 1"
    )

    import plotwit.*
    import plotwit.PlotTargets.websocket
    val allPlots = grid(List(List(p1, p2), List(p3, p4)))

    display(allPlots)
