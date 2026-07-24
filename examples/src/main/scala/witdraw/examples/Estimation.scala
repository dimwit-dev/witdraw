package witdraw.examples

import witdraw.*
import dimwit.*
import dimwit.stats.*
import witdraw.mcmc.NUTSAlgorithm
import witdraw.plotting.Plotting
import witdraw.given
import viz.PlotTargets.websocket

object Estimation:

  trait Sample derives Label

  object NormalMean extends Site[Tensor0[Float32]]
  object NormalStdDev extends Site[Tensor0[Float32]]
  object NormalSample extends Site[Tensor1[Sample, Float32]]

  val nSamples = 1000

  lazy val gtModel: ProbProg[(gtSamples: Tensor1[Sample, Float32])] =
    ProbProg[(gtSamples: Tensor1[Sample, Float32])]:
      val samples = RandomVar(
        site = NormalSample,
        dist = Normal(
          Tensor1(Axis[Sample] -> nSamples).fill(1f) *! Tensor0(0.5f),
          Tensor1(Axis[Sample] -> nSamples).fill(1f) *! Tensor0(0.2f)
        )
      )
      (gtSamples = samples)

  val model = ProbProg:
    val mean = RandomVar(
      site = NormalMean,
      dist = Normal(Tensor0(0.0f), Tensor0(1.0f))
    )
    val stddev = RandomVar(
      site = NormalStdDev,
      dist = Exponential(Tensor0(1.0f))
    )

    val samples = RandomVar(
      site = NormalSample,
      dist = Normal(
        Tensor1(Axis[Sample] -> nSamples).fill(1f) *! mean,
        Tensor1(Axis[Sample] -> nSamples).fill(1f) *! stddev
      )
    )
    (normalSample = samples, normalMean = mean, normalStdDev = stddev)

  @main
  def runEstimation(): Unit =
    dimwit.initialize()

    val gtSamples =
      ForwardSampler.run(Key(42L))(gtModel).gtSamples

    val conditioned = model.observe(NormalSample := gtSamples)

    type PosteriorParams = (
        normalMean: Tensor0[Float32],
        normalStdDev: Tensor0[Float32]
    )

    val logDensity = (params: PosteriorParams) =>
      ModelScorer.run(conditioned.observeAll(params))

    val burnIn = 500
    val nSamples = 2000

    val nuts = NUTSAlgorithm[PosteriorParams](stepSize = 0.1)(logDensity)
    val (nutsSamples, nutsInfos) = nuts
      .run((normalMean = Tensor0(0f), normalStdDev = Tensor0(1f)), Key(42))
      .drop(burnIn)
      .take(nSamples)
      .toList
      .unzip

    val mp = Plotting.plotHistogram(
      stack(nutsSamples.map(_.normalMean), Axis[Sample]),
      title = Some("Posterior of normalMean")
    )

    Plotting.display(mp)
