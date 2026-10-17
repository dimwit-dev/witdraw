package witdraw
package examples.rethinking

import dimwit.*
import dimwit.stats.*
import witdraw.*
import witdraw.mcmc.OptimizerKernel
import witdraw.mcmc.NUTSAlgorithm

import plotwit.PlotTargets.websocket
import plotwit.{display, grid}
import plotwit.plots.histogramPlot

object WaterLandExample:

  object WaterProp extends Site[Tensor0[Float32]]
  object WaterCount extends Site[Tensor0[Int32]]

  def model(numTosses: Int)(using ProbContext) =
    val wp = WaterProp ~ Uniform(0f, 1f)
    val wc = WaterCount ~ Binomial(numTosses, Prob(wp))
    (waterProp = wp, waterCount = wc)

  @main
  def main(): Unit =
    dimwit.initialize()
    val numTosses = 9
    val waterCountData = Tensor0(6)

    trait Sample derives Label

    val samples =
      Key(42).split(10000).map(key => ForwardSampler.run(key)(model(numTosses)))

    val priorSamples = samples.map(_.waterProp)
    val priorPredictiveSamples = samples.map(_.waterCount)

    val predictiveSamples =
      Key(42)
        .split(10000)
        .map: key =>
          ForwardSampler.run(key):
            observe(WaterProp := Prob(0.7f))(model(numTosses))

    type PosteriorParams = (waterProp: Tensor0[Float32])

    val logDensity = (params: PosteriorParams) =>
      ModelScorer.run:
        observe(WaterCount := waterCountData):
          observeAll(params)(model(numTosses))

    val burnIn = 500
    val nSamples = 2000

    val ok = OptimizerKernel.toSamplingAlgorithm(
      optimizer = dimwit.optimizer.Adam(learningRate = -0.1f),
      gradFn = (params: PosteriorParams) => Autodiff.grad(logDensity)(params)
    )

    val map = ok
      .run((waterProp = Tensor0(0.5f)), Key(42))
      .take(100)
      .toList
      .last
      ._1

    println(s"MAP estimate: ${map.waterProp}")

    val nuts = NUTSAlgorithm[PosteriorParams](stepSize = 0.1)(logDensity)
    val (nutsSamples, nutsInfos) = nuts
      .run(map, Key(42))
      .drop(burnIn)
      .take(nSamples)
      .toList
      .unzip

    val divergences = nutsInfos.count(_.isDivergent)
    println(f"NUTS divergences: $divergences / $nSamples")
    println(
      f"NUTS mean acceptance rate: ${nutsInfos.map(_.acceptanceRate).sum / nSamples}%.3f"
    )

    val posteriorSamples = stack(nutsSamples.map(_.waterProp), Axis[Sample])
    display(
      histogramPlot(
        posteriorSamples,
        _.title := "Posterior distribution of water proportion"
      )
    )
