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

    def conditioned(using ProbContext) =
      observe(WaterCount := waterCountData)(model(numTosses))

    // The latent sites are found by tracing one run of the conditioned
    // model. WaterProp is a proportion, so it is sampled on the logit scale.
    val latent = Latent.discover(
      Constraint(WaterProp, Bijector.interval(Tensor0(0f), Tensor0(1f)))
    )(conditioned)
    println(s"Latent sites: ${latent.sites.map(_.name).mkString(", ")}")

    val burnIn = 500
    val nSamples = 2000

    val ok = OptimizerKernel.toSamplingAlgorithm(
      optimizer = dimwit.optimizer.Adam(learningRate = -0.1f),
      gradFn =
        (params: latent.Params) => Autodiff.grad(latent.modelLogDensity)(params)
    )

    val map = ok
      .run(latent.initWith(WaterProp := Tensor0(0.5f)), Key(42))
      .take(100)
      .toList
      .last
      ._1

    println(s"MAP estimate: ${map(WaterProp)}")

    val nuts = NUTSAlgorithm[latent.Params](stepSize = 0.1)(latent.logDensity)
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

    val posteriorSamples = stack(nutsSamples.map(_(WaterProp)), Axis[Sample])
    display(
      histogramPlot(
        posteriorSamples,
        _.title := "Posterior distribution of water proportion"
      )
    )
