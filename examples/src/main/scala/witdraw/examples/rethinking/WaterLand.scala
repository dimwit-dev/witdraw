package witdraw
package examples.rethinking

import dimwit.*
import dimwit.stats.*

import witdraw.*
import witdraw.inference.NUTS
import witdraw.plots.TracePlot.tracePlot

import plotwit.PlotTargets.desktopBrowser
import plotwit.{display, grid}
import plotwit.plots.histogramPlot
import scala.concurrent.{Future, blocking}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Await
import scala.concurrent.duration.Duration
import witdraw.inference.NUTSAdaptation
import witdraw.ProbContext

object WaterLandExample:

  object WaterProp extends Site[Tensor0[Float32]]
  object WaterCount extends Site[Tensor0[Int32]]

  def model(numTosses: Int)(using ProbContext) =
    val wp = WaterProp ~ Uniform(0f, 1f)
    val wc = WaterCount ~ Binomial(numTosses, Prob(wp))
    (waterProp = wp, waterCount = wc)

  @main
  def runWaterLand(): Unit =
    dimwit.initialize()
    val numTosses = 9
    val waterCountData = Tensor0(6)

    trait Sample derives Label

    // one key for each random step, so that they do not share randomness
    val Seq(priorKey, warmupKey, sampleKey, predictiveKey) = Key(42).split(4)

    val samples = ForwardSampler.sample(priorKey, 1000):
      model(numTosses)

    val priorSamples = samples.map(_.waterProp)
    val priorPredictiveSamples = samples.map(_.waterCount)

    def conditioned(using ProbContext) =
      observe(WaterCount := waterCountData)(model(numTosses))

    val latent = Latent.discover(conditioned)
    println(s"Latent sites: ${latent.sites.map(_.name).mkString(", ")}")

    val numWarmup = 500
    val nSamples = 2000

    // The warmup tunes the step size and the mass matrix.
    val adapted =
      NUTS.adapt(latent, warmupKey, latent.init, numSteps = numWarmup)

    println(f"NUTS adapted step size: ${adapted.stepSize}%.3f")
    val runs =
      for key <- sampleKey.split(4)
      yield Future(blocking(adapted.sample(key, nSamples)))
    val chains = Await.result(Future.sequence(runs), Duration.Inf)

    // the model's result at each draw, as for the prior samples
    val posterior = chains.flatMap(_.samples)
    val posteriorSamples = posterior.map(_.waterProp)

    // Posterior predictive check: the model run forward, with the parameters
    // of each posterior draw
    val posteriorPredictiveSamples =
      posterior.zip(predictiveKey.split(posterior.size)).map: (s, key) =>
        ForwardSampler.run(key):
          observe(WaterProp := s.waterProp)(model(numTosses))

    def histogram(values: Seq[Tensor0[Float32]], title: String) =
      histogramPlot(stack(values, Axis[Sample]), _.title := title)
    display(
      grid(
        Seq(
          Seq(
            histogram(priorSamples, "prior samples"),
            histogram(
              priorPredictiveSamples.map(_.asFloat32),
              "prior predictive samples"
            ),
            histogram(posteriorSamples, "posterior samples"),
            histogram(
              posteriorPredictiveSamples.map(_.waterCount.asFloat32),
              s"posterior predictive samples (observed: ${waterCountData.item})"
            )
          ),
          Seq(
            tracePlot(chains)(s => (waterProp = s.waterProp))
          )
        )
      )
    )
    println("check your browser to see the plot")
