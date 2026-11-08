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

    val latent = Latent.discover()(conditioned)
    println(s"Latent sites: ${latent.sites.map(_.name).mkString(", ")}")

    val numWarmup = 500
    val nSamples = 2000

    // The warmup tunes the step size and the mass matrix.
    val adapted =
      NUTS.adapt(latent, warmupKey, latent.init, numSteps = numWarmup)

    println(f"NUTS adapted step size: ${adapted.stepSize}%.3f")
    val chain = adapted.sample(sampleKey, nSamples)

    println(f"NUTS divergences: ${chain.numDivergent} / $nSamples")
    val acceptance = chain.infos.map(_.acceptanceRate)
    println(
      f"NUTS mean acceptance rate: ${acceptance.sum / acceptance.size}%.3f"
    )

    // Posterior predictive check
    val predictiveKeys = predictiveKey.split(chain.size)
    val posteriorPredictiveSamples = chain.positions
      .zip(predictiveKeys)
      .map: (p, key) =>
        ForwardSampler.run(key):
          observe(WaterProp := p(WaterProp))(model(numTosses))

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
            histogram(chain(WaterProp), "posterior samples"),
            histogram(
              posteriorPredictiveSamples.map(_.waterCount.asFloat32),
              s"posterior predictive samples (observed: ${waterCountData.item})"
            )
          ),
          Seq(
            tracePlot(Seq(chain))(WaterProp)
          )
        )
      )
    )
    println("check your browser to see the plot")
