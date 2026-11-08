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

    val samples = ForwardSampler.sample(Key(42), 1000):
      model(numTosses)

    val priorSamples = samples.map(_.waterProp)
    val priorPredictiveSamples = samples.map(_.waterCount)

    val predictiveSamples =
      ForwardSampler.sample(Key(42), 1000):
        observe(WaterProp := Prob(0.7f))(model(numTosses))

    def conditioned(using ProbContext) =
      observe(WaterCount := waterCountData)(model(numTosses))

    val latent = Latent.discover()(conditioned)
    println(s"Latent sites: ${latent.sites.map(_.name).mkString(", ")}")

    val numWarmup = 500
    val nSamples = 2000

    // The warmup tunes the step size and the mass matrix.
    val (warmupKey, sampleKey) = Key(42).split2()
    val adapted =
      NUTS.adapt(latent, warmupKey, latent.init, numSteps = numWarmup)

    println(f"NUTS adapted step size: ${adapted.stepSize}%.3f")
    val chain = adapted.sample(sampleKey, nSamples)

    println(f"NUTS divergences: ${chain.numDivergent} / $nSamples")
    val acceptance = chain.infos.map(_.acceptanceRate)
    println(
      f"NUTS mean acceptance rate: ${acceptance.sum / acceptance.size}%.3f"
    )

    def histogram(values: Seq[Tensor0[Float32]], title: String) =
      histogramPlot(stack(values, Axis[Sample]), _.title := title)
    display(
      grid(
        Seq(
          Seq(
            histogram(priorSamples, "prior samples"),
            histogram(
              predictiveSamples.map(_.waterCount.asFloat32),
              "predictive samples"
            ),
            histogram(chain(WaterProp), "posterior samples")
          ),
          Seq(
            tracePlot(Seq(chain))(WaterProp)
          )
        )
      )
    )
    println("check your browser to see the plot")
