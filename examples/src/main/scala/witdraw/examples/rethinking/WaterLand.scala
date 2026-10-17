package witdraw
package examples.rethinking

import dimwit.*
import dimwit.stats.*
import witdraw.*
import witdraw.inference.{Laplace, MAP, NUTS}

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

    val map = MAP(latent, start = Seq(WaterProp := Tensor0(0.5f)))
    println(f"MAP estimate: ${map(WaterProp).item}%.3f   (exact 6/9 = 0.667)")

    val nuts = NUTS(latent, stepSize = 0.1)
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

    val laplace = Laplace(latent)
    val laplaceSamples = laplace.run(map, Key(43)).take(nSamples).map(_._1).toList

    // With a uniform prior and 6 waters in 9 tosses, the posterior is Beta(7, 4).
    def summary(draws: List[latent.Params]): String =
      val ps = draws.map(_(WaterProp).item)
      val mean = ps.sum / ps.size
      val sd = math.sqrt(ps.map(p => (p - mean) * (p - mean)).sum / ps.size)
      f"mean $mean%.3f, sd $sd%.3f"
    println(s"NUTS:    ${summary(nutsSamples)}   (exact Beta(7, 4): mean 0.636, sd 0.139)")
    println(s"Laplace: ${summary(laplaceSamples)}")

    def histogram(draws: List[latent.Params], title: String) =
      histogramPlot(stack(draws.map(_(WaterProp)), Axis[Sample]), _.title := title)
    display(
      grid(
        List(
          List(
            histogram(nutsSamples, "Water proportion: NUTS"),
            histogram(laplaceSamples, "Water proportion: Laplace")
          )
        )
      )
    )
