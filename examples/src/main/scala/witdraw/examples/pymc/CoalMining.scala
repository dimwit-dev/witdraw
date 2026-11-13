package witdraw
package examples.pymc

import dimwit.*
import dimwit.stats.{DiscreteUniform, Exponential, Poisson}

import witdraw.*
import witdraw.inference.NUTS

import scala.concurrent.{Await, Future, blocking}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration

import plotwit.*
import plotwit.plots.{linePlot, scatterPlot}
import plotwit.PlotTargets.desktopBrowser
import witdraw.plots.TracePlot.tracePlot
import witdraw.plots.Summary.summary

/** The coal mining disasters switchpoint model, after the PyMC example "Coal
  * mining disasters": the yearly number of disasters in the UK from 1851 to
  * 1961 is Poisson, with one rate up to an unknown switchpoint year and another
  * one after it.
  *
  * The model is the same as in PyMC; inference differs. PyMC samples the
  * discrete switchpoint with a Metropolis step. Here, inference uses
  * `p(rates, switchpoint | data) = p(rates | data) p(switchpoint | rates, data)`:
  *   1. NUTS samples `p(rates | data)`, with the switchpoint summed out by
  *      `marginalize`, which is exact as it takes only 111 values.
  *   1. `recover` completes every draw with a switchpoint drawn from
  *      `p(switchpoint | rates, data)`, which is exact too.
  */
object CoalMining:

  trait Year derives Label // one entry per year with an observed count

  object SwitchPoint extends Site[Tensor0[Int32]]
  object EarlyRate extends Site[Tensor0[Float32]]
  object LateRate extends Site[Tensor0[Float32]]
  object Disasters extends Site[Tensor1[Year, Int32]]

  def model(years: Tensor1[Year, Int32])(using ProbContext) =
    // the upper bound is exclusive: the switchpoint is one of 1851 to 1961
    val switchpoint = SwitchPoint ~ DiscreteUniform(1851, 1962)
    val earlyRate = EarlyRate ~ Exponential(1f)
    val lateRate = LateRate ~ Exponential(1f)
    val rate = where_!(switchpoint >=! years, earlyRate, lateRate)
    val disasters = Disasters ~ Poisson(rate)
    (
      switchpoint = switchpoint,
      earlyRate = earlyRate,
      lateRate = lateRate,
      rate = rate,
      disasters = disasters
    )

  @main
  def runCoalMining(): Unit =
    dimwit.initialize()

    // format: off
    val disaster_data = Seq(
        4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
        3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
        2, 2, 3, 4, 2, 1, 3, Float.NaN, 2, 1, 1, 1, 1, 3, 0, 0,
        1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
        0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
        3, 3, 1, Float.NaN, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1
    )
    // format: on
    val years = Seq.range(1851, 1962)
    val df =
      for (numDisasters, year) <- disaster_data.zip(years).toVector
      yield (
        numDisasters =
          if numDisasters.isNaN() then None else Some(numDisasters),
        year = year
      )

    trait Disaster derives Label
    display(
      scatterPlot(
        Tensor1(Axis[Disaster]).fromArray(
          df.map(_.year.toFloat).toArray
        ),
        Tensor1(Axis[Disaster]).fromArray(
          df.map(_.numDisasters.getOrElse(0f)).toArray
        ),
        df.map(d =>
          if d.numDisasters.isEmpty then "missing (NaN)" else "observed"
        )
      )
    )

    // PyMC imputes the two missing counts; here they are left out. The
    // switchpoint may still be any year.
    val observed = df.collect { case (numDisasters = Some(n), year = y) =>
      (n.toInt, y)
    }
    val observedYears =
      Tensor1(Axis[Year]).fromArray(observed.map(_._2).toArray)
    val counts = Tensor1(Axis[Year]).fromArray(observed.map(_._1).toArray)

    // Step 1: NUTS on p(rates | data), with the switchpoint summed out
    val posterior =
      Latent.discover:
        marginalize(SwitchPoint):
          observe(Disasters := counts)(model(observedYears))
    println(
      s"latent sites: ${posterior.sites.map(_.name).mkString(", ")}, " +
        s"summed out: ${posterior.marginalizedSites.map(_.name).mkString(", ")}"
    )

    val numChains = 4
    val numDraws = 1000
    val Seq(initKey, chainsKey, recoverKey) = Random.Key(42).split(3)
    val starts = initKey.split(numChains).map(posterior.initRandom(_))
    def chain(key: Random.Key, start: posterior.Params) =
      val (warmupKey, sampleKey) = key.split2()
      val adapted = NUTS.adapt(posterior, warmupKey, start, numSteps = 1000)
      adapted.sample(sampleKey, numDraws)

    val runs = chainsKey
      .split(numChains)
      .zip(starts)
      .map((key, start) => Future(blocking(chain(key, start))))
    val chains = Await.result(Future.sequence(runs), Duration.Inf)
    println(
      s"NUTS divergences: ${chains.map(_.numDivergent).sum} / ${numChains * numDraws}"
    )

    // Step 2: complete every draw with switchpoint ~ p(switchpoint | rates, data)
    val recovered = posterior.recover(chains, recoverKey)

    display(
      tracePlot(recovered)(s =>
        (
          switchpoint = s.switchpoint.asFloat32,
          earlyRate = s.earlyRate,
          lateRate = s.lateRate
        )
      )
    )
    println(
      summary(recovered)(s =>
        (
          switchpoint = s.switchpoint.asFloat32,
          earlyRate = s.earlyRate,
          lateRate = s.lateRate
        )
      )
    )

    // Instead of the recovered draws, the exact posterior of the switchpoint
    // given the rates, averaged over the draws: the same posterior, with less
    // noise
    val switchpointPosterior = posterior.posteriorOf(SwitchPoint, chains)
    val switchpoints = posterior.valuesOf(SwitchPoint)
    println("most likely switchpoints (last year of the early rate):")
    for (year, p) <- switchpoints.toArray
        .zip(switchpointPosterior.toArray)
        .sortBy(-_._2)
        .take(5)
    do println(f"  $year: $p%.3f")
    display(
      linePlot(
        switchpoints.asFloat32,
        switchpointPosterior,
        _.title := "posterior of the switchpoint"
      )
    )
