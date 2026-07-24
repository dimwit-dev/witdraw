package witdraw.examples.rethinking

import dimwit.*
import witdraw.Site
import dimwit.stats.Normal
import witdraw.ProbProg
import witdraw.RandomVar
import dimwit.stats.MVNormal
import witdraw.ForwardSampler
import witdraw.plotting.Plotting
import viz.PlotTargets.websocket

object Cafe:

  @main
  def cafe(): Unit =
    dimwit.initialize()

    val avgMorningWaitTime = Tensor0(3.5f)
    val avgDifferenceAfternoonMorning = Tensor0(-1f)
    val stddevMorningWaitTime = Tensor0(1f)
    val stddevDifferenceAfternoonMorning = Tensor0(1f)
    val correlation = Tensor0(-0.7f)

    trait A derives Label
    trait B derives Label
    object Mu extends Site[Tensor1[A, Float32]]
    object Sigma extends Site[Tensor2[A, Prime[A], Float32]]
    object Effects extends Site[Tensor1[A, Float32]]

    case class Result(
        effects: Effects.Value
    )

    val model = ProbProg[Result]:
      val mu = stack(
        Seq(
          avgMorningWaitTime,
          avgMorningWaitTime + avgDifferenceAfternoonMorning
        ),
        Axis[A]
      )

      val Rho = stack(
        Seq(
          stack(Seq(Tensor0(1f), correlation), Axis[Prime[A]]),
          stack(Seq(correlation, Tensor0(1f)), Axis[Prime[A]])
        ),
        Axis[A]
      )
      val sigma =
        stack(
          Seq(stddevMorningWaitTime, stddevDifferenceAfternoonMorning),
          Axis[A]
        )
      val cov = sigma.outerProduct(sigma) * Rho

      val nCafes = 20
      val n = MVNormal(mu, cov)
      val effects = RandomVar(Effects, n)
      Result(effects)

    trait Sample derives Label
    val samples =
      Key(42).splitvmap(Axis[Sample] -> 20)(k =>
        ForwardSampler.run(k)(model).effects
      )

    val plot = Plotting.plotScatter(
      samples.slice(Axis[A].at(0)),
      samples.slice(Axis[A].at(1))
    )
    Plotting.display(plot)
