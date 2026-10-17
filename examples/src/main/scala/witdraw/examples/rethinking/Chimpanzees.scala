package witdraw.examples.rethinking

import dimwit.*
import dimwit.stats.Bernoulli
import dimwit.stats.Exponential
import dimwit.stats.Normal
import dimwit.stats.Uniform
import viz.PlotTargets.websocket
import viz.macros.BoolField
import witdraw.Constraint
import witdraw.Latent
import witdraw.ProbContext
import witdraw.Site
import witdraw.observe
import witdraw.reparam.Bijector
import witling.DataFrame
import witdraw.mcmc.NUTSAlgorithm
import plotwit.PlotTargets.websocket
import plotwit.{display, grid}

import java.io.File
import plotwit.plots.histogramPlot

object Chimpanzees:

  @main
  def chimps(): Unit =
    dimwit.initialize()

    type Schema = (
        actor: Int,
        recipient: String,
        condition: Int,
        block: Int,
        trial: Int,
        prosoc_left: Int,
        chose_prosoc: Int,
        pulled_left: Int
    )

    val dataDir = new File(System.getProperty("user.dir"))
    val df =
      DataFrame.fromCSV[Schema](new File(dataDir, "data/chimpanzees.csv"))

    trait Trial derives Label // Each row is one experimental trial
    trait Chimp derives Label // The 7 individual chimpanzee subjects

    // Whether a partner chimp was present in this trial (the "condition")
    val hasPartner =
      Tensor1(Axis[Trial]).fromArray(df.cols.condition.toArray)
    // Whether the prosocial food option was on the left side
    val prosocialOnLeft =
      Tensor1(Axis[Trial]).fromArray(df.cols.prosoc_left.toArray)
    // Observed outcome: did the chimp pull the left lever?
    val pulledLeft =
      Tensor1(Axis[Trial]).fromArray(
        df.cols.pulled_left.toArray.map(b => if b == 1 then true else false)
      )
    val chimps = df.cols.actor.toSet.toList.sorted

    // For each trial, the index of the chimp that performed it
    val chimpIndexPerTrial = df.cols.actor.map(actor => chimps.indexOf(actor))

    val nTrials = hasPartner.shape(Axis[Trial])
    val nChimps = df.cols.actor.toSet.size

    object PulledLeft extends Site[Tensor1[Trial, Bool]]

    // Per-chimp baseline log-odds of pulling left (partial pooling via hierarchical prior)
    object ChimpIntercept extends Site[Tensor1[Chimp, Float32]]
    // Population-level mean log-odds (the grand intercept)
    object PopulationIntercept extends Site[Tensor0[Float32]]
    // Effect of the prosocial option being on the left
    object ProsocialEffect extends Site[Tensor0[Float32]]
    // Additional effect when a partner is present (interaction term)
    object PartnerEffect extends Site[Tensor0[Float32]]
    // Between-chimp standard deviation (controls partial pooling strength)
    object ChimpVariability extends Site[Tensor0[Float32]]

    def model(using ProbContext) =
      // How much extra pull-left tendency when a partner is present
      val partnerEffect = PartnerEffect ~ Normal(Tensor0(0f), Tensor0(10f))

      // Between-chimp spread — smaller = more pooling toward population mean
      val chimpVariability = ChimpVariability ~ Exponential(Tensor0(1f))

      // Per-chimp intercepts drawn from a common population distribution
      val chimpIntercept = ChimpIntercept ~ Normal(
        Tensor1(Axis[Chimp] -> nChimps).fill(0f),
        Tensor1(Axis[Chimp] -> nChimps).fill(1f) *! chimpVariability
      )

      // Shared baseline across all chimps
      val populationIntercept =
        PopulationIntercept ~ Normal(Tensor0(0f), Tensor0(10f))
      // Look up each trial's chimp intercept
      val chimpBaselinePerTrial: Tensor1[Trial, Float32] =
        chimpIntercept
          .relabelTo(Axis[Trial])
          .slice(Axis[Trial].at(chimpIndexPerTrial))

      // Effect of the prosocial option's side
      val prosocialEffect = ProsocialEffect ~ Normal(Tensor0(0f), Tensor0(10f))

      // Partner modulation: partnerEffect is only active when a partner is present
      val partnerModulation = partnerEffect *! hasPartner.asFloat32
      // Log-odds → probability of pulling left
      val probPullLeft = Prob(
        Tensor.sigmoid(
          populationIntercept +! chimpBaselinePerTrial +
            ((prosocialEffect +! partnerModulation) * prosocialOnLeft.asFloat32)
        )
      )
      val pulledLeft = PulledLeft ~ Bernoulli(probPullLeft)
      (
        chimpIntercept = chimpIntercept,
        populationIntercept = populationIntercept,
        prosocialEffect = prosocialEffect,
        partnerEffect = partnerEffect,
        chimpVariability = chimpVariability,
        pulledLeft = pulledLeft
      )

    // The between-chimp spread is a scale, so it is sampled on the log scale.
    val posterior =
      Latent.discover(Key(1), Constraint(ChimpVariability, Bijector.exp)):
        observe(PulledLeft := pulledLeft)(model)
    println(s"latent sites: ${posterior.sites.map(_.name).mkString(", ")}")

    val burnIn = 2000
    val nSamples = 5000
    val nuts =
      NUTSAlgorithm[posterior.Params](stepSize = 0.1)(posterior.logDensity)
    val (nutsSamples, nutsInfos) = nuts
      .run(posterior.init, Key(42))
      .drop(burnIn)
      .take(nSamples)
      .toList
      .unzip
    println(s"NUTS divergences: ${nutsInfos.count(_.isDivergent)} / $nSamples")
    for site <- List(
        PopulationIntercept,
        ProsocialEffect,
        PartnerEffect,
        ChimpVariability
      )
    do
      val draws = stack(nutsSamples.map(_(site)), Axis[Trial])
      println(
        f"${site.name}%-20s mean ${draws.mean.item}%7.3f  sd ${draws.std.item}%6.3f"
      )

    // val pChimpIntercept =
    //   Plotting.plotHistogram(
    //     stack(nutsSamples.map(_.chimpIntercept), Axis[Trial]),
    //     title = Some("Posterior of chimpIntercept")
    //   )
    val pProsocialEffect =
      histogramPlot(
        stack(nutsSamples.map(_(ProsocialEffect)), Axis[Trial]),
        _.title := "Posterior of prosocialEffect"
      )
    val pPartnerEffect =
      histogramPlot(
        stack(nutsSamples.map(_(PartnerEffect)), Axis[Trial]),
        _.title := "Posterior of partnerEffect"
      )

    val chimp0Plot = histogramPlot(
      stack(
        nutsSamples.map(_(ChimpIntercept).slice(Axis[Chimp].at(0))),
        Axis[Trial]
      ),
      _.title := "Posterior of chimpIntercept[0]"
    )
    val chimp1Plot = histogramPlot(
      stack(
        nutsSamples.map(_(ChimpIntercept).slice(Axis[Chimp].at(1))),
        Axis[Trial]
      ),
      _.title := "Posterior of chimpIntercept[1]"
    )
    val allPlots = grid(
      List(List(chimp0Plot, pProsocialEffect), List(pPartnerEffect, chimp1Plot))
    )
    display(allPlots)
