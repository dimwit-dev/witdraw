package witdraw.examples.rethinking

import deepwit.base.ActivationFunction
import dimwit.*
import dimwit.stats.Bernoulli
import dimwit.stats.Exponential
import dimwit.stats.Normal
import dimwit.stats.Uniform
import viz.PlotTargets.websocket
import viz.macros.BoolField
import witdraw.ModelScorer
import witdraw.ProbProg
import witdraw.RandomVar
import witdraw.Site
import witdraw.dataframe.DataFrame
import witdraw.mcmc.NUTSAlgorithm
import witdraw.plotting.Plotting
import witdraw.plotting.Plotting.display

import java.io.File

object Chimpanzees:

  @main
  def chimps(): Unit =
    dimwit.initialize()

    val schema = (
      actor = DataFrame.col[Int],
      recipient = DataFrame.col[String],
      condition = DataFrame.col[Int],
      block = DataFrame.col[Int],
      trial = DataFrame.col[Int],
      prosoc_left = DataFrame.col[Int],
      chose_prosoc = DataFrame.col[Int],
      pulled_left = DataFrame.col[Int]
    )

    val dataDir = new File(System.getProperty("user.dir"))
    val df =
      DataFrame.fromCSV(new File(dataDir, "data/chimpanzees.csv"), schema)

    trait Trial derives Label // Each row is one experimental trial
    trait Chimp derives Label // The 7 individual chimpanzee subjects

    // Whether a partner chimp was present in this trial (the "condition")
    val hasPartner =
      Tensor1(Axis[Trial]).fromArray(df.col(_.condition).toArray)
    // Whether the prosocial food option was on the left side
    val prosocialOnLeft =
      Tensor1(Axis[Trial]).fromArray(df.col(_.prosoc_left).toArray)
    // Observed outcome: did the chimp pull the left lever?
    val pulledLeft =
      Tensor1(Axis[Trial]).fromArray(df.col(_.pulled_left == 1).toArray)
    val chimps = df.col(_.actor).toSet.toList.sorted

    // For each trial, the index of the chimp that performed it
    val chimpIndexPerTrial = df.col(row => chimps.indexOf(row.actor))

    val nTrials = hasPartner.shape(Axis[Trial])
    val nChimps = df.col(_.actor).toSet.size

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

    case class Result(
        chimpIntercept: ChimpIntercept.Value,
        populationIntercept: PopulationIntercept.Value,
        prosocialEffect: ProsocialEffect.Value,
        partnerEffect: PartnerEffect.Value,
        chimpVariability: ChimpVariability.Value,
        pulledLeft: PulledLeft.Value
    )

    val model = ProbProg[Result]:

      // How much extra pull-left tendency when a partner is present
      val partnerEffect = RandomVar(
        site = PartnerEffect,
        dist = Normal(Tensor0(0f), Tensor0(10f))
      )
      // Between-chimp spread — smaller = more pooling toward population mean
      val chimpVariability =
        RandomVar(site = ChimpVariability, dist = Exponential(Tensor0(1f)))

      // Per-chimp intercepts drawn from a common population distribution
      val chimpIntercept = RandomVar(
        site = ChimpIntercept,
        dist = Normal(
          Tensor1(Axis[Chimp] -> nChimps).fill(0f),
          Tensor1(Axis[Chimp] -> nChimps).fill(1f) *! chimpVariability
        )
      )
      // Shared baseline across all chimps
      val populationIntercept = RandomVar(
        site = PopulationIntercept,
        dist = Normal(Tensor0(0f), Tensor0(10f))
      )
      // Look up each trial's chimp intercept
      val chimpBaselinePerTrial: Tensor1[Trial, Float32] =
        chimpIntercept
          .relabelTo(Axis[Trial])
          .slice(Axis[Trial].at(chimpIndexPerTrial))

      // Effect of the prosocial option's side
      val prosocialEffect = RandomVar(
        site = ProsocialEffect,
        dist = Normal(Tensor0(0f), Tensor0(10f))
      )
      // Partner modulation: partnerEffect is only active when a partner is present
      val partnerModulation = partnerEffect *! hasPartner.asFloat32
      // Log-odds → probability of pulling left
      val probPullLeft = Prob(
        ActivationFunction.sigmoid(
          populationIntercept +! chimpBaselinePerTrial +
            ((prosocialEffect +! partnerModulation) * prosocialOnLeft.asFloat32)
        )
      )
      val pulledLeft =
        RandomVar(site = PulledLeft, dist = Bernoulli(probPullLeft))
      Result(
        chimpIntercept,
        populationIntercept,
        prosocialEffect,
        partnerEffect,
        chimpVariability,
        pulledLeft
      )

    case class Posterior(
        chimpIntercept: ChimpIntercept.Value,
        populationIntercept: PopulationIntercept.Value,
        prosocialEffect: ProsocialEffect.Value,
        partnerEffect: PartnerEffect.Value,
        chimpVariability: ChimpVariability.Value
    ) derives TensorTree

    def logDensity(posteriorParams: Posterior) =
      val observedModel = model.observe(
        PulledLeft := pulledLeft,
        ChimpIntercept := posteriorParams.chimpIntercept,
        PopulationIntercept := posteriorParams.populationIntercept,
        ProsocialEffect := posteriorParams.prosocialEffect,
        PartnerEffect := posteriorParams.partnerEffect,
        ChimpVariability := posteriorParams.chimpVariability
      )

      ModelScorer.run(observedModel)

    val burnIn = 2000
    val nSamples = 5000
    val nuts = NUTSAlgorithm[Posterior](stepSize = 0.1)(logDensity)
    val (nutsSamples, nutsInfos) = nuts
      .run(
        Posterior(
          Tensor1(Axis[Chimp] -> nChimps).fill(0f),
          populationIntercept = Tensor0(0f),
          prosocialEffect = Tensor0(0f),
          partnerEffect = Tensor0(0f),
          chimpVariability = Tensor0(1f)
        ),
        Key(42)
      )
      .drop(burnIn)
      .take(nSamples)
      .toList
      .unzip

    // val pChimpIntercept =
    //   Plotting.plotHistogram(
    //     stack(nutsSamples.map(_.chimpIntercept), Axis[Trial]),
    //     title = Some("Posterior of chimpIntercept")
    //   )
    val pProsocialEffect =
      Plotting.plotHistogram(
        stack(nutsSamples.map(_.prosocialEffect), Axis[Trial]),
        title = Some("Posterior of prosocialEffect")
      )
    val pPartnerEffect =
      Plotting.plotHistogram(
        stack(nutsSamples.map(_.partnerEffect), Axis[Trial]),
        title = Some("Posterior of partnerEffect")
      )

    val chimp0Plot = Plotting.plotHistogram(
      stack(
        nutsSamples.map(_.chimpIntercept.slice(Axis[Chimp].at(0))),
        Axis[Trial]
      ),
      title = Some("Posterior of chimpIntercept[0]")
    )
    val chimp1Plot = Plotting.plotHistogram(
      stack(
        nutsSamples.map(_.chimpIntercept.slice(Axis[Chimp].at(1))),
        Axis[Trial]
      ),
      title = Some("Posterior of chimpIntercept[1]")
    )
    val allPlots = Plotting.grid(
      List(List(chimp0Plot, pProsocialEffect), List(pPartnerEffect, chimp1Plot))
    )
    display(allPlots)
