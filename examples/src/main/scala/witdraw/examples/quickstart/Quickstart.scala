package witdraw.examples.quickstart

import dimwit.*
import dimwit.stats.*
import witdraw.*
import witdraw.mcmc.NUTSAlgorithm
import witdraw.reparam.Bijector

object Quickstart:

  trait Feature derives Label
  trait Sample derives Label

  object Alpha extends Site[Tensor0[Float32]] // intercept
  object Beta extends Site[Tensor1[Feature, Float32]] // regression weights
  object Sigma extends Site[Tensor0[Float32]] // noise scale (> 0)
  object Y extends Site[Tensor1[Sample, Float32]] // outcomes

  def model(x: Tensor2[Sample, Feature, Float32])(using ProbContext) =
    val alpha = Alpha ~ Normal(0f, 10f)
    val beta = Beta ~ Normal.isotropic(
      Tensor1(x.extent(Axis[Feature])).fill(0f),
      10f
    )
    val sigma = Sigma ~ HalfCauchy(0f, 5f)
    val mu = x.dot(Axis[Feature])(beta) +! alpha
    val y = Y ~ Normal.isotropic(mu, sigma)
    (alpha = alpha, beta = beta, sigma = sigma, y = y)

  @main
  def runQuickstart(): Unit =
    dimwit.initialize()

    val (dataKey, noiseKey, priorKey, fixedKey, nutsKey) =
      val ks = Key(42).split(5)
      (ks(0), ks(1), ks(2), ks(3), ks(4))

    val sampleExtent = Axis[Sample] -> 100
    val featureExtent = Axis[Feature] -> 2

    // independent standard-normal features
    val xData =
      Normal.standardNormal(Shape(sampleExtent, featureExtent)).sample(dataKey)

    // y = alpha + x · beta + sigma * eps
    val alphaTrue = Tensor0(0.5f)
    val betaTrue = Tensor(Shape(featureExtent)).fromArray(Array(-1.5f, 2.0f))
    val sigmaTrue = Tensor0(0.8f)
    val eps = Normal.standardNormal(Shape(sampleExtent)).sample(noiseKey)
    val yData =
      (xData.dot(Axis[Feature])(betaTrue) +! alphaTrue) + (eps *! sigmaTrue)

    val priorPredictive =
      priorKey.split(5).map(k => ForwardSampler.run(k)(model(xData)))

    priorPredictive.foreach { s =>
      println(
        f"alpha=${s.alpha.item}%7.2f  sigma=${s.sigma.item}%7.2f  mean(y)=${s.y.mean.item}%8.2f"
      )
    }

    val withFixedSigma = fixedKey.split(5).map { k =>
      ForwardSampler.run(k):
        observe(Sigma := Tensor0(1f)):
          model(xData)
    }

    withFixedSigma.foreach { s =>
      println(f"sigma=${s.sigma.item}%4.2f  mean(y)=${s.y.mean.item}%8.2f")
    }

    val latent = Latent.discover(Constraint(Sigma, Bijector.exp)):
      observe(Y := yData):
        model(xData)

    val burnIn = 500
    val nSamples = 5000
    val nuts = NUTSAlgorithm[latent.Params](stepSize = 0.05)(latent.logDensity)
    val (samples, infos) = nuts
      .run(latent.init, nutsKey)
      .drop(burnIn)
      .take(nSamples)
      .toList
      .unzip

    trait Draw derives Label

    def summary[T <: Tuple: Labels](draws: List[Tensor[T, Float32]]) =
      val stacked = stack(draws, Axis[Draw])
      (stacked.mean(Axis[Draw]), stacked.std(Axis[Draw]))

    val (alphaMean, alphaSd) = summary(samples.map(_(Alpha)))
    val (betaMean, betaSd) = summary(samples.map(_(Beta)))
    val (sigmaMean, sigmaSd) = summary(samples.map(_(Sigma)))

    println(s"alpha: mean $alphaMean, sd $alphaSd   (true $alphaTrue)")
    println(s"beta:  mean $betaMean, sd $betaSd   (true $betaTrue)")
    println(s"sigma: mean $sigmaMean, sd $sigmaSd   (true $sigmaTrue)")
