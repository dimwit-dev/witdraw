package witdraw.examples.quickstart

import dimwit.*
import dimwit.stats.*
import witdraw.*
import witdraw.inference.NUTS
import witdraw.inference.Summary.summary

import scala.concurrent.{Await, Future, blocking}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration

/** The code of the quickstart guide, docs/mdoc/quickstart.md, as one program:
  * Bayesian linear regression, from the joint distribution to the posterior and
  * the posterior predictive check.
  */
object Quickstart:

  trait Feature derives Label
  trait Sample derives Label

  // one site per random variable: alpha, beta, sigma and y
  object Alpha extends Site[Tensor0[Float32]] // intercept
  object Beta extends Site[Tensor1[Feature, Float32]] // regression weights
  object Sigma extends Site[Tensor0[Float32]] // noise scale (> 0)
  object Y extends Site[Tensor1[Sample, Float32]] // outcomes

  /** The joint distribution p(alpha, beta, sigma, y | x): one `~` statement
    * per line of the model, the one for `Y` covering all samples at once.
    */
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

    // Features: the model's inputs, drawn at random.

    val (featureKey, priorKey, fixedKey, dataKey, nutsKey, replicateKey) =
      val ks = Key(42).split(6)
      (ks(0), ks(1), ks(2), ks(3), ks(4), ks(5))

    val sampleExtent = Axis[Sample] -> 100
    val featureExtent = Axis[Feature] -> 2

    // independent standard-normal features
    val xData =
      Normal
        .standardNormal(Shape(sampleExtent, featureExtent))
        .sample(featureKey)

    // Drawing from the joint distribution: the y of each draw is a draw from
    // the prior predictive distribution p(y | x).

    val priorPredictive = ForwardSampler.sample(priorKey, 5)(model(xData))

    println("Prior predictive draws:")
    priorPredictive.foreach { s =>
      println(
        f"alpha=${s.alpha.item}%7.2f  sigma=${s.sigma.item}%7.2f  mean(y)=${s.y.mean.item}%8.2f"
      )
    }

    // Fixing a variable: draws from p(alpha, beta, y | sigma = 1, x).

    val withFixedSigma =
      ForwardSampler.sample(fixedKey, 5):
        observe(Sigma := Tensor0(1f)):
          model(xData)

    println("Draws with sigma fixed to 1:")
    withFixedSigma.foreach { s =>
      println(f"sigma=${s.sigma.item}%4.2f  mean(y)=${s.y.mean.item}%8.2f")
    }

    // Simulating data from known parameters: one draw from
    // p(y | alpha, beta, sigma, x) at the true values.

    val alphaTrue = Tensor0(0.5f)
    val betaTrue = Tensor(Shape(featureExtent)).fromArray(Array(-1.5f, 2.0f))
    val sigmaTrue = Tensor0(0.8f)

    val yData =
      ForwardSampler
        .run(dataKey):
          observe(Alpha := alphaTrue, Beta := betaTrue, Sigma := sigmaTrue):
            model(xData)
        .y

    // Conditioning on the data: theta => log p(theta, y_obs | x), the log
    // posterior up to a constant.

    def logJoint(alpha: Float, beta: Array[Float], sigma: Float): Float =
      ModelScorer
        .run:
          observe(
            Y := yData,
            Alpha := Tensor0(alpha),
            Beta := Tensor1(Axis[Feature]).fromArray(beta),
            Sigma := Tensor0(sigma)
          ):
            model(xData)
        .item

    println(
      f"log p(theta, y_obs | x) at the true parameters: ${logJoint(0.5f, Array(-1.5f, 2.0f), 0.8f)}%.2f"
    )
    println(
      f"log p(theta, y_obs | x) at alpha = 0, beta = 0, sigma = 1: ${logJoint(0f, Array(0f, 0f), 1f)}%.2f"
    )

    // Latent finds the parameters of this function: the sites not observed.
    val latent = Latent.discover:
      observe(Y := yData):
        model(xData)

    println(s"Latent sites: ${latent.sites.map(_.name).mkString(", ")}")

    // Running NUTS: a chain of draws from the posterior.

    val (warmupKey, sampleKey) = nutsKey.split2()
    val adapted = NUTS.adapt(latent, warmupKey, latent.init, numSteps = 1000)
    val nSamples = 2000
    val nutsChain = adapted.sample(sampleKey, nSamples)
    // the model's result at each draw, as for the prior predictive draws
    val samples = nutsChain.samples

    // Checking the result: posterior means, standard deviations and credible
    // intervals, estimated from the draws.

    println(
      summary(Seq(nutsChain))(s =>
        (alpha = s.alpha, beta = s.beta, sigma = s.sigma)
      )
    )
    println(s"true values: alpha $alphaTrue, beta $betaTrue, sigma $sigmaTrue")

    val meanAcceptance =
      nutsChain.infos.map(_.acceptanceRate).sum / nutsChain.size
    println(
      f"divergences: ${nutsChain.numDivergent}, mean acceptance rate: $meanAcceptance%.2f"
    )

    // Running several chains, from random starting points, at the same time.

    def chain(key: Key, start: latent.Params) =
      val (warmupKey, sampleKey) = key.split2()
      val adapted = NUTS.adapt(latent, warmupKey, start)
      adapted.sample(sampleKey, 1000)

    val Seq(initKey, chainsKey) = nutsKey.split(2)
    val starts = initKey.split(4).map(latent.initRandom(_))
    val runs = chainsKey
      .split(4)
      .zip(starts)
      .map((key, start) => Future(blocking(chain(key, start))))
    val chains = Await.result(Future.sequence(runs), Duration.Inf)

    println(
      summary(chains)(s => (alpha = s.alpha, beta = s.beta, sigma = s.sigma))
    )

    // Asking the posterior new questions: posterior predictive draws, by
    // fixing the parameters to posterior draws and running forwards.

    val thinned = samples.grouped(10).map(_.head).toSeq
    val replicated =
      replicateKey.split(thinned.size).zip(thinned).map { (key, s) =>
        ForwardSampler
          .run(key):
            observe(Alpha := s.alpha, Beta := s.beta, Sigma := s.sigma):
              model(xData)
          .y
      }

    // posterior predictive check with T(y) = sd(y)
    val replicatedSd = replicated.map(_.std.item)
    val observedSd = yData.std.item
    val pValue = replicatedSd.count(_ > observedSd).toFloat / replicatedSd.size
    println(
      f"sd of the data: $observedSd%.3f, posterior predictive p-value: $pValue%.2f"
    )
