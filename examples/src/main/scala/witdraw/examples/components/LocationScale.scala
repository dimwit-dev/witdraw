package witdraw.examples.components

import dimwit.*
import dimwit.stats.*
import dimwit.tensortree.TreeOf.foreachWithName
import dimwit.tensortree.TreeOf.ops.*
import witdraw.*
import witdraw.inference.NUTS

/** A model built from [[Component]]s: a location-scale regression for two
  * outcomes, e.g. blood pressure and cholesterol, measured on the same
  * patients.
  *
  * Each outcome has a mean and a scale that both depend on the features, the
  * scale through a log link:
  * {{{
  * y_n ~ Normal(a + x_n · b,  exp(c + x_n · d))
  * }}}
  *
  * The building blocks are components, written as modules: a trait declares a
  * part, and each object that extends it is one instance of it.
  *   - [[Linear]]: an intercept and weights, `intercept + x · weights`.
  *   - [[Outcome]]: two `Linear`s, one for the mean and one for the log-scale,
  *     and the outcome itself.
  *
  * The model has two `Outcome`s, so four `Linear`s. Each instance has its own
  * sites, named by its path, e.g. `bloodPressure.scale.Weights`.
  *
  * Each part also has a typed record of its values, e.g. [[Linear.Values]],
  * which nest like the parts do. A record can be read from a run of the model
  * or from a posterior draw, turned into observations, and, being a tensor
  * tree, averaged as a whole.
  */
object LocationScale:

  trait Patient derives Label // one entry per patient
  trait Feature derives Label // one entry per feature, e.g. age and weight

  /** `intercept + x · weights`, with normal priors of scale `priorScale`. */
  trait Linear(priorScale: Float) extends Component:
    object Intercept extends Site[Tensor0[Float32]]
    object Weights extends Site[Tensor1[Feature, Float32]]

    def apply(x: Tensor2[Patient, Feature, Float32])(using
        ProbContext
    ): Tensor1[Patient, Float32] =
      val nFeatures = x.shape(Axis[Feature])
      val intercept = Intercept ~ Normal(0f, priorScale)
      val weights = Weights ~ Normal.isotropic(
        Tensor1(Axis[Feature] -> nFeatures).fill(0f),
        priorScale
      )
      x.dot(Axis[Feature])(weights) +! intercept

    def values(v: SiteValues) = Linear.Values(v(Intercept), v(Weights))
    def observations(x: Linear.Values) =
      Seq(Intercept := x.intercept, Weights := x.weights)

  object Linear:
    case class Values(
        intercept: Tensor0[Float32],
        weights: Tensor1[Feature, Float32]
    )

    def values(intercept: Float, weights: Float*): Values =
      Values(Tensor0(intercept), Tensor1(Axis[Feature]).fromArray(weights.toArray))

  /** One outcome: its mean and log-scale are linear in the features. The
    * log-scale has tighter priors, as it acts through `exp`.
    */
  trait Outcome extends Component:
    object mean extends Linear(priorScale = 1f)
    object scale extends Linear(priorScale = 0.5f)
    object Y extends Site[Tensor1[Patient, Float32]]

    def apply(x: Tensor2[Patient, Feature, Float32])(using ProbContext) =
      Y ~ Normal(mean(x), scale(x).exp)

    def values(v: SiteValues) = Outcome.Values(mean.values(v), scale.values(v))
    def observations(x: Outcome.Values) =
      mean.observations(x.mean) ++ scale.observations(x.scale)

  object Outcome:
    case class Values(mean: Linear.Values, scale: Linear.Values)

  object bloodPressure extends Outcome
  object cholesterol extends Outcome

  def model(x: Tensor2[Patient, Feature, Float32])(using ProbContext) =
    (bloodPressure = bloodPressure(x), cholesterol = cholesterol(x))

  /** The values of all parameters of the model. */
  case class Values(bloodPressure: Outcome.Values, cholesterol: Outcome.Values)

  def values(v: SiteValues) =
    Values(bloodPressure.values(v), cholesterol.values(v))
  def observations(x: Values) =
    bloodPressure.observations(x.bloodPressure) ++
      cholesterol.observations(x.cholesterol)

  // -- runnable example -------------------------------------------------

  @main
  def locationScaleExample(): Unit =
    dimwit.initialize()

    // ---- simulated data -----------------------------------------------
    val nPatients = 300
    val (kx, ky) = Key(0L).split2()
    val x = Normal
      .isotropic(Tensor2(Axis[Patient] -> nPatients, Axis[Feature] -> 2).fill(0f), 1f)
      .sample(kx)

    // The true values, as one typed record. Observing them and running the
    // model forward draws the outcomes.
    val truth = Values(
      bloodPressure = Outcome.Values(
        mean = Linear.values(1.0f, 0.8f, -0.5f),
        scale = Linear.values(-0.7f, 0.4f, 0.0f)
      ),
      cholesterol = Outcome.Values(
        mean = Linear.values(-1.0f, 0.0f, 1.2f),
        scale = Linear.values(0.0f, 0.0f, -0.3f)
      )
    )
    val ys = ForwardSampler.run(ky)(observe(observations(truth)*)(model(x)))

    // ---- inference ----------------------------------------------------
    // All parameters are unconstrained: the scales are positive through the
    // log link, so no constraint is needed.
    val latent = Latent.discover():
      observe(
        bloodPressure.Y := ys.bloodPressure,
        cholesterol.Y := ys.cholesterol
      )(model(x))

    println("latent sites:")
    latent.sites.foreach(s => println(s"  ${s.name}"))

    // Start every parameter at 0, i.e. mean 0 and scale 1 for every patient,
    // rather than at a random prior draw, which can have extreme scales.
    val zero = Tensor0(0f) **! truth
    val start = latent.initWith(observations(zero)*)

    // The warmup tunes the step size and the mass matrix.
    val (warmupKey, sampleKey) = Key(1L).split2()
    val adapted = NUTS.adapt(latent, start, warmupKey, numSteps = 500)
    println(f"NUTS adapted step size: ${adapted.stepSize}%.3f")
    val (samples, infos) = adapted.sampler
      .run(adapted.position, sampleKey)
      .take(1000)
      .toList
      .unzip
    println(f"NUTS divergences: ${infos.count(_.isDivergent)} / ${samples.length}")

    // ---- posterior ----------------------------------------------------
    // Each draw as a typed record, and their mean, computed on the whole tree.
    val draws = samples.map(p => values(latent.values(p)))
    val posteriorMean = Tensor0(1f / draws.size) **! draws.reduce(_ ++ _)
    println(
      f"bloodPressure.scale.weights: ${posteriorMean.bloodPressure.scale.weights}"
    )

    println("\nposterior mean (true value)")
    def leaves(v: Values): List[(String, String)] =
      val out = List.newBuilder[(String, String)]
      v.foreachWithName([T <: Tuple] =>
        (_: Labels[T]) ?=>
          (name: String, t: Tensor[T, Float32]) => out += name -> t.toString
      )
      out.result()
    for ((name, mean), (_, trueValue)) <- leaves(posteriorMean).zip(leaves(truth)) do
      println(f"  $name%-32s $mean   ($trueValue)")
