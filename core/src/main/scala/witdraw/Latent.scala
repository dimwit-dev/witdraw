package witdraw

import dimwit.*
import dimwit.random.Random
import dimwit.tensor.Labels
import dimwit.tensortree.{TensorTree, TreeOf}

/** The axis of [[Latent.Params]]: the unconstrained values of all latent sites,
  * one after the other.
  */
trait LatentDim derives Label

/** The latent sites of a model, found by tracing one forward run.
  *
  * Inference moves in `latent.Params`: the unconstrained values of all latent
  * sites, in one vector, e.g. a starting value or a draw of NUTS. What you
  * read is what the model returns: `latent.result(params)` at given
  * parameters, and `chain.samples` at every draw of a chain, the same type as
  * a forward run of the model.
  *
  * {{{
  * def model(using ProbContext) =
  *   val sigma = Sigma ~ HalfNormal(0f, 1f)
  *   val y = Y ~ Normal.isotropic(mu, sigma)
  *   (sigma = sigma, y = y)
  *
  * val latent = Latent.discover(observe(Y := yData)(model))
  * val chain = NUTS.adapt(latent, warmupKey, latent.init).sample(key, 1000)
  * chain.samples.map(_.sigma)     // always positive
  * latent.result(latent.init).sigma
  * }}}
  */
sealed abstract class Latent private (
    model: ProbProg[Any],
    slots: Vector[Latent.Slot[?]],
    resultTree: TensorTree[?]
):
  import Latent.{Flat, Slot, traceOf}

  private val bySite: Map[SiteDef[?, ?], Slot[?]] =
    slots.map(s => s.site -> s).toMap

  /** What inference algorithms keep for this latent, e.g. compiled functions,
    * so that later calls, e.g. for further chains, reuse them.
    */
  private val caches = scala.collection.concurrent.TrieMap.empty[String, AnyRef]

  /** The value kept under `key`, made by `make` on first use. */
  private[witdraw] def cached(key: String)(make: => AnyRef): AnyRef =
    caches.getOrElseUpdate(key, make)

  /** The type of what the model returns, e.g. a named tuple of tensors. It is
    * inferred from the model, so it only needs a name where it is written
    * down, e.g. `(r: latent.Result) => r.sigma`.
    *
    * A function that returns a latent keeps it only if its result type is
    * inferred: declared as plain `Latent`, the result's fields are lost.
    */
  type Result

  /** Values for the latent sites of the model: their unconstrained values, in
    * one vector. This is where inference algorithms move, e.g. a starting value
    * or a draw, not something to read: [[result]] gives the model's result at
    * them.
    */
  opaque type Params = Tensor1[LatentDim, Float32]

  object Params:
    given TensorTree[Params] = TensorTree.tensor[Tuple1[LatentDim], Float32]
    given TreeOf[Params, Float32] = TreeOf.tensor[Tuple1[LatentDim], Float32]
    given ModelResult[Params, Result] = result(_)

  /** The latent sites, in the order in which the model samples them. */
  def sites: Seq[SiteDef[?, ?]] = slots.map(_.site)

  /** The values of the latent sites at `params`, in the model's own space, by
    * site, e.g. for tests of the supports. Users read [[result]] instead.
    */
  private[witdraw] def values(params: Params): SiteValues =
    // one run of the model, for the sites whose support depends on others
    lazy val run = traceOf(substitute(params)(model))
    new SiteValues:
      def apply[T <: Tuple, V](site: SiteDef[T, V]): Tensor[T, V] =
        slotOf(site)
          .fixedValueIn(params)
          .getOrElse(run(site))
          .asInstanceOf[Tensor[T, V]]

  /** The model's result at `params`: the model run with the latent sites set to
    * their values at `params`, and the observed sites to their data. The same
    * type as a forward run of the model, e.g. a prior sample, so prior and
    * posterior samples are read the same way. Nothing is drawn.
    */
  def result(params: Params): Result =
    if compileFailed then eagerResult(params)
    else
      try compiledResult(params)
      catch
        case scala.util.control.NonFatal(_) =>
          compileFailed = true
          System.err.println(
            "witdraw: the model's result cannot be compiled, e.g. as the model calls `.item` on a sampled value, so it runs eagerly, which is slower."
          )
          eagerResult(params)

  private def eagerResult(params: Flat): Result =
    ForwardSampler
      .run(Random.Key(0))(substitute(params)(model))
      .asInstanceOf[Result]

  /** [[result]], compiled once. Compiling fails for a model that needs concrete
    * values, e.g. calls `.item` on a sampled value; then the model runs
    * eagerly.
    */
  private lazy val compiledResult: Params => Result =
    jit(eagerResult)(using
      summon[TensorTree[Flat]],
      resultTree.asInstanceOf[TensorTree[Result]]
    )

  @volatile private var compileFailed = false

  /** Starting values for inference: the values of the traced run. */
  def init: Params = initWith()

  /** Starting values for inference: the given values for some sites, e.g.
    * `latent.initWith(Sigma := Tensor0(1f))`, and the values of the traced run
    * for the others.
    *
    * Every value must lie inside its support, e.g. be positive for a
    * `HalfNormal` site or increasing for an `Ordered` one. A traced value can
    * lie outside when given values move the support of another site, e.g.
    * `X ~ Uniform(0, sigma)` with a smaller given sigma, and then that site
    * needs a given value too.
    */
  def initWith(values: Observation[?, ?]*): Params =
    val givenValues: Map[SiteDef[?, ?], Tensor[?, ?]] =
      values.map(o => slotOf(o.site).site -> o.value).toMap
    def startValue(slot: Slot[?]): Tensor[?, ?] =
      givenValues.getOrElse(slot.site, slot.traced)
    // the supports at the starting values, for the sites whose support
    // depends on others
    lazy val atStart =
      val start = slots.map(slot => slot.observation(startValue(slot)))
      traceOf(observe(start*)(model))

    val parts = slots.map: slot =>
      if givenValues.isEmpty then slot.initial
      else
        val t = slot.fixed.getOrElse(
          slot.bijectorFor(atStart.entry(slot.site).get.support)
        )
        slot.inverse(t, startValue(slot))
    for
      (slot, part) <- slots.zip(parts)
      if !java.lang.Float.isFinite(part.sum.item)
    do
      val name = slot.site.name
      val origin = if givenValues.contains(slot.site) then "given" else "traced"
      throw IllegalArgumentException(
        s"The $origin value of '$name' lies outside its support ($name = ${startValue(slot)}). " +
          s"Give a starting value inside it with latent.initWith($name := ...)."
      )
    concatenate(parts, Axis[LatentDim])

  /** Random starting values for inference, as in Stan: every unconstrained
    * value is drawn uniformly between `-radius` and `radius`. Since they are
    * unconstrained, the values always lie inside the supports.
    *
    * Use it to start several chains from different positions, e.g.
    * `key.split(4).map(latent.initRandom(_))`.
    */
  def initRandom(key: Random.Key, radius: Float = 2f): Params =
    // the size from the slots, without running the model
    val shape = Shape(Axis[LatentDim] -> slots.map(_.size).sum)
    dimwit.stats
      .Uniform(Tensor(shape).fill(-radius), Tensor(shape).fill(radius))
      .sample(key)

  /** The log-density to sample from, e.g. with NUTS: the density of the
    * unconstrained values, which is [[modelLogDensity]] plus the log-Jacobian
    * of the bijectors.
    */
  def logDensity(params: Params): Tensor0[Float32] =
    val (modelLogDensity, logJacobian) = score(params)
    modelLogDensity + logJacobian

  /** The log-density of the model, observed on the sites' values at `params`.
    * Maximise this one to find the posterior mode: unlike [[logDensity]], its
    * maximum does not depend on the bijectors.
    */
  def modelLogDensity(params: Params): Tensor0[Float32] = score(params)._1

  /** The log-density of the model at `params`, and the log-Jacobian of the
    * bijectors, from one run of the model.
    */
  private def score(params: Flat): (Tensor0[Float32], Tensor0[Float32]) =
    var logJacobian = Tensor0(0f)
    val modelLogDensity = ModelScorer.run:
      substitute(params, logDetJac => logJacobian = logJacobian + logDetJac)(
        model
      )
    (modelLogDensity, logJacobian)

  /** Runs `model` with the latent sites observed on their values at `params`.
    *
    * The value of a site is computed when the model samples it, through the
    * bijector for the support of that statement, so a support may depend on the
    * values of the sites before it, e.g. `X ~ Uniform(0, sigma)`.
    * `addLogJacobian` receives the log-Jacobian of every bijector.
    */
  private def substitute[A](
      params: Flat,
      addLogJacobian: Tensor0[Float32] => Unit = _ => ()
  )(model: ProbProg[A])(using outer: ProbContext): A =
    model(using
      new ProbContext:
        def sample[T <: Tuple, V](
            statement: SampleStatement[T, V]
        ): Tensor[T, V] =
          bySite.get(statement.site) match
            case None       => outer.sample(statement)
            case Some(slot) =>
              val t = slot.bijectorFor(statement.support)
              val (value, logDetJac) = slot.forward(t, slot.part(params))
              addLogJacobian(logDetJac)
              outer.sample(
                statement.copy(observed =
                  Some(value.asInstanceOf[Tensor[T, V]])
                )
              )
    )

  private def slotOf(site: SiteDef[?, ?]): Slot[?] =
    bySite.getOrElse(
      site,
      throw java.util.NoSuchElementException(
        s"Site '${site.name}' is not a latent site of the model. Latent sites: ${sites.map(_.name).mkString(", ")}"
      )
    )

object Latent:

  /** Runs `model` forward once, and collects its latent sites: the sites that
    * are not observed. The values of that run are the default starting point,
    * [[Latent.init]].
    *
    * The run uses a fixed key: which sites are latent cannot depend on the
    * values drawn, so the key only decides the default starting point. Use
    * [[Latent.initWith]] to start elsewhere.
    *
    * The parameters are meant for gradient-based inference, so every latent
    * site must be continuous (`Float32`); a discrete latent site is an error.
    *
    * Each latent site is sampled through the [[Bijector]] of the [[Support]]
    * of its distribution, e.g. on the log scale for `HalfNormal`, by
    * stick-breaking for `Dirichlet`, or as increasing values for `Ordered`.
    * The bijector is built anew in every run of the model, so a support may
    * depend on other sites, e.g. `X ~ Uniform(0, sigma)`. Increasing values
    * come from the distribution `Ordered`, e.g.
    * `Mu ~ Ordered(Normal(0f, 10f), Axis[Comp] -> k)`. A latent site whose
    * support is unknown is an error.
    *
    * The model must return a tensor tree, e.g. a named tuple of tensors: its
    * result is a sample of the posterior, see [[Latent.result]]. The result is
    * compiled, so reading the samples of a chain is fast.
    */
  def discover[A](model: ProbProg[A])(using resultTree: TensorTree[A]): Latent {
    type Result = A
  } =
    val tr = traceOf(model)
    // A second run, with other values, only to make reading values fast. A
    // site whose support is the same in both runs, e.g. HalfNormal(0, 1), does
    // not depend on other sites, so `values(params)` computes its value directly.
    // A site whose support changes, e.g. Uniform(0, sigma), needs a run of the
    // model to read its value, as only that run knows the current sigma.
    // Sampling does not use this: `substitute` builds every bijector from the
    // support of the current run. Without the second run, `values(params)` would
    // have to run the model for every site, e.g. 5000 runs to summarise 5
    // sites over 1000 draws.
    lazy val otherRun = traceOf(model, Random.Key(1))
    val latentSites = tr.latentSites

    for site <- latentSites if tr(site).dtype != DType.Float32 do
      throw IllegalArgumentException(
        s"Site '${site.name}' is ${tr(site).dtype}, but latent sites must be continuous (Float32). " +
          "Observe it, or marginalise it out of the model."
      )

    // latent sites are Float32, as checked above
    val slots = latentSites.foldLeft(Vector.empty[Slot[?]]): (acc, site) =>
      val offset = acc.lastOption.fold(0)(s => s.offset + s.size)
      val entry = tr.entry(site).get.asInstanceOf[TraceEntry[Tuple, Float32]]
      acc :+ slotFor(entry, otherRun.entry(site), offset)
    new Latent(model, slots, resultTree):
      type Result = A

  private type Flat = Tensor1[LatentDim, Float32]

  /** The trace of a forward run of `model`. When every latent site is filled
    * in, as by `observe` or `substitute`, nothing is drawn, and the key does
    * not matter.
    */
  private def traceOf(model: ProbProg[Any], key: Random.Key = Random.Key(0)) =
    ForwardSampler.run(key)(withTrace(model))._2

  /** The slot of a latent site, traced as `entry`, and as `other` in another
    * run of the model, starting at `offset`.
    */
  private def slotFor[T <: Tuple](
      entry: TraceEntry[T, Float32],
      other: => Option[TraceEntry[?, ?]],
      offset: Int
  ): Slot[T] =
    val t = bijectorOf(entry)
    // `fixed`: the bijector when it is the same for all values of the other
    // sites, so that the site's value can be read without running the model.
    // That holds for a support that is the same in another run with other
    // values (see `discover`). A custom support cannot be compared, as its
    // bijector is a new object in every run, so it is read through the model.
    // `fixed` only speeds up reading: if it were wrong, `values(params)` would
    // show a wrong value, but sampling would still be right, as it never uses
    // `fixed`.
    def supportIsFixed = entry.support match
      case _: Support.Custom[?, ?, ?, ?] => false
      case support                       => other.exists(_.support == support)
    val fixed = Option.when(supportIsFixed)(t)

    given TensorTree[t.Z] = t.zTree
    val z = t.bijector.inverse(entry.value)
    val ravel = TensorTree.ravel(z, Axis[LatentDim])
    val initial = ravel.flatten(z)
    Slot(
      entry.site,
      offset,
      initial.shape.size,
      entry.value,
      initial,
      fixed,
      ravel.flatten.asInstanceOf[Any => Flat],
      ravel.unflatten
    )

  /** The bijector for the support of a latent site, traced as `entry`. */
  private def bijectorOf[T <: Tuple](
      entry: TraceEntry[T, Float32]
  ): BijectorTo[T, Float32] =
    val name = entry.site.name
    Support
      .bijectorFor(entry.support)(using entry.site.labels)
      .getOrElse:
        val distribution = entry.distribution.getClass.getSimpleName
        entry.support match
          case Support.Unknown() =>
            throw IllegalArgumentException(
              s"Site '$name' is latent, but the support of its distribution " +
                s"($distribution) is unknown. The support is looked up for the static type of the " +
                "distribution where `~` is written, so it is unknown when that type is a plain " +
                "Distribution, or a type parameter of a helper. Keep the distribution's own type at `~`, " +
                "pass the support into a helper with `(using SupportOf[D, T, V])`, or give the " +
                "distribution's class a `given SupportOf`."
            )
          case support =>
            throw IllegalArgumentException(
              s"Site '$name' is latent, and witdraw has no bijector for the support $support of its distribution " +
                s"($distribution). Give the distribution a Support.Custom with its own bijector."
            )

  /** The part `[offset, offset + size)` of the parameter vector that holds the
    * unconstrained value of `site`, and how to convert between that part and
    * the site's value.
    *
    * @param traced
    *   the site's value in the traced run
    * @param initial
    *   the part for the traced value
    * @param fixed
    *   the bijector, if its support does not depend on the values of the other
    *   sites. Then the site's value can be read without running the model.
    *   Only reading uses it, sampling never does.
    * @param flatten
    *   the unconstrained value of a bijector to its part
    * @param unflatten
    *   a part to the unconstrained value of a bijector
    */
  private final class Slot[T <: Tuple](
      val site: SiteDef[T, Float32],
      val offset: Int,
      val size: Int,
      val traced: Tensor[T, Float32],
      val initial: Flat,
      val fixed: Option[BijectorTo[T, Float32]],
      flatten: Any => Flat,
      unflatten: Flat => Any
  ):
    def part(params: Flat): Flat =
      params.select(Axis[LatentDim].at(offset until offset + size))

    /** The bijector for a statement of this site with the given support. */
    def bijectorFor(support: Support[?, ?]): BijectorTo[T, Float32] =
      val s = support.asInstanceOf[Support[T, Float32]]
      Support
        .bijectorFor(s)(using site.labels)
        .getOrElse(
          throw IllegalStateException(
            s"No bijector for the support $s of '${site.name}'."
          )
        )

    /** The value for the unconstrained part `z`, and `log |det J|` at `z`. */
    def forward(
        t: BijectorTo[T, Float32],
        z: Flat
    ): (Tensor[T, Float32], Tensor0[Float32]) =
      val zt = unflatten(z).asInstanceOf[t.Z]
      (t.bijector.forward(zt), t.bijector.logDetJac(zt))

    /** The unconstrained part for the value `x`. */
    def inverse(t: BijectorTo[T, Float32], x: Tensor[?, ?]): Flat =
      flatten(t.bijector.inverse(x.asInstanceOf[Tensor[T, Float32]]))

    /** The site's value at `params`, if its bijector is fixed. */
    def fixedValueIn(params: Flat): Option[Tensor[T, Float32]] =
      fixed.map(t => forward(t, part(params))._1)

    def observation(x: Tensor[?, ?]): Observation[T, Float32] =
      Observation(site, x.asInstanceOf[Tensor[T, Float32]])
