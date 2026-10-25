package witdraw

import dimwit.*
import dimwit.random.Random
import dimwit.tensor.Labels
import dimwit.tensortree.{TensorTree, TreeOf}

/** The axis of [[Latent.Params]]: the unconstrained values of all latent sites,
  * one after the other.
  */
trait LatentDim derives Label

/** The latent sites of a model, found by tracing one forward run, and a
  * parameter type for them.
  *
  * {{{
  * val latent = Latent.discover():
  *   observe(Y := yData)(model(x))
  *
  * val nuts = NUTS[latent.Params](stepSize = 0.05)(latent.logDensity)
  * val draws = nuts.run(latent.init, key).drop(500).take(2000).map(_._1).toList
  * draws.map(_(Sigma))   // List[Sigma.Value], always positive
  * }}}
  *
  * `latent.Params` holds the unconstrained values of all latent sites in one
  * vector, which is what inference algorithms move around in; `params(site)`
  * returns the value of a site in the model's own space.
  */
final class Latent private (
    model: ProbProg[Any],
    slots: Vector[Latent.Slot[?]]
):
  import Latent.{Flat, Slot, traceOf}

  private val bySite: Map[SiteDef[?, ?], Slot[?]] =
    slots.map(s => s.site -> s).toMap

  /** Values for the latent sites of the model: their unconstrained values, in
    * one vector.
    */
  opaque type Params = Tensor1[LatentDim, Float32]

  object Params:
    given TensorTree[Params] = TensorTree.tensor[Tuple1[LatentDim], Float32]
    given TreeOf[Params, Float32] = TreeOf.tensor[Tuple1[LatentDim], Float32]

    extension (params: Params)
      /** The value of `site`, in the model's own (constrained) space.
        *
        * For a site whose support depends on other sites, e.g.
        * `X ~ Uniform(0, sigma)`, this runs the model. To read several such
        * sites, [[values]] runs it only once.
        */
      def apply[T <: Tuple, V](site: SiteDef[T, V]): Tensor[T, V] =
        values(params)(site)

  /** The latent sites, in the order in which the model samples them. */
  def sites: Seq[SiteDef[?, ?]] = slots.map(_.site)

  /** The values of the latent sites at `params`, in the model's own space, e.g.
    * for a [[Component]] to read them as a typed record.
    */
  def values(params: Params): SiteValues =
    // one run of the model, for the sites whose transform depends on others
    lazy val run = traceOf(substitute(params)(model))
    new SiteValues:
      def apply[T <: Tuple, V](site: SiteDef[T, V]): Tensor[T, V] =
        slotOf(site)
          .fixedValueIn(params)
          .getOrElse(run(site))
          .asInstanceOf[Tensor[T, V]]

  /** Starting values for inference: the values of the traced run. */
  def init: Params = initWith()

  /** Starting values for inference: the given values for some sites, e.g.
    * `latent.initWith(Sigma := Tensor0(1f))`, and the values of the traced run
    * for the others.
    *
    * Every value must lie inside its support or constraint, e.g. be positive
    * for a `HalfNormal` site or increasing for `Bijector.ordered`. A traced
    * value can lie outside when a constraint is stricter than the model, or
    * when given values move the support of another site, and then that site
    * needs a given value too.
    */
  def initWith(values: Observation[?, ?]*): Params =
    val givenValues: Map[SiteDef[?, ?], Tensor[?, ?]] =
      values.map(o => slotOf(o.site).site -> o.value).toMap
    def startValue(slot: Slot[?]): Tensor[?, ?] =
      givenValues.getOrElse(slot.site, slot.traced)
    // the supports at the starting values, for the sites whose transform
    // depends on others
    lazy val atStart =
      val start = slots.map(slot => slot.observation(startValue(slot)))
      traceOf(observe(start*)(model))

    val parts = slots.map: slot =>
      if givenValues.isEmpty then slot.initial
      else
        val t = slot.fixed.getOrElse(
          slot.transformFor(atStart.entry(slot.site).get.support)
        )
        slot.inverse(t, startValue(slot))
    for
      (slot, part) <- slots.zip(parts)
      if !java.lang.Float.isFinite(part.sum.item)
    do
      val name = slot.site.name
      val origin = if givenValues.contains(slot.site) then "given" else "traced"
      throw IllegalArgumentException(
        s"The $origin value of '$name' lies outside its support or constraint ($name = ${startValue(slot)}). " +
          s"Give a starting value inside it with latent.initWith($name := ...)."
      )
    concatenate(parts, Axis[LatentDim])

  /** The log-density to sample from, e.g. with NUTS: the density of the
    * unconstrained values, which is [[modelLogDensity]] plus the log-Jacobian
    * of the transforms.
    */
  def logDensity(params: Params): Tensor0[Float32] =
    val (modelLogDensity, logJacobian) = score(params)
    modelLogDensity + logJacobian

  /** The log-density of the model, observed on the values `params(site)`.
    * Maximise this one to find the posterior mode: unlike [[logDensity]], its
    * maximum does not depend on the transforms.
    */
  def modelLogDensity(params: Params): Tensor0[Float32] = score(params)._1

  /** The log-density of the model at `params`, and the log-Jacobian of the
    * transforms, from one run of the model.
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
    * transform for the support of that statement, so a support may depend on
    * the values of the sites before it, e.g. `X ~ Uniform(0, sigma)`.
    * `addLogJacobian` receives the log-Jacobian of every transform.
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
              val t = slot.transformFor(statement.support)
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
    * Each latent site is sampled through the bijector that matches the
    * [[Support]] of its distribution, e.g. on the log scale for `HalfNormal` or
    * by stick-breaking for `Dirichlet`. The bijector is built anew in every run
    * of the model, so a support may depend on other sites, e.g.
    * `X ~ Uniform(0, sigma)`. `constraints` overrides the bijector for some
    * sites, e.g. `Constraint(Mu, Bijector.ordered[Comp])`. A latent site whose
    * support is unknown needs a constraint.
    */
  def discover(constraints: Constraint[?, ?]*)(model: ProbProg[Any]): Latent =
    val tr = traceOf(model)
    // A second run, with other values, only to make reading values fast. A
    // site whose support is the same in both runs, e.g. HalfNormal(0, 1), does
    // not depend on other sites, so `params(site)` computes its value directly.
    // A site whose support changes, e.g. Uniform(0, sigma), needs a run of the
    // model to read its value, as only that run knows the current sigma.
    // Sampling does not use this: `substitute` builds every transform from the
    // support of the current run. Without the second run, `params(site)` would
    // have to run the model for every site, e.g. 5000 runs to summarise 5
    // sites over 1000 draws.
    lazy val otherRun = traceOf(model, Random.Key(1))
    val latentSites = tr.latentSites
    def names = latentSites.map(_.name).mkString(", ")

    for (site, cs) <- constraints.groupBy(_.site) if cs.size > 1 do
      throw IllegalArgumentException(
        s"Site '${site.name}' has more than one constraint."
      )
    for c <- constraints if !latentSites.contains(c.site) do
      throw IllegalArgumentException(
        s"Site '${c.site.name}' has a constraint, but is not a latent site of the model. Latent sites: $names."
      )
    for site <- latentSites if tr(site).dtype != DType.Float32 do
      throw IllegalArgumentException(
        s"Site '${site.name}' is ${tr(site).dtype}, but latent sites must be continuous (Float32). " +
          "Observe it, or marginalise it out of the model."
      )

    // latent sites are Float32, as checked above
    val bySite = constraints
      .map(c => c.site -> c.transform.asInstanceOf[Transform[Tuple, Float32]])
      .toMap
    val slots = latentSites.foldLeft(Vector.empty[Slot[?]]): (acc, site) =>
      val offset = acc.lastOption.fold(0)(s => s.offset + s.size)
      val entry = tr.entry(site).get.asInstanceOf[TraceEntry[Tuple, Float32]]
      acc :+ slotFor(entry, bySite.get(site), otherRun.entry(site), offset)
    new Latent(model, slots)

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
      constraint: Option[Transform[T, Float32]],
      other: => Option[TraceEntry[?, ?]],
      offset: Int
  ): Slot[T] =
    val t = constraint.getOrElse(defaultTransform(entry))
    // `fixed`: the transform when it is the same for all values of the other
    // sites, so that the site's value can be read without running the model.
    // That holds for a constraint, and for a support that is the same in
    // another run with other values (see `discover`). A custom support cannot
    // be compared, as its bijector is a new object in every run, so it is read
    // through the model. `fixed` only speeds up reading: if it were wrong,
    // `params(site)` would show a wrong value, but sampling would still be
    // right, as it never uses `fixed`.
    def supportIsFixed = entry.support match
      case _: Support.Custom[?, ?, ?, ?] => false
      case support                       => other.exists(_.support == support)
    val fixed = constraint.orElse(Option.when(supportIsFixed)(t))

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
      constraint,
      fixed,
      ravel.flatten.asInstanceOf[Any => Flat],
      ravel.unflatten
    )

  /** The transform for the support of a latent site, traced as `entry`. */
  private def defaultTransform[T <: Tuple](
      entry: TraceEntry[T, Float32]
  ): Transform[T, Float32] =
    val name = entry.site.name
    Support
      .transform(entry.support)(using entry.site.labels)
      .getOrElse:
        val distribution = entry.distribution.getClass.getSimpleName
        entry.support match
          case Support.Unknown() =>
            throw IllegalArgumentException(
              s"Site '$name' is latent, but the support of its distribution " +
                s"($distribution) is unknown, e.g. because it was typed as a plain Distribution. " +
                s"Give it a Constraint, e.g. Latent.discover(Constraint($name, Bijector.exp)) for positive values, " +
                "keep the distribution's own type at `~`, or give its class a `given SupportOf`."
            )
          case support =>
            throw IllegalArgumentException(
              s"Site '$name' is latent, and witdraw has no bijector for the support $support of its distribution " +
                s"($distribution). Give it a Constraint, or give the distribution a Support.Custom with its own bijector."
            )

  /** The part `[offset, offset + size)` of the parameter vector that holds the
    * unconstrained value of `site`, and how to convert between that part and
    * the site's value.
    *
    * @param traced
    *   the site's value in the traced run
    * @param initial
    *   the part for the traced value
    * @param constraint
    *   the transform of an explicit [[Constraint]], used in every run
    * @param fixed
    *   the transform, if it is the same for all values of the other sites: the
    *   constraint, or the transform of a support that does not depend on them.
    *   Then the site's value can be read without running the model. Only
    *   reading uses it, sampling never does.
    * @param flatten
    *   the unconstrained value of a transform to its part
    * @param unflatten
    *   a part to the unconstrained value of a transform
    */
  private final class Slot[T <: Tuple](
      val site: SiteDef[T, Float32],
      val offset: Int,
      val size: Int,
      val traced: Tensor[T, Float32],
      val initial: Flat,
      constraint: Option[Transform[T, Float32]],
      val fixed: Option[Transform[T, Float32]],
      flatten: Any => Flat,
      unflatten: Flat => Any
  ):
    def part(params: Flat): Flat =
      params.select(Axis[LatentDim].at(offset until offset + size))

    /** The transform for a statement of this site with the given support: the
      * constraint, or the transform of the support.
      */
    def transformFor(support: Support[?, ?]): Transform[T, Float32] =
      constraint.getOrElse:
        val s = support.asInstanceOf[Support[T, Float32]]
        Support
          .transform(s)(using site.labels)
          .getOrElse(
            throw IllegalStateException(
              s"No bijector for the support $s of '${site.name}'."
            )
          )

    /** The value for the unconstrained part `z`, and `log |det J|` at `z`. */
    def forward(
        t: Transform[T, Float32],
        z: Flat
    ): (Tensor[T, Float32], Tensor0[Float32]) =
      val zt = unflatten(z).asInstanceOf[t.Z]
      (t.bijector.forward(zt), t.bijector.logDetJac(zt))

    /** The unconstrained part for the value `x`. */
    def inverse(t: Transform[T, Float32], x: Tensor[?, ?]): Flat =
      flatten(t.bijector.inverse(x.asInstanceOf[Tensor[T, Float32]]))

    /** The site's value at `params`, if its transform is fixed. */
    def fixedValueIn(params: Flat): Option[Tensor[T, Float32]] =
      fixed.map(t => forward(t, part(params))._1)

    def observation(x: Tensor[?, ?]): Observation[T, Float32] =
      Observation(site, x.asInstanceOf[Tensor[T, Float32]])
