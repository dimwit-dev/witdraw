package witdraw

import dimwit.*
import dimwit.random.Random
import dimwit.tensortree.{TensorTree, TreeOf}

/** The axis of [[Latent.Params]]: the unconstrained values of all latent
  * sites, one after the other.
  */
trait LatentDim derives Label

/** The latent sites of a model, found by tracing one forward run, and a
  * parameter type for them.
  *
  * {{{
  * val latent = Latent.discover(Constraint(Sigma, Bijector.exp)):
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
    slots: Vector[Latent.Slot[?, ?]]
):
  private val bySite: Map[SiteDef[?, ?], Latent.Slot[?, ?]] =
    slots.map(s => s.site -> s).toMap

  /** Values for the latent sites of the model: their unconstrained values, in
    * one vector.
    */
  opaque type Params = Tensor1[LatentDim, Float32]

  object Params:
    given TensorTree[Params] = TensorTree.tensor[Tuple1[LatentDim], Float32]
    given TreeOf[Params, Float32] = TreeOf.tensor[Tuple1[LatentDim], Float32]

    extension (params: Params)
      /** The value of `site`, in the model's own (constrained) space. */
      def apply[T <: Tuple, V](site: SiteDef[T, V]): Tensor[T, V] =
        slotOf(site).valueIn(params).asInstanceOf[Tensor[T, V]]

  /** The latent sites, in the order in which the model samples them. */
  def sites: Seq[SiteDef[?, ?]] = slots.map(_.site)

  /** The values of the latent sites at `params`, in the model's own space, e.g.
    * for a [[Component]] to read them as a typed record.
    */
  def values(params: Params): SiteValues =
    new SiteValues:
      def apply[T <: Tuple, V](site: SiteDef[T, V]): Tensor[T, V] =
        Params.apply(params)(site)

  /** Starting values for inference: the values of the traced run. */
  def init: Params = initWith()

  /** Starting values for inference: the given values for some sites, e.g.
    * `latent.initWith(Sigma := Tensor0(1f))`, and the values of the traced run
    * for the others.
    *
    * Every value must lie inside its constraint, e.g. be positive for
    * `Bijector.exp` or increasing for `Bijector.ordered`. A traced value can
    * lie outside when the constraint is stricter than the model, and then the
    * site needs a given value.
    */
  def initWith(values: Observation[?, ?]*): Params =
    val givenValues: Map[SiteDef[?, ?], Tensor[?, ?]] =
      values.map(o => slotOf(o.site).site -> o.value).toMap
    val start = slots.map: slot =>
      givenValues.get(slot.site).fold(slot.initial)(slot.toFlatUnchecked)
    for (slot, part) <- slots.zip(start)
    if !java.lang.Float.isFinite(part.sum.item) do
      val name = slot.site.name
      val origin = if givenValues.contains(slot.site) then "given" else "traced"
      throw IllegalArgumentException(
        s"The $origin value of '$name' lies outside its constraint ($name = ${slot.fromFlat(part)}). " +
          s"Give a starting value inside it with latent.initWith($name := ...)."
      )
    concatenate(start, Axis[LatentDim])

  /** The log-density to sample from, e.g. with NUTS: the density of the
    * unconstrained values, which is [[modelLogDensity]] plus the log-Jacobian
    * of the constraints.
    */
  def logDensity(params: Params): Tensor0[Float32] =
    val logJacobian =
      slots.map(_.logDetJacIn(params)).foldLeft(Tensor0(0f))(_ + _)
    modelLogDensity(params) + logJacobian

  /** The log-density of the model, observed on the values `params(site)`.
    * Maximise this one to find the posterior mode: unlike [[logDensity]], its
    * maximum does not depend on the constraints.
    */
  def modelLogDensity(params: Params): Tensor0[Float32] =
    ModelScorer.run(observe(slots.map(_.observationIn(params))*)(model))

  private def slotOf(site: SiteDef[?, ?]): Latent.Slot[?, ?] =
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
    * `constraints` gives a change of variables for some of the latent sites.
    */
  def discover(constraints: Constraint[?, ?]*)(model: ProbProg[Any]): Latent =
    val (_, tr) = ForwardSampler.run(Random.Key(0))(withTrace(model))
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

    val bySite = constraints.map(c => c.site -> c).toMap
    val slots = latentSites.foldLeft(Vector.empty[Slot[?, ?]]): (acc, site) =>
      val offset = acc.lastOption.fold(0)(s => s.offset + s.size)
      val c = bySite.getOrElse(site, Constraint.none(site))
      acc :+ slotFor(c, tr(c.site), offset)
    new Latent(model, slots)

  private type Flat = Tensor1[LatentDim, Float32]

  /** The part `[offset, offset + size)` of the parameter vector that holds the
    * unconstrained value of `site`, and how to convert between that part and
    * the site's value.
    *
    * @param initial
    *   the part for the site's traced value
    */
  private final class Slot[T <: Tuple, V](
      val site: SiteDef[T, V],
      val offset: Int,
      val size: Int,
      val initial: Flat,
      val toFlat: Tensor[T, V] => Flat,
      val fromFlat: Flat => Tensor[T, V],
      logDetJac: Flat => Tensor0[Float32]
  ):
    def part(params: Flat): Flat =
      params.select(Axis[LatentDim].at(offset until offset + size))

    def valueIn(params: Flat): Tensor[T, V] = fromFlat(part(params))
    def observationIn(params: Flat): Observation[T, V] =
      Observation(site, valueIn(params))
    def logDetJacIn(params: Flat): Tensor0[Float32] = logDetJac(part(params))

    /** `toFlat` for the value of an observation of `site`. */
    def toFlatUnchecked(value: Tensor[?, ?]): Flat =
      toFlat(value.asInstanceOf[Tensor[T, V]])

  /** The slot for the site of `c`, starting at `offset`, with `traced` as the
    * site's starting value.
    */
  private def slotFor[T <: Tuple, V](
      c: Constraint[T, V],
      traced: Tensor[T, V],
      offset: Int
  ): Slot[T, V] =
    given TensorTree[c.Z] = c.zTree
    val ravel = TensorTree.ravel(c.bijector.inverse(traced), Axis[LatentDim])
    val toFlat = (x: Tensor[T, V]) => ravel.flatten(c.bijector.inverse(x))
    val initial = toFlat(traced)
    Slot(
      c.site,
      offset,
      initial.shape.size,
      initial,
      toFlat,
      fromFlat = flat => c.bijector.forward(ravel.unflatten(flat)),
      logDetJac = flat => c.bijector.logDetJac(ravel.unflatten(flat))
    )
