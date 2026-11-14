package witdraw

import dimwit.*
import dimwit.random.Random
import dimwit.tensortree.TensorTree

/** Values for the latent sites of a model, unconstrained, in one vector. */
private[witdraw] type Flat = Tensor1[LatentDim, Float32]

/** Where the latent sites of a model lie in the parameter vector, and how to
  * convert between their parts of it and their values. It never runs the
  * model: the bijector of a site comes from the support of the run at hand.
  */
private[witdraw] final class ParamLayout(val slots: Vector[ParamLayout.Slot[?]]):
  import ParamLayout.Slot

  private val bySite: Map[SiteDef[?, ?], Slot[?]] =
    slots.map(s => s.site -> s).toMap

  /** The length of the parameter vector. */
  val size: Int = slots.map(_.size).sum

  /** The latent sites, in the order in which the model samples them. */
  def sites: Seq[SiteDef[?, ?]] = slots.map(_.site)

  def slotOf(site: SiteDef[?, ?]): Slot[?] =
    bySite.getOrElse(
      site,
      throw java.util.NoSuchElementException(
        s"Site '${site.name}' is not a latent site of the model. Latent sites: ${sites.map(_.name).mkString(", ")}"
      )
    )

  /** Runs `model` with the latent sites set to their values at `params`.
    * Returns the model's result and the sum of the log-Jacobians of the
    * bijectors.
    *
    * The value of a site is computed when the model samples it, through the
    * bijector for the support of that statement, so a support may depend on
    * the values of the sites before it, e.g. `X ~ Uniform(0, sigma)`. Other
    * sites, e.g. the observed ones or the ones summed out, are passed on as
    * they are.
    */
  def substitute[A](params: Flat)(model: ProbProg[A])(using
      outer: ProbContext
  ): (A, Tensor0[Float32]) =
    var logJacobian = Tensor0(0f)
    val result = model(using
      new ProbContext:
        def sample[T <: Tuple, V](
            statement: SampleStatement[T, V]
        ): Tensor[T, V] =
          bySite.get(statement.site) match
            case None       => outer.sample(statement)
            case Some(slot) =>
              val t = slot.bijectorFor(statement.support)
              val (value, logDetJac) = slot.forward(t, slot.part(params))
              logJacobian = logJacobian + logDetJac
              outer.sample(
                statement.copy(value = Some(value.asInstanceOf[Tensor[T, V]]))
              )
    )
    (result, logJacobian)

  /** The parameters for the values of the traced run. */
  def initial: Flat = checked(slots.map(_.initial), _.traced, _ => "traced")

  /** Every unconstrained value drawn uniformly between `-radius` and `radius`. */
  def random(key: Random.Key, radius: Float): Flat =
    val shape = Shape(Axis[LatentDim] -> size)
    dimwit.stats
      .Uniform(Tensor(shape).fill(-radius), Tensor(shape).fill(radius))
      .sample(key)

  /** The parameters for the values `valueOf(slot)`, with the bijectors for the
    * supports `supportOf(slot)` of a run at those values. `origin` says where
    * a value comes from, for the error when it lies outside its support.
    */
  def pack(
      valueOf: Slot[?] => Tensor[?, ?],
      supportOf: Slot[?] => Support[?, ?],
      origin: Slot[?] => String
  ): Flat =
    val parts = slots.map(s => s.inverse(s.bijectorFor(supportOf(s)), valueOf(s)))
    checked(parts, valueOf, origin)

  private def checked(
      parts: Vector[Flat],
      valueOf: Slot[?] => Tensor[?, ?],
      origin: Slot[?] => String
  ): Flat =
    for
      (slot, part) <- slots.zip(parts)
      if !java.lang.Float.isFinite(part.sum.item)
    do
      val name = slot.site.name
      throw IllegalArgumentException(
        s"The ${origin(slot)} value of '$name' lies outside its support ($name = ${valueOf(slot)}). " +
          s"Give a starting value inside it with latent.initWith($name := ...)."
      )
    concatenate(parts, Axis[LatentDim])

private[witdraw] object ParamLayout:

  /** The layout of the latent sites traced as `entries`, in this order. */
  def of(entries: Seq[TraceEntry[Tuple, Float32]]): ParamLayout =
    ParamLayout(entries.foldLeft(Vector.empty[Slot[?]]): (acc, entry) =>
      val offset = acc.lastOption.fold(0)(s => s.offset + s.size)
      acc :+ slotFor(entry, offset))

  /** The slot of a latent site, traced as `entry`, starting at `offset`. */
  private def slotFor[T <: Tuple](
      entry: TraceEntry[T, Float32],
      offset: Int
  ): Slot[T] =
    val t = bijectorOf(entry)
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
    * @param flatten
    *   the unconstrained value of a bijector to its part
    * @param unflatten
    *   a part to the unconstrained value of a bijector
    */
  final class Slot[T <: Tuple](
      val site: SiteDef[T, Float32],
      val offset: Int,
      val size: Int,
      val traced: Tensor[T, Float32],
      val initial: Flat,
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

    def observation(x: Tensor[?, ?]): Observation[T, Float32] =
      Observation(site, x.asInstanceOf[Tensor[T, Float32]])
