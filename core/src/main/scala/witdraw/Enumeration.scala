package witdraw

import dimwit.*

/** The axis along the values of a site summed out with [[marginalize]], e.g.
  * of [[Latent.valuesOf]] and [[Latent.posteriorOf]].
  */
trait Choice derives Label

/** The axis along all combinations of values of the sites summed out with
  * [[marginalize]]. Users never see it: [[Latent.posteriorOf]] gives the
  * posterior of one site along [[Choice]].
  */
private[witdraw] trait Combination derives Label

/** All combinations of values of the sites summed out with [[marginalize]],
  * along [[Combination]], the first site varying slowest: the discrete
  * counterpart of [[ParamLayout]]. A combination is an index `k`, and the
  * model is conditioned on it by observing the sites on their values in it.
  *
  * A model with no sites summed out has the trivial enumeration: one
  * combination, of no sites, which leaves the model as it is.
  */
private[witdraw] final class Enumeration private (
    val sites: Vector[Enumeration.Enumerated[?]],
    val size: Int
):
  import Enumeration.Enumerated

  private val bySite: Map[SiteDef[?, ?], Enumerated[?]] =
    sites.map(e => e.site -> e).toMap

  /** Whether no site is summed out. */
  def isTrivial: Boolean = sites.isEmpty

  val combinations: Tensor1[Combination, Int32] =
    Tensor1(Axis[Combination]).fromArray(Array.range(0, size))

  /** `model` conditioned on the combination `k`, which may be traced. */
  def condition[A](k: Tensor0[Int32])(model: ProbProg[A]): ProbProg[A] =
    if isTrivial then model
    else observe(sites.map(_.observationAt(k))*)(model)

  def enumeratedOf(site: SiteDef[?, ?]): Enumerated[?] =
    bySite.getOrElse(
      site,
      throw java.util.NoSuchElementException(
        s"Site '${site.name}' is not summed out with marginalize. Sites summed out: ${sites.map(_.site.name).mkString(", ")}"
      )
    )

private[witdraw] object Enumeration:

  /** The enumeration of a model with no sites summed out. */
  val trivial: Enumeration = Enumeration(Vector.empty, 1)

  /** The most combinations of values of sites summed out. */
  private val maxCombinations = 1_000_000

  /** The enumeration of the sites summed out, traced as `entries`, and as in
    * `other` in another run, which shows whether their values depend on the
    * other sites.
    */
  def of(entries: Seq[TraceEntry[?, ?]], other: => Trace): Enumeration =
    if entries.isEmpty then trivial
    else
      val values = entries.map: e =>
        val (description, tensor) = valuesOf(e)
        val otherDescription = other.entry(e.site).map(valuesOf(_)._1)
        if !otherDescription.contains(description) then
          throw IllegalArgumentException(
            s"Site '${e.site.name}' is marginalized, but its values depend on other sites " +
              s"($description in one run, ${otherDescription.getOrElse("not sampled")} in another). " +
              "The values of a site summed out must be fixed."
          )
        tensor
      val sizes = values.map(_.shape(Axis[Choice]))
      val total = sizes.foldLeft(1L)(_ * _)
      if total > maxCombinations then
        throw IllegalArgumentException(
          s"The sites summed out (${entries.map(_.site.name).mkString(", ")}) have $total combinations of values, " +
            s"more than $maxCombinations: every log-density evaluation would run the model for each of them."
        )
      val n = total.toInt
      val strides = sizes.scanRight(1)(_ * _).tail
      val enumerated = entries.zip(values).zip(sizes.zip(strides)).map:
        case ((e, v), (size, stride)) =>
          val index = Array.tabulate(n)(k => (k / stride) % size)
          Enumerated(
            e.site,
            v,
            v.select(Axis[Choice].at(Tensor1(Axis[Combination]).fromArray(index))),
            Tensor2(Axis[Combination], Axis[Choice]).fromArray(
              Array.tabulate(n, size)((k, j) => if index(k) == j then 1f else 0f)
            )
          )
      Enumeration(enumerated.toVector, n)

  /** The values of a site to sum out, traced as `e`: a description to compare
    * them between runs, and the values along [[Choice]].
    */
  private def valuesOf(e: TraceEntry[?, ?]): (Any, Tensor1[Choice, ?]) =
    val name = e.site.name
    if e.value.shape.dimensions.nonEmpty then
      throw IllegalArgumentException(
        s"Site '$name' is marginalized, but it is not scalar (shape ${e.value.shape}). Only scalar sites can be summed out."
      )
    e.support match
      case Support.Integers(lo, hi) if e.value.dtype == DType.Int32 =>
        val (l, h) = (
          lo.asInstanceOf[Tensor0[Int32]].item,
          hi.asInstanceOf[Tensor0[Int32]].item
        )
        if l >= h then
          throw IllegalArgumentException(
            s"Site '$name' is marginalized, but it has no values: the integers from $l until $h."
          )
        (s"the integers from $l until $h", Tensor1(Axis[Choice]).fromArray(Array.range(l, h)))
      case Support.Booleans() =>
        ("false and true", Tensor1(Axis[Choice]).fromArray(Array(false, true)))
      case support =>
        val distribution = e.distribution.getClass.getSimpleName
        throw IllegalArgumentException(
          s"Site '$name' is marginalized, but the values of its distribution ($distribution) cannot be listed " +
            s"(support $support). Sites with a DiscreteUniform, Categorical or Bernoulli distribution can be summed out."
        )

  /** A site summed out.
    *
    * @param values
    *   its values
    * @param valuesByCombination
    *   its value in every combination
    * @param oneHot
    *   which of its values each combination has
    */
  final class Enumerated[V](
      val site: SiteDef[?, ?],
      val values: Tensor1[Choice, V],
      valuesByCombination: Tensor1[Combination, V],
      val oneHot: Tensor2[Combination, Choice, Float32]
  ):
    /** Its value in the combination `k`, which may be traced. */
    def observationAt(k: Tensor0[Int32]): Observation[EmptyTuple, V] =
      Observation(
        site.asInstanceOf[SiteDef[EmptyTuple, V]],
        valuesByCombination.select(Axis[Combination].at(k))
      )
