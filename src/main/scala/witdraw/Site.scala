package witdraw

import dimwit.*
import dimwit.tensor.{Labels, Tensor}


// Match types: reduce to concrete shape/element at compile time for a known Tensor type
private type ShapeOf[Ten <: Tensor[?, ?]] <: Tuple = Ten match
  case Tensor[t, v] => t

private type ElemOf[Ten <: Tensor[?, ?]] = Ten match
  case Tensor[t, v] => v

/** Input to a `sample` call: either a free Site or an observed (fixed) site. */
sealed trait SiteVariable[Ten <: Tensor[?, ?]]:
  def name: String
  def site: Site[?]
  given labels: Labels[ShapeOf[Ten]]

/** A typed, named descriptor for a model variable.
  *
  * Declare as `object Alpha extends Site[Tensor1[Feature, Float]]`. Pass to `sample` as a free variable, or fix a value
  * with `site := value`.
  */
trait Site[Ten <: Tensor[?, ?]](using _lbl: Labels[ShapeOf[Ten]]) extends SiteVariable[Ten]:
  def name: String = getClass.getSimpleName.stripSuffix("$")
  def site: Site[?] = this

  /** The tensor type of this site. Use as `Alpha.Value` in case class fields. */
  type Value = Ten

  /** The model parameter type for this site (free or observed). Use as `Alpha.Variable` in model signatures. */
  type Variable = SiteVariable[Ten]

  override given labels: Labels[ShapeOf[Ten]] = _lbl

  /** Fix a value: `alpha := tensor` produces an [[ObservedVariable]]. */
  def :=(value: Ten): ObservedVariable[Ten] =
    ObservedVariable(this, value)(using _lbl)

object Site:
  def apply[Ten <: Tensor[?, ?]](siteName: String)(using Labels[ShapeOf[Ten]]): Site[Ten] =
    new Site[Ten]:
      override def name: String = siteName

/** A site with a fixed observed value. Create via `site := value`. */
case class ObservedVariable[Ten <: Tensor[?, ?]](
    parentSite: Site[?],
    value: Ten
)(using val lbl: Labels[ShapeOf[Ten]])
    extends SiteVariable[Ten]:
  def name: String = parentSite.name
  def site: Site[?] = parentSite
  given labels: Labels[ShapeOf[Ten]] = lbl

type SiteInput0[V] = SiteVariable[Tensor0[V]]
type SiteInput1[L, V] = SiteVariable[Tensor1[L, V]]
type SiteInput2[L1, L2, V] = SiteVariable[Tensor2[L1, L2, V]]
