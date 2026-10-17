package witdraw

import dimwit.*
import dimwit.stats.Distribution
import dimwit.tensor.{Labels, Tensor}

/** The axis labels of a tensor type: `ShapeOf[Tensor1[L, V]] = Tuple1[L]`. */
type ShapeOf[Ten <: Tensor[?, ?]] <: Tuple = Ten match
  case Tensor[t, v] => t

/** The element type of a tensor type: `ElemOf[Tensor1[L, V]] = V`. */
type ElemOf[Ten <: Tensor[?, ?]] = Ten match
  case Tensor[t, v] => v

/** A distribution whose values have the tensor type `Ten`. */
type DistributionOf[Ten <: Tensor[?, ?]] =
  Distribution[ShapeOf[Ten], ElemOf[Ten]]

/** A named random variable of a model, whose values have the tensor type `Ten`.
  *
  * Declare a site as an object; its name is the object's name:
  * {{{
  * object Beta extends Site[Tensor1[Feature, Float32]]
  * }}}
  *
  * Inside a model, `Beta ~ dist` says that the site is distributed according to
  * `dist`. Outside, `Beta := value` creates an [[Observation]] for [[observe]].
  *
  * Sites are compared by identity: two different objects are two different
  * sites, even if they have the same name. The name is used in error messages
  * and by [[observeAll]], which matches field names to site names.
  */
trait Site[Ten <: Tensor[?, ?]](using Labels[ShapeOf[Ten]]):
  def name: String = getClass.getSimpleName.stripSuffix("$")

  /** The tensor type of this site, e.g. `Beta.Value` in a parameter record. */
  type Value = Ten

  /** This site is distributed according to `dist`. Returns the site's value:
    * drawn from `dist`, or an observed value, depending on the handlers and the
    * interpreter that run the model.
    */
  infix def ~(dist: DistributionOf[Ten])(using ctx: ProbContext): Ten =
    ctx.sample(SampleStatement(this, dist, observed = None))

  /** An observation of this site, to pass to [[observe]]. */
  def :=(value: Ten): Observation[Ten] = Observation(this, value)

  override def toString: String = s"Site($name)"

object Site:
  /** A site with an explicit name, e.g. for sites created in a loop. */
  def apply[Ten <: Tensor[?, ?]](siteName: String)(using
      Labels[ShapeOf[Ten]]
  ): Site[Ten] =
    new Site[Ten]:
      override def name: String = siteName

/** A value for a site, created with `site := value`. */
case class Observation[Ten <: Tensor[?, ?]](site: Site[Ten], value: Ten)
