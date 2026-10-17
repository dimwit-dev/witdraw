package witdraw

import dimwit.*
import dimwit.stats.Distribution
import dimwit.tensor.{Labels, Tensor}

/** A named random variable of a model, written with the tensor type of its
  * values. Declare a site as an object; its name is the object's name:
  * {{{
  * object Beta extends Site[Tensor1[Feature, Float32]]
  * }}}
  *
  * Inside a model, `Beta ~ dist` says that the site is distributed according to
  * `dist`. Outside, `Beta := value` creates an [[Observation]] for [[observe]].
  *
  * Sites are compared by identity: two different objects are two different
  * sites, even if they have the same name. The name is only used for display,
  * e.g. in error messages.
  */
type Site[Ten <: Tensor[?, ?]] = Ten match
  case Tensor[t, v] => SiteDef[t, v]

/** The trait behind [[Site]]: `Site[Tensor[T, V]]` is `SiteDef[T, V]`. Code
  * that works with any site uses `SiteDef[T, V]` or `SiteDef[?, ?]`.
  */
trait SiteDef[T <: Tuple, V](using val labels: Labels[T]):
  def name: String = getClass.getSimpleName.stripSuffix("$")

  /** The tensor type of this site, e.g. `Beta.Value` in a parameter record. */
  type Value = Tensor[T, V]

  /** This site is distributed according to `dist`. Returns the site's value:
    * drawn from `dist`, or an observed value, depending on the handlers and the
    * interpreter that run the model.
    */
  infix def ~(dist: Distribution[T, V])(using ctx: ProbContext): Tensor[T, V] =
    ctx.sample(SampleStatement(this, dist, observed = None))

  /** An observation of this site, to pass to [[observe]]. */
  def :=(value: Tensor[T, V]): Observation[T, V] = Observation(this, value)

  override def toString: String = s"Site($name)"

/** A value for a site, created with `site := value`. */
case class Observation[T <: Tuple, V](site: SiteDef[T, V], value: Tensor[T, V])
