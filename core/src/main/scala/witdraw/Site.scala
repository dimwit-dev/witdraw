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
  * e.g. in error messages. A site declared inside a [[Component]] is named by
  * the component's path, e.g. `tv.Effect`.
  */
type Site[Ten <: Tensor[?, ?]] = Ten match
  case Tensor[t, v] => SiteDef[t, v]

/** The trait behind [[Site]]: `Site[Tensor[T, V]]` is `SiteDef[T, V]`. Code
  * that works with any site uses `SiteDef[T, V]` or `SiteDef[?, ?]`.
  */
trait SiteDef[T <: Tuple, V](using val labels: Labels[T], owner: Scope):
  def name: String =
    (owner.path :+ getClass.getSimpleName.stripSuffix("$")).mkString(".")

  /** The tensor type of this site, e.g. `Beta.Value` in a parameter record. */
  type Value = Tensor[T, V]

  /** This site is distributed according to `dist`. Returns the site's value:
    * drawn from `dist`, or an observed value, depending on the handlers and the
    * interpreter that run the model.
    *
    * The [[Support]] of `dist` is looked up for its static type `D`, see
    * [[SupportOf]].
    */
  infix def ~[D <: Distribution[T, V]](dist: D)(using
      ctx: ProbContext,
      supportOf: SupportOf[D, T, V] = SupportOf.unknown[D, T, V]
  ): Tensor[T, V] =
    ctx.sample(
      SampleStatement(this, dist, supportOf(dist), observed = None)
    )

  /** An observation of this site, to pass to [[observe]]. */
  def :=(value: Tensor[T, V]): Observation[T, V] = Observation(this, value)

  override def toString: String = s"Site($name)"

/** A value for a site, created with `site := value`. */
case class Observation[T <: Tuple, V](site: SiteDef[T, V], value: Tensor[T, V])

/** Values of sites, e.g. of one run of a model ([[Trace]]) or of a point in a
  * posterior (`latent.values(params)`).
  *
  * It lets a [[Component]] read its values as a typed record, the same way from
  * every source:
  * {{{
  * trait Linear extends Component:
  *   object Intercept extends Site[Tensor0[Float32]]
  *   object Weights extends Site[Tensor1[Feature, Float32]]
  *   def values(v: SiteValues) = Linear.Values(v(Intercept), v(Weights))
  *
  * object Linear:
  *   case class Values(intercept: Tensor0[Float32], weights: Tensor1[Feature, Float32])
  * }}}
  *
  * A record of tensors like `Linear.Values` is a `TensorTree`, so it can be
  * averaged, saved or mapped over as a whole.
  */
trait SiteValues:
  /** The value of `site`. */
  def apply[T <: Tuple, V](site: SiteDef[T, V]): Tensor[T, V]
