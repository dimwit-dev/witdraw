package witdraw

import dimwit.*
import dimwit.tensor.Labels
import dimwit.tensortree.TensorTree

/** A bijector chosen for a site, as from
  * `Mu.withBijector(Bijector.ordered[Comp])`, to pass to [[transform]].
  */
final class SiteBijector[T <: Tuple, V] private[witdraw] (
    val site: SiteDef[T, V],
    private[witdraw] val bijector: BijectorTo[T, V]
)

/** Runs `model` with the given sites sampled through the given bijectors,
  * instead of the ones [[Latent.discover]] reads off the [[Support]] of their
  * distributions.
  * {{{
  * Latent.discover:
  *   transform(Mu.withBijector(Bijector.ordered[Comp])):
  *     observe(Y := y)(model(x))
  * }}}
  * Inference moves freely in an unconstrained space, and the model sees
  * `bijector.forward(z)`. A bijector whose values are a part of the support
  * restricts the posterior to that part: `Bijector.ordered` keeps `Mu`
  * increasing, which its `Normal` prior does not say. Only inference uses the
  * bijectors: `ForwardSampler` and `ModelScorer` run the model as it is.
  *
  * As with [[observe]], it is an error to give a site a bijector twice, or to
  * give one to a site the model never samples. Only latent sites can have a
  * bijector, not observed ones.
  */
def transform[A](bijectors: SiteBijector[?, ?]*)(model: ProbProg[A])(using
    ProbContext
): A =
  intercept("given a bijector", "transform", bijectors.map(b => b.site -> b.bijector))(
    [T <: Tuple, V] =>
      (statement: SampleStatement[T, V], bijector: BijectorTo[?, ?]) =>
        if statement.bijector.isDefined then
          throw IllegalArgumentException(
            s"Site '${statement.site.name}' is given a bijector twice, by nested transform blocks."
          )
        statement.copy(bijector = Some(bijector.asInstanceOf[BijectorTo[T, V]]))
  )(model)

/** A bijector to values `Tensor[T, V]`, from an unconstrained type `Z` that it
  * hides, together with what [[Latent]] needs to flatten `Z` into its parameter
  * vector.
  */
private[witdraw] sealed abstract class BijectorTo[T <: Tuple, V]:
  type Z
  def bijector: Bijector[Z, Tensor[T, V]]
  def zTree: TensorTree[Z]

private[witdraw] object BijectorTo:
  def apply[TZ <: Tuple: Labels, VZ, T <: Tuple, V](
      b: Bijector[Tensor[TZ, VZ], Tensor[T, V]]
  ): BijectorTo[T, V] =
    new BijectorTo[T, V]:
      type Z = Tensor[TZ, VZ]
      def bijector = b
      def zTree = TensorTree.tensor[TZ, VZ]

  /** No change of variables: inference moves in the site's own space. */
  def identity[T <: Tuple: Labels, V]: BijectorTo[T, V] =
    BijectorTo(Bijector.identity[T, V])
