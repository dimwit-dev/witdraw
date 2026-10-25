package witdraw

import dimwit.*
import dimwit.tensor.Labels
import dimwit.tensortree.TensorTree

/** A change of variables for a latent site, which overrides the one that
  * [[Latent.discover]] reads off the [[Support]] of the site's distribution.
  * Inference algorithms move freely in an unconstrained space, and the model
  * sees `bijector.forward(z)`. For example,
  * `Constraint(Mu, Bijector.ordered[Comp])` keeps `Mu` increasing, which its
  * `Normal` prior does not say.
  */
final class Constraint[T <: Tuple, V] private (
    val site: SiteDef[T, V],
    private[witdraw] val transform: Transform[T, V]
)

object Constraint:
  def apply[TZ <: Tuple: Labels, VZ, T <: Tuple, V](
      site: SiteDef[T, V],
      bijector: Bijector[Tensor[TZ, VZ], Tensor[T, V]]
  ): Constraint[T, V] =
    new Constraint(site, Transform(bijector))

/** A bijector to values `Tensor[T, V]`, from an unconstrained type `Z` that it
  * hides, together with what [[Latent]] needs to flatten `Z` into its parameter
  * vector.
  */
private[witdraw] sealed abstract class Transform[T <: Tuple, V]:
  type Z
  def bijector: Bijector[Z, Tensor[T, V]]
  def zTree: TensorTree[Z]

private[witdraw] object Transform:
  def apply[TZ <: Tuple: Labels, VZ, T <: Tuple, V](
      b: Bijector[Tensor[TZ, VZ], Tensor[T, V]]
  ): Transform[T, V] =
    new Transform[T, V]:
      type Z = Tensor[TZ, VZ]
      def bijector = b
      def zTree = TensorTree.tensor[TZ, VZ]

  /** No change of variables: inference moves in the site's own space. */
  def identity[T <: Tuple: Labels, V]: Transform[T, V] =
    Transform(Bijector.identity[T, V])
