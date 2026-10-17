package witdraw

import dimwit.*
import dimwit.tensor.Labels
import dimwit.tensortree.TensorTree

/** A change of variables for a latent site: inference algorithms move freely in
  * an unconstrained space `Z`, and the model sees `bijector.forward(z)`. For
  * example, `Constraint(Sigma, Bijector.exp)` keeps `Sigma` positive.
  */
sealed abstract class Constraint[T <: Tuple, V](val site: SiteDef[T, V]):
  /** The type of the unconstrained value. */
  type Z
  private[witdraw] def bijector: Bijector[Z, Tensor[T, V]]
  private[witdraw] def zTree: TensorTree[Z]

object Constraint:
  def apply[TZ <: Tuple: Labels, VZ, T <: Tuple, V](
      site: SiteDef[T, V],
      bijector: Bijector[Tensor[TZ, VZ], Tensor[T, V]]
  ): Constraint[T, V] =
    val b = bijector
    new Constraint(site):
      type Z = Tensor[TZ, VZ]
      def bijector = b
      def zTree = TensorTree.tensor[TZ, VZ]

  /** No change of variables: inference moves in the site's own space. */
  private[witdraw] def none[T <: Tuple, V](s: SiteDef[T, V]): Constraint[T, V] =
    Constraint(s, Bijector.identity[T, V])(using s.labels)
