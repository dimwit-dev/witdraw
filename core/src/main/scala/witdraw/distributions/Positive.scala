package witdraw.distributions

import dimwit.*
import dimwit.stats.*

/** Distributions of class `D` whose values are positive, element-wise.
  * For example, [[Inverse]] takes the reciprocal of such a distribution.
  * `zero` gives the lower bound of the support, a tensor of zeros of the
  * distribution's shape.
  */
trait Positive[-D, T <: Tuple, V]:
  def zero(d: D): Tensor[T, V]

object Positive:
  given gamma[T <: Tuple: Labels, V: IsFloating]: Positive[Gamma[T, V], T, V] =
    d => Tensor.like(d.rate).fill(0f)
  given exponential[T <: Tuple: Labels, V: IsFloating]
      : Positive[Exponential[T, V], T, V] =
    d => Tensor.like(d.rate).fill(0f)
