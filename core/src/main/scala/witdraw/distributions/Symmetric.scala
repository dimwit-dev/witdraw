package witdraw.distributions

import dimwit.*
import dimwit.stats.*

/** Distributions of class `D` that are symmetric around their location `loc`,
  * element-wise. For example, [[Half]] folds such a distribution at `loc`.
  */
trait Symmetric[-D, T <: Tuple, V]:
  def loc(d: D): Tensor[T, V]

object Symmetric:
  given normal[T <: Tuple, V]: Symmetric[Normal[T, V], T, V] = _.loc
  given cauchy[T <: Tuple, V]: Symmetric[Cauchy[T, V], T, V] = _.loc
  given studentT[T <: Tuple, V]: Symmetric[StudentT[T, V], T, V] = _.loc
