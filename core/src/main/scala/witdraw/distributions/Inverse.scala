package witdraw.distributions

import dimwit.*
import dimwit.stats.*
import witdraw.{Support, SupportOf}

/** The reciprocal of the distribution `base`: `y = 1 / X` with `X ~ base`. Its
  * density is `p_base(1 / y) / y^2` for `y > 0`, and zero elsewhere. For
  * example, `Inverse(Gamma(a, b))` is the inverse-gamma distribution with
  * shape `a` and scale `b`.
  *
  * Create it with [[Inverse.apply]], which accepts only [[Positive]]
  * distributions.
  */
class Inverse[T <: Tuple: Labels, V: IsFloating] private (
    val base: IndependentDistribution[T, V],
    val zero: Tensor[T, V]
) extends IndependentDistribution[T, V]:

  override def elementWiseLogProb(y: Tensor[T, V]): Tensor[T, LogProb] =
    val reciprocal = Tensor.like(y).fill(1f) / y
    val logJacobian = y.log.asFloat32 *! Tensor0(-2f)
    val density = base.elementWiseLogProb(reciprocal).asFloat + logJacobian
    val negInf = Tensor.like(density).fill(Float.NegativeInfinity)
    LogProb(where(y > zero, density, negInf))

  override def sample(k: Random.Key): Tensor[T, V] =
    val x = base.sample(k)
    Tensor.like(x).fill(1f) / x

object Inverse:

  /** The reciprocal of `base`. */
  def apply[D <: IndependentDistribution[T, V], T <: Tuple: Labels, V: IsFloating](
      base: D
  )(using positive: Positive[D, T, V]): Inverse[T, V] =
    new Inverse(base, positive.zero(base))

  given support[T <: Tuple, V]: SupportOf[Inverse[T, V], T, V] =
    d => Support.GreaterThan(d.zero)
