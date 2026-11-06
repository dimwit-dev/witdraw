package witdraw.distributions

import dimwit.*
import dimwit.stats.*
import witdraw.{Support, SupportOf}

/** The distribution `base` folded at its location: `x = loc + |X - loc|` with
  * `X ~ base`. Its density is twice that of `base` for `x >= loc`, and zero
  * below. For example, `Half(StudentT(5f, 0f, 1f))` is a half-Student-t, and
  * `Half(Normal(0f, 1f))` is the same as `HalfNormal(0f, 1f)`.
  *
  * Create it with [[Half.apply]], which accepts only [[Symmetric]]
  * distributions.
  */
class Half[T <: Tuple: Labels, V: IsFloating] private (
    val base: IndependentDistribution[T, V],
    val loc: Tensor[T, V]
) extends IndependentDistribution[T, V]:

  private val log2 = Tensor0(math.log(2).toFloat)

  override def elementWiseLogProb(x: Tensor[T, V]): Tensor[T, LogProb] =
    val folded = base.elementWiseLogProb(x).asFloat +! log2
    val negInf = Tensor.like(folded).fill(Float.NegativeInfinity)
    LogProb(where(x >= loc, folded, negInf))

  override def sample(k: Random.Key): Tensor[T, V] =
    (base.sample(k) - loc).abs + loc

object Half:

  /** `base` folded at its location. */
  def apply[D <: IndependentDistribution[T, V], T <: Tuple: Labels, V: IsFloating](
      base: D
  )(using symmetric: Symmetric[D, T, V]): Half[T, V] =
    new Half(base, symmetric.loc(base))

  given support[T <: Tuple, V]: SupportOf[Half[T, V], T, V] =
    d => Support.GreaterThan(d.loc)
