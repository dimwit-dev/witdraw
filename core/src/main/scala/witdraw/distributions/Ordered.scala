package witdraw.distributions

import dimwit.*
import dimwit.stats.*
import witdraw.{Support, SupportOf}

/** `k` independent draws from `base`, sorted: a strictly increasing vector
  * along `L`, as Stan's `ordered[k]`. Its density is `k!` times the density of
  * the `k` draws for increasing values, and zero elsewhere.
  *
  * Use it when the order of the values matters, e.g. to make the components of
  * a mixture identifiable:
  * {{{
  * val mu = Mu ~ Ordered(Normal(0f, 10f), Axis[Comp] -> k)
  * }}}
  *
  * `base` is one scalar distribution that takes any real value, see
  * [[OnReals]], so the `k` draws are independent and identically
  * distributed, as sorting them requires.
  */
class Ordered[L: Label] private (
    val base: Distribution[EmptyTuple, Float32],
    val size: Int
) extends Distribution[Tuple1[L], Float32]:

  private val axis = Axis[L]
  private val logFactorial =
    Tensor0((1 to size).map(i => math.log(i)).sum.toFloat)

  override def sample(k: Random.Key): Tensor1[L, Float32] =
    k.splitvmap(AxisExtent(axis, size))(base.sample).sort(axis)

  override def logProb(x: Tensor1[L, Float32]): Tensor0[LogProb] =
    val draws = x.vmap(axis)(v => base.logProb(v).asFloat).sum + logFactorial
    if size <= 1 then LogProb(draws)
    else
      val gaps =
        x.select(axis.at(1 until size)) - x.select(axis.at(0 until size - 1))
      val increasing = gaps.min > Tensor0(0f)
      LogProb(where(increasing, draws, Tensor0(Float.NegativeInfinity)))

object Ordered:

  /** `extent.size` draws from `base`, sorted along the axis of `extent`.
    * `base` must take any real value, e.g. `Normal` or `StudentT`.
    */
  def apply[D <: Distribution[EmptyTuple, Float32], L: Label](
      base: D,
      extent: AxisExtent[L]
  )(using OnReals[D]): Ordered[L] =
    new Ordered(base, extent.size)

  given support[L: Label]: SupportOf[Ordered[L], Tuple1[L], Float32] =
    _ => Support.Ordered()
