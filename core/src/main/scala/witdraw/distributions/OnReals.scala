package witdraw.distributions

import dimwit.*
import dimwit.stats.*
import scala.annotation.implicitNotFound

/** Scalar distributions of class `D` that can take any real value. For
  * example, [[Ordered]] sorts draws of such a distribution.
  *
  * To use a distribution of your own, put an instance in its companion object:
  * {{{
  * object Logistic:
  *   given OnReals[Logistic] = OnReals.instance
  * }}}
  */
@implicitNotFound(
  "${D} is not known to take any real value, which Ordered needs, e.g. Normal, Cauchy or StudentT. Give it a `given OnReals[${D}]` if it does."
)
trait OnReals[-D]

object OnReals:
  /** An instance, to declare that a distribution takes any real value. */
  def instance[D]: OnReals[D] = new OnReals[D] {}

  given normal: OnReals[Normal[EmptyTuple, Float32]] = instance
  given cauchy: OnReals[Cauchy[EmptyTuple, Float32]] = instance
  given studentT: OnReals[StudentT[EmptyTuple, Float32]] = instance
