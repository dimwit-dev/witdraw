package witdraw

import dimwit.*
import dimwit.tensor.Label

/** Names for the entries along an axis, e.g. the predictors of a coefficient
  * vector.
  */
final case class AxisNames[L](axis: Axis[L], names: Seq[String])(using
    val label: Label[L]
)

extension [L: Label](axis: Axis[L])
  /** Names for the entries along this axis; see [[AxisNames]]. */
  def names(names: Seq[String]): AxisNames[L] = AxisNames(axis, names)
