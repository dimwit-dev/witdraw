package witdraw

import scala.NamedTuple.NamedTuple

/** The names of the fields of a named tuple, e.g. `(sigma = ..., beta = ...)`,
  * which name the rows of a summary or a plot.
  */
trait FieldNames[R]:
  def names: Seq[String]

object FieldNames:
  inline given namedTuple[N <: Tuple, V <: Tuple]: FieldNames[NamedTuple[N, V]] =
    val fieldNames = compiletime.constValueTuple[N].toList.map(_.toString)
    new FieldNames[NamedTuple[N, V]]:
      def names = fieldNames
