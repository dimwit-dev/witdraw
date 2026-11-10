package witdraw.plots

import dimwit.*
import dimwit.tensor.Labels
import dimwit.tensortree.TensorTree

import scala.NamedTuple.NamedTuple

import witdraw.AxisNames

/** The names of the fields of a named tuple, e.g. `(sigma = ..., beta = ...)`,
  * which name the rows of a plot.
  */
trait FieldNames[R]:
  def names: Seq[String]

object FieldNames:
  inline given namedTuple[N <: Tuple, V <: Tuple]: FieldNames[NamedTuple[N, V]] =
    val fieldNames = compiletime.constValueTuple[N].toList.map(_.toString)
    new FieldNames[NamedTuple[N, V]]:
      def names = fieldNames

/** The draws of one quantity: one sequence per chain, all of the same length.
  */
private[plots] final class Draws[T <: Tuple](
    val name: String,
    val chains: Seq[Seq[Tensor[T, Float32]]],
    val elementNames: Seq[String]
)(using val labels: Labels[T]):

  require(chains.nonEmpty, "no chains given")
  val numDraws: Int = chains.head.size
  require(numDraws > 0, "the chains have no draws")
  require(
    chains.forall(_.size == numDraws),
    s"all chains must have the same number of draws, but have ${chains.map(_.size).distinct.mkString(", ")}"
  )

  /** The draws as nested arrays: chain, draw, flattened element. They are
    * stacked into one tensor and read at once, as reading them draw by draw is
    * slow.
    */
  lazy val values: Seq[Seq[Array[Double]]] =
    val all = Draws.flat(stack(chains.map(c => stack(c, Axis[Draws.Draw])), Axis[Draws.Chain]))
    val numElements = all.length / (chains.size * numDraws)
    Seq.tabulate(chains.size, numDraws): (c, d) =>
      val start = (c * numDraws + d) * numElements
      all.slice(start, start + numElements)

  def numElements: Int = elementNames.size

private[plots] object Draws:
  trait Chain derives Label
  trait Draw derives Label

  def flat[T <: Tuple: Labels](t: Tensor[T, Float32]): Array[Double] =
    t.flatten.toArray.map(_.toDouble)

  /** The draws of each field of `records`, one record per draw of each chain,
    * in the order of the fields. The elements are named after the field and,
    * along each axis, the entry's name from `axisNames`, or its index:
    * `beta[male]`, `m[0, 2]`; a scalar after the field alone.
    */
  def ofRecords[R](records: Seq[Seq[R]], axisNames: Seq[AxisNames[?]])(using
      tree: TensorTree[R],
      fields: FieldNames[R]
  ): Seq[Draws[?]] =
    require(records.nonEmpty && records.head.nonEmpty, "no draws given")
    val names = fields.names
    // the labels and the size of each axis, of each field
    val axes = tree
      .mapLeaves(
        records.head.head,
        [T <: Tuple, V] => (l: Labels[T]) ?=> (t: Tensor[T, V]) => (l: Labels[?], t.shape.dimensions.toList)
      )
      .toVector
    require(
      axes.size == names.size,
      s"each field must be one tensor, but the fields ${names.mkString(", ")} hold ${axes.size} tensors"
    )
    val byAxis = axisNames.map(a => a.label.name -> a.names).toMap
    require(byAxis.size == axisNames.size, "names given twice for the same axis")
    val leaves = records.map(_.map: r =>
      tree.mapLeaves(r, [T <: Tuple, V] => (_: Labels[T]) ?=> (t: Tensor[T, V]) => t: Tensor[?, ?]).toVector)
    names.indices.map: i =>
      val (labels, sizes) = axes(i)
      given Labels[Tuple] = labels.asInstanceOf[Labels[Tuple]]
      Draws[Tuple](
        names(i),
        leaves.map(_.map(_(i).asInstanceOf[Tensor[Tuple, Float32]])),
        elementNames(names(i), labels.names.zip(sizes), byAxis)
      )

  /** The names of the elements of a field, in the order of its flattened
    * values.
    */
  private def elementNames(
      field: String,
      axes: List[(String, Int)],
      byAxis: Map[String, Seq[String]]
  ): Seq[String] =
    val entries = axes.map: (axis, size) =>
      byAxis.get(axis) match
        case Some(names) =>
          require(names.size == size, s"${names.size} names for axis $axis of $field, which has $size entries")
          names
        case None => (0 until size).map(_.toString)
    if entries.isEmpty then Seq(field)
    else
      val combined = entries.foldLeft(Seq(List.empty[String]))((acc, names) => for a <- acc; n <- names yield a :+ n)
      combined.map(parts => s"$field[${parts.mkString(", ")}]")
