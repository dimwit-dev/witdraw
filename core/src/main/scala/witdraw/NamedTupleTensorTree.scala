package witdraw

import dimwit.autodiff.{FloatTree, TensorTree}
import dimwit.jax.Jax
import dimwit.tensor.*

import scala.NamedTuple.NamedTuple

/** TensorTree and FloatTree instances for named tuples.
  *
  * These delegate to the underlying value tuple `V`, so any named tuple whose
  * field types all have TensorTree instances automatically gets one too. They
  * are top-level givens in the `witdraw` package, so `import witdraw.*` brings
  * them into scope automatically.
  *
  * This allows replacing:
  * {{{
  *   case class Posterior(x: Tensor0[Float32], y: Tensor1[L, Float32]) derives TensorTree
  * }}}
  * with a plain type alias:
  * {{{
  *   type Posterior = (x: Tensor0[Float32], y: Tensor1[L, Float32])
  * }}}
  */

given namedTupleTensorTree[N <: Tuple, V <: Tuple](using
    tt: TensorTree[V]
): TensorTree[NamedTuple[N, V]] with

  def map(
      p: NamedTuple[N, V],
      f: [T <: Tuple, Vt] => Labels[T] ?=> (Tensor[T, Vt] => Tensor[T, Vt])
  ): NamedTuple[N, V] =
    tt.map(p.toTuple, f).asInstanceOf[NamedTuple[N, V]]

  def mapWithName(
      p: NamedTuple[N, V],
      f: [T <: Tuple, Vt] => Labels[T] ?=> (
          (String, Tensor[T, Vt]) => Tensor[T, Vt]
      ),
      path: String = ""
  ): NamedTuple[N, V] =
    tt.mapWithName(p.toTuple, f, path).asInstanceOf[NamedTuple[N, V]]

  def mapLeaves[A](
      p: NamedTuple[N, V],
      f: [T <: Tuple, Vt] => Labels[T] ?=> (Tensor[T, Vt] => A)
  ): Iterator[A] =
    tt.mapLeaves(p.toTuple, f)

  def foreach(
      p: NamedTuple[N, V],
      f: [T <: Tuple, Vt] => Labels[T] ?=> (Tensor[T, Vt] => Unit)
  ): Unit =
    tt.foreach(p.toTuple, f)

  def foreachWithName(
      p: NamedTuple[N, V],
      f: [T <: Tuple, Vt] => Labels[T] ?=> ((String, Tensor[T, Vt]) => Unit),
      path: String = ""
  ): Unit =
    tt.foreachWithName(p.toTuple, f, path)

  def zipMap(
      p1: NamedTuple[N, V],
      p2: NamedTuple[N, V],
      f: [T <: Tuple, Vt] => Labels[T] ?=> (
          (Tensor[T, Vt], Tensor[T, Vt]) => Tensor[T, Vt]
      )
  ): NamedTuple[N, V] =
    tt.zipMap(p1.toTuple, p2.toTuple, f).asInstanceOf[NamedTuple[N, V]]

  def toPyTree(p: NamedTuple[N, V]): Jax.PyAny =
    tt.toPyTree(p.toTuple)

  def fromPyTree(pyVal: Jax.PyAny): NamedTuple[N, V] =
    tt.fromPyTree(pyVal).asInstanceOf[NamedTuple[N, V]]

given namedTupleFloatTree[N <: Tuple, V <: Tuple, Fl](using
    FloatTree[V, Fl]
): FloatTree[NamedTuple[N, V], Fl] with {}
