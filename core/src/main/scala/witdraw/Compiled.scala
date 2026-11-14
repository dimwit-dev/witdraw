package witdraw

import dimwit.*
import dimwit.tensortree.TensorTree

/** Compiles the functions of one model with `jit`, each on first use. A model
  * that needs concrete values, e.g. calls `.item` on a sampled value, cannot be
  * compiled: once one function fails to compile, all of them run eagerly.
  */
private[witdraw] final class Compiled:
  @volatile private var failed = false

  def apply[A, B](eager: A => B)(using TensorTree[A], TensorTree[B]): A => B =
    lazy val compiled = jit(eager)
    a => orEager(compiled(a), eager(a))

  def apply[A, B, C](eager: (A, B) => C)(using
      TensorTree[A],
      TensorTree[B],
      TensorTree[C]
  ): (A, B) => C =
    lazy val compiled = jit(eager)
    (a, b) => orEager(compiled(a, b), eager(a, b))

  /** `compiled`, or `eager` once compiling has failed. */
  private def orEager[X](compiled: => X, eager: => X): X =
    if failed then eager
    else
      try compiled
      catch
        case scala.util.control.NonFatal(_) =>
          failed = true
          System.err.println(
            "witdraw: the model cannot be compiled, e.g. as the model calls `.item` on a sampled value, so it runs eagerly, which is slower."
          )
          eager
