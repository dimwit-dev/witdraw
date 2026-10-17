package witdraw
package inference

import dimwit.*
import dimwit.autodiff.{Autodiff, Grad}
import dimwit.optimizer.Adam
import dimwit.tensortree.{TensorTree, TreeOf}

/** The maximum a posteriori (MAP) estimate of the latent sites: the values at
  * which the model's density is highest.
  *
  * {{{
  * val map = MAP(latent)
  * map(Sigma)   // the MAP of Sigma
  * }}}
  *
  * The search starts from the traced values, or from the given ones, e.g.
  * `MAP(latent, start = Seq(Sigma := Tensor0(1f)))`. It moves in the
  * unconstrained space, so it stays inside the constraints, and it maximises
  * [[Latent.modelLogDensity]], so the result does not depend on them.
  */
object MAP:
  def apply(
      latent: Latent,
      start: Seq[Observation[?, ?]] = Nil,
      steps: Int = 1000,
      learningRate: Float = 0.05f
  ): latent.Params =
    maximise("MAP", latent.modelLogDensity, latent.initWith(start*), steps, learningRate)

/** Maximises `f` from `start` with Adam, and checks that it got there: over
  * the last tenth of the steps, `f` must have stopped increasing.
  */
private[witdraw] def maximise[P: TensorTree](
    what: String,
    f: P => Tensor0[Float32],
    start: P,
    steps: Int,
    learningRate: Float
)(using TreeOf[P, Float32]): P =
  val value = jit(f)
  val gradient = jit((p: P) => Autodiff.grad(f)(p).value)
  val ascent = Adam(Tensor0(-learningRate)) // a negative rate maximises

  def run(p: P, state: ascent.State[P], n: Int) =
    (1 to n).foldLeft((p, state)): (acc, _) =>
      ascent.update(Grad(gradient(acc._1)), acc._1, acc._2)

  val last = math.max(1, steps / 10)
  val (almost, state) = run(start, ascent.init(start), steps - last)
  val (result, _) = run(almost, state, last)
  val (before, after) = (value(almost).item, value(result).item)
  if !java.lang.Float.isFinite(after) || after - before > 1e-3f then
    throw IllegalStateException(
      s"$what did not converge in $steps steps: the log-density went from $before to $after over the last $last steps. " +
        "Increase steps or learningRate. If it keeps increasing, the density may have no maximum, " +
        "e.g. when a scale parameter of a hierarchical model goes to 0."
    )
  result
