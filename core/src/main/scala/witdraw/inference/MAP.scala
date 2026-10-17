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
      maxSteps: Int = 10000,
      learningRate: Float = 0.05f
  ): latent.Params =
    maximise("MAP", latent.modelLogDensity, latent.initWith(start*), maxSteps, learningRate)

/** Maximises `f` from `start` with Adam. It stops once `f` has increased by
  * less than `tolerance` over a round of 100 steps, and fails if that has not
  * happened after `maxSteps` steps.
  */
private[witdraw] def maximise[P: TensorTree](
    what: String,
    f: P => Tensor0[Float32],
    start: P,
    maxSteps: Int,
    learningRate: Float,
    tolerance: Float = 1e-3f
)(using TreeOf[P, Float32]): P =
  val value = jit(f)
  val gradient = jit((p: P) => Autodiff.grad(f)(p).value)
  val ascent = Adam(Tensor0(-learningRate)) // a negative rate maximises
  val round = math.min(100, maxSteps)

  // the position after 0, 100, 200, … steps, and the log-density there
  val checkpoints = ascent
    .iterate(start)(p => Grad(gradient(p)))
    .zipWithIndex
    .collect { case (p, step) if step % round == 0 => p }
    .take(maxSteps / round + 1)
    .map(p => (p, value(p).item))

  checkpoints
    .sliding(2)
    .collectFirst:
      case Seq(_, (_, height)) if !java.lang.Float.isFinite(height) =>
        throw IllegalStateException(
          s"$what diverged: the log-density became $height. Reduce learningRate."
        )
      case Seq((_, before), (p, after)) if after - before < tolerance => p
    .getOrElse:
      throw IllegalStateException(
        s"$what did not converge in $maxSteps steps: the log-density was still increasing. " +
          "Increase maxSteps or learningRate. If it keeps increasing, the density may have no maximum, " +
          "e.g. when a scale parameter of a hierarchical model goes to 0."
      )
