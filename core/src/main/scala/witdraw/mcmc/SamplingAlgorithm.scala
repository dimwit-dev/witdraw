package witdraw
package mcmc

import dimwit.*
import dimwit.autodiff.{Autodiff, Grad}
import dimwit.random.Random.Key
import dimwit.optimizer.GradientOptimizer
import dimwit.tensortree.FloatTreeFor

/** The state carried between kernel steps. Every state must expose its chain
  * position.
  */
trait KernelState[P]:
  def position: P

case class SamplingAlgorithm[P, S <: KernelState[P], I](
    init: P => S,
    step: (Key, S) => (S, I)
):
  /** Infinite iterator of `(position, info)` pairs, driven by key splitting. */
  def run(initial: P, key: Key): Iterator[(P, I)] =
    Iterator.unfold((init(initial), key)) { case (state, rng) =>
      val (k1, k2) = rng.split2()
      val (next, info) = step(k1, state)
      Some(((next.position, info), (next, k2)))
    }

/** Low-level MCMC algorithm: the log-density is passed explicitly at each call.
  *
  * Mirrors BlackJax's lower-level API where `logdensity_fn` is threaded through
  * both `init` and `kernel`. Passing the density explicitly enables adaptation
  * (e.g. mass-matrix tuning) where the density or its metric may change between
  * steps.
  *
  * {{{
  * val algo: MCMCAlgorithm[P, S, I] = ...
  * val state              = algo.init(position, logDensity)
  * val (next, info)       = algo.kernel(key, state, logDensity)
  * }}}
  */
trait MCMCAlgorithm[P, S <: KernelState[P], I]:
  def init(position: P, logDensity: P => Tensor0[Float32]): S
  def kernel(key: Key, state: S, logDensity: P => Tensor0[Float32]): (S, I)

  /** Close over a fixed log-density to produce a [[SamplingAlgorithm]]. */
  def withLogDensity(
      logDensity: P => Tensor0[Float32]
  ): SamplingAlgorithm[P, S, I] =
    val algo = this
    SamplingAlgorithm(
      init = pos => algo.init(pos, logDensity),
      step = (key, state) => algo.kernel(key, state, logDensity)
    )

case class OptimizerKernelState[P, S](
    position: P,
    optState: S
) extends KernelState[P]

object OptimizerKernel:
  def toSamplingAlgorithm[P: TensorTree: FloatTreeFor[Float32]](
      optimizer: GradientOptimizer,
      gradFn: P => Grad[P]
  ): SamplingAlgorithm[
    P,
    OptimizerKernelState[P, optimizer.State[P, Float32]],
    Unit
  ] =
    SamplingAlgorithm(
      init = p => OptimizerKernelState(p, optimizer.init(p)),
      step = (_, state) =>
        val (newParams, newOptState) = optimizer.update(
          gradFn(state.position),
          state.position,
          state.optState
        )
        (OptimizerKernelState(newParams, newOptState), ())
    )
