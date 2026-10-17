package witdraw
package inference

import dimwit.*
import dimwit.python.PyBridge
import dimwit.tensortree.TensorTree
import me.shadaj.scalapy.py

/** The state of [[NUTS]]: BlackJax's state, and its compiled step function.
  * `position` is decoded from BlackJax's state when asked for.
  */
case class NUTSState[P: TensorTree](
    private[witdraw] val raw: py.Dynamic,
    private[witdraw] val jittedStep: py.Dynamic
) extends KernelState[P]:
  def position: P = TensorTree[P].fromPyTree(raw.position)

/** Diagnostics of one NUTS step. Each field is read from BlackJax only when
  * asked for.
  */
case class NUTSInfo(private val raw: py.Dynamic):
  lazy val acceptanceRate: Double = raw.acceptance_rate.item().as[Double]
  lazy val numIntegrationSteps: Int = raw.num_integration_steps.item().as[Int]
  lazy val isDivergent: Boolean = raw.is_divergent.item().as[Boolean]
  lazy val energy: Double = raw.energy.item().as[Double]

/** The No-U-Turn Sampler, a variant of Hamiltonian Monte Carlo that chooses the
  * length of each trajectory itself, run by BlackJax.
  * {{{
  * val nuts = NUTS(latent, stepSize = 0.05)
  * val draws = nuts.run(latent.init, key).drop(500).take(2000).map(_._1).toList
  * draws.map(_(Sigma))
  * }}}
  *
  * The step size is not adapted: with many divergences or a low acceptance
  * rate, reduce it. The mass matrix is the identity.
  */
object NUTS:
  /** NUTS for the posterior of `latent`: it samples [[Latent.logDensity]]. */
  def apply(
      latent: Latent,
      stepSize: Double
  ): SamplingAlgorithm[latent.Params, NUTSState[latent.Params], NUTSInfo] =
    apply[latent.Params](stepSize)(latent.logDensity)

  /** NUTS for `logDensity`, for any parameter type. */
  def apply[P: TensorTree](stepSize: Double)(
      logDensity: P => Tensor0[Float32]
  ): SamplingAlgorithm[P, NUTSState[P], NUTSInfo] =
    val jax = py.module("jax")
    val blackjax = py.module("blackjax")
    val pyLogDensity = PyBridge.toPyFn(logDensity)
    SamplingAlgorithm(
      init = position =>
        val pyPosition = TensorTree[P].toPyTree(position)
        val leaves = jax.tree_util.tree_leaves(pyPosition)
        val identityMass = jax.numpy.ones_like(
          jax.numpy.concatenate(jax.tree_util.tree_map(jax.numpy.ravel, leaves))
        )
        val nuts = blackjax.nuts(pyLogDensity, stepSize, identityMass)
        NUTSState(nuts.init(pyPosition), jax.jit(nuts.step))
      ,
      step = (key, state) =>
        val result = state.jittedStep(key.jaxKey, state.raw)
        (
          state.copy(raw = result.bracketAccess(0)),
          NUTSInfo(result.bracketAccess(1))
        )
    )
