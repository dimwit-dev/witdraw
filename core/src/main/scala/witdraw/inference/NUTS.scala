package witdraw
package inference

import dimwit.*
import dimwit.python.PyBridge
import dimwit.random.Random.Key
import dimwit.tensortree.TensorTree
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

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

/** The inverse mass matrix of [[NUTS]]. The sampler works best when it
  * approximates the covariance of the target, which [[NUTS.adapt]] estimates.
  */
sealed trait InverseMassMatrix[P]:
  /** BlackJax's representation, a vector or a matrix over the flattened
    * parameters. `position` gives the structure of the parameters.
    */
  private[witdraw] def toPy(position: P)(using TensorTree[P]): py.Dynamic

object InverseMassMatrix:
  /** The identity: every direction is scaled alike. */
  case class Identity[P]() extends InverseMassMatrix[P]:
    private[witdraw] def toPy(position: P)(using TensorTree[P]): py.Dynamic =
      py.module("jax.numpy").ones_like(flatten(position))

  /** A diagonal matrix: `variances` holds the variance of each element of the
    * parameters.
    */
  case class Diagonal[P](variances: P) extends InverseMassMatrix[P]:
    private[witdraw] def toPy(position: P)(using TensorTree[P]): py.Dynamic =
      flatten(variances)

  /** A dense matrix over the flattened parameters, as estimated by
    * [[NUTS.adapt]].
    */
  final class Dense[P] private[witdraw] (private val raw: py.Dynamic)
      extends InverseMassMatrix[P]:
    private[witdraw] def toPy(position: P)(using TensorTree[P]): py.Dynamic =
      raw

  private def flatten[P: TensorTree](p: P): py.Dynamic =
    py.module("jax.flatten_util")
      .ravel_pytree(TensorTree[P].toPyTree(p))
      .bracketAccess(0)

/** The result of [[NUTS.adapt]]: the tuned step size and inverse mass matrix,
  * a sampler that uses them, and the position at the end of the warmup, from
  * which to start it.
  */
case class NUTSAdaptation[P](
    sampler: SamplingAlgorithm[P, NUTSState[P], NUTSInfo],
    position: P,
    stepSize: Double,
    inverseMassMatrix: InverseMassMatrix[P],
    numDivergent: Int
)

/** The No-U-Turn Sampler, a variant of Hamiltonian Monte Carlo that chooses the
  * length of each trajectory itself, run by BlackJax.
  *
  * [[NUTS.adapt]] runs a warmup that tunes the step size and the inverse mass
  * matrix, and returns a sampler that uses them:
  * {{{
  * val (warmupKey, sampleKey) = key.split2()
  * val adapted = NUTS.adapt(latent, latent.init, warmupKey)
  * val draws = adapted.sampler
  *   .run(adapted.position, sampleKey)
  *   .take(2000)
  *   .map(_._1)
  *   .toList
  * draws.map(_(Sigma))
  * }}}
  *
  * `NUTS(latent, stepSize)` instead uses a fixed step size and the identity
  * mass matrix: with many divergences or a low acceptance rate, reduce the
  * step size.
  */
object NUTS:
  /** NUTS for the posterior of `latent`: it samples [[Latent.logDensity]]. */
  def apply(
      latent: Latent,
      stepSize: Double
  ): SamplingAlgorithm[latent.Params, NUTSState[latent.Params], NUTSInfo] =
    apply(latent, stepSize, InverseMassMatrix.Identity())

  /** NUTS for the posterior of `latent`, with the given inverse mass matrix. */
  def apply(
      latent: Latent,
      stepSize: Double,
      inverseMassMatrix: InverseMassMatrix[latent.Params]
  ): SamplingAlgorithm[latent.Params, NUTSState[latent.Params], NUTSInfo] =
    apply[latent.Params](stepSize, inverseMassMatrix)(latent.logDensity)

  /** NUTS for `logDensity`, for any parameter type. */
  def apply[P: TensorTree](stepSize: Double)(
      logDensity: P => Tensor0[Float32]
  ): SamplingAlgorithm[P, NUTSState[P], NUTSInfo] =
    apply[P](stepSize, InverseMassMatrix.Identity[P]())(logDensity)

  /** NUTS for `logDensity`, for any parameter type, with the given inverse
    * mass matrix.
    */
  def apply[P: TensorTree](
      stepSize: Double,
      inverseMassMatrix: InverseMassMatrix[P]
  )(
      logDensity: P => Tensor0[Float32]
  ): SamplingAlgorithm[P, NUTSState[P], NUTSInfo] =
    val pyLogDensity = PyBridge.toPyFn(logDensity)
    SamplingAlgorithm(
      init = position =>
        build(
          pyLogDensity,
          py.Any.from(stepSize),
          inverseMassMatrix.toPy(position),
          TensorTree[P].toPyTree(position).as[py.Dynamic]
        )
      ,
      step = step[P]
    )

  /** Runs Stan's window adaptation for the posterior of `latent`, starting
    * from `initial`: `numSteps` NUTS steps, during which the step size is
    * tuned by dual averaging towards `targetAcceptanceRate`, and the inverse
    * mass matrix is estimated from the draws in windows of growing length.
    *
    * The mass matrix is diagonal, or `dense` over all parameter elements. A
    * dense one captures correlations, but it needs more warmup steps to be
    * estimated well.
    */
  def adapt(
      latent: Latent,
      initial: latent.Params,
      key: Key,
      numSteps: Int = 1000,
      dense: Boolean = false,
      targetAcceptanceRate: Double = 0.8
  ): NUTSAdaptation[latent.Params] =
    adapt[latent.Params](initial, key, numSteps, dense, targetAcceptanceRate)(
      latent.logDensity
    )

  /** Runs Stan's window adaptation for `logDensity`, for any parameter type.
    * See the other `adapt`.
    */
  def adapt[P: TensorTree](
      initial: P,
      key: Key,
      numSteps: Int,
      dense: Boolean,
      targetAcceptanceRate: Double
  )(
      logDensity: P => Tensor0[Float32]
  ): NUTSAdaptation[P] =
    val blackjax = py.module("blackjax")
    val pyLogDensity = PyBridge.toPyFn(logDensity)
    // keep only the divergences of the warmup steps, not their states
    val infoFn = blackjax.adaptation.base.get_filter_adapt_info_fn(
      py.Dynamic.global.set(),
      py.Dynamic.global.set(Seq("is_divergent").toPythonProxy),
      py.Dynamic.global.set()
    )
    val warmup = blackjax.window_adaptation(
      blackjax.nuts,
      pyLogDensity,
      is_mass_matrix_diagonal = !dense,
      target_acceptance_rate = targetAcceptanceRate,
      adaptation_info_fn = infoFn
    )
    val result = warmup.run(
      key.jaxKey,
      TensorTree[P].toPyTree(initial),
      numSteps
    )
    val adapted = result.bracketAccess(0)
    val info = result.bracketAccess(1)
    val parameters = adapted.parameters
    val pyStepSize = parameters.bracketAccess("step_size")
    val pyInverseMass = parameters.bracketAccess("inverse_mass_matrix")
    val pyPosition = adapted.state.position

    val inverseMassMatrix: InverseMassMatrix[P] =
      if dense then InverseMassMatrix.Dense(pyInverseMass)
      else
        val unravel =
          py.module("jax.flatten_util").ravel_pytree(pyPosition).bracketAccess(1)
        InverseMassMatrix.Diagonal(TensorTree[P].fromPyTree(unravel(pyInverseMass)))

    val sampler = SamplingAlgorithm[P, NUTSState[P], NUTSInfo](
      init = position =>
        build(
          pyLogDensity,
          pyStepSize,
          pyInverseMass,
          TensorTree[P].toPyTree(position).as[py.Dynamic]
        )
      ,
      step = step[P]
    )
    NUTSAdaptation(
      sampler = sampler,
      position = TensorTree[P].fromPyTree(pyPosition),
      stepSize = pyStepSize.item().as[Double],
      inverseMassMatrix = inverseMassMatrix,
      numDivergent = info.info.is_divergent.sum().item().as[Int]
    )

  private def build[P: TensorTree](
      pyLogDensity: py.Dynamic,
      stepSize: py.Any,
      inverseMassMatrix: py.Dynamic,
      pyPosition: py.Dynamic
  ): NUTSState[P] =
    val nuts =
      py.module("blackjax").nuts(pyLogDensity, stepSize, inverseMassMatrix)
    NUTSState(nuts.init(pyPosition), py.module("jax").jit(nuts.step))

  private def step[P: TensorTree](
      key: Key,
      state: NUTSState[P]
  ): (NUTSState[P], NUTSInfo) =
    val result = state.jittedStep(key.jaxKey, state.raw)
    (
      state.copy(raw = result.bracketAccess(0)),
      NUTSInfo(result.bracketAccess(1))
    )
