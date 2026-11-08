package witdraw
package inference

import dimwit.*
import dimwit.python.PyBridge
import dimwit.random.Random.Key
import dimwit.tensortree.TensorTree
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

/** The state of [[NUTS]]: the position, and the log-density and its gradient
  * there, as in BlackJax's `HMCState`. It also holds the step size and the
  * inverse mass matrix, so that chains with different ones, e.g. adapted
  * each on its own, share one compiled loop.
  */
case class NUTSState[P](
    position: P,
    logDensity: Tensor0[Float32],
    logDensityGrad: P,
    private[witdraw] val stepSize: Tensor0[Float32],
    /** The diagonal of the inverse mass matrix, or the dense matrix row by
      * row, over the flattened parameters.
      */
    private[witdraw] val inverseMass: Tensor1[MassElement, Float32]
) extends KernelState[P]:
  private[witdraw] def toPy(using tree: TensorTree[P]): py.Dynamic =
    py.module("blackjax.mcmc.hmc")
      .HMCState(
        tree.toPyTree(position),
        TensorTree[Tensor0[Float32]].toPyTree(logDensity),
        tree.toPyTree(logDensityGrad)
      )

  /** The inverse mass matrix in BlackJax's representation: a vector for a
    * diagonal one, a matrix for a dense one. Shapes are known while tracing,
    * so this is decided once per compilation.
    */
  private[witdraw] def pyInverseMass(using tree: TensorTree[P]): py.Dynamic =
    val n = py.module("jax.flatten_util")
      .ravel_pytree(tree.toPyTree(position))
      .bracketAccess(0)
      .size
      .as[Int]
    val flat = TensorTree[Tensor1[MassElement, Float32]].toPyTree(inverseMass).as[py.Dynamic]
    if flat.size.as[Int] == n then flat else flat.reshape(n, n)

object NUTSState:
  given [P: TensorTree]: TensorTree[NUTSState[P]] = TensorTree.derived

  /** The state for BlackJax's `raw` state, with the given settings. */
  private[witdraw] def fromPy[P](
      raw: py.Dynamic,
      stepSize: Tensor0[Float32],
      inverseMass: Tensor1[MassElement, Float32]
  )(using tree: TensorTree[P]): NUTSState[P] =
    NUTSState(
      tree.fromPyTree(raw.position),
      TensorTree[Tensor0[Float32]].fromPyTree(raw.logdensity),
      tree.fromPyTree(raw.logdensity_grad),
      stepSize,
      inverseMass
    )

/** The axis of the flattened inverse mass matrix in [[NUTSState]]. */
private[witdraw] trait MassElement derives Label

/** Diagnostics of one NUTS step. They are kept as tensors, so that a step can
  * run in a compiled loop, and each is read only when asked for.
  */
case class NUTSInfo(
    private[witdraw] val acceptanceRateT: Tensor0[Float32],
    private[witdraw] val numIntegrationStepsT: Tensor0[Int32],
    private[witdraw] val isDivergentT: Tensor0[Bool],
    private[witdraw] val energyT: Tensor0[Float32]
):
  def acceptanceRate: Double = acceptanceRateT.item.toDouble
  def numIntegrationSteps: Int = numIntegrationStepsT.item
  def isDivergent: Boolean = isDivergentT.item
  def energy: Double = energyT.item.toDouble
object NUTSInfo:
  given SamplerInfo[NUTSInfo] = _.isDivergent

  given TensorTree[NUTSInfo] = TensorTree.derived

  private[witdraw] def fromPy(raw: py.Dynamic): NUTSInfo =
    def leaf[T](x: py.Dynamic)(using tree: TensorTree[T]) = tree.fromPyTree(x)
    NUTSInfo(
      leaf[Tensor0[Float32]](raw.acceptance_rate),
      leaf[Tensor0[Int32]](raw.num_integration_steps),
      leaf[Tensor0[Bool]](raw.is_divergent),
      leaf[Tensor0[Float32]](raw.energy)
    )

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
  * which [[sample]] continues.
  */
case class NUTSAdaptation[P](
    sampler: SamplingAlgorithm[P, NUTSState[P], NUTSInfo],
    position: P,
    stepSize: Double,
    inverseMassMatrix: InverseMassMatrix[P],
    numDivergent: Int
):
  /** `numDraws` draws of the tuned sampler, continuing from the end of the
    * warmup.
    */
  def sample(key: Key, numDraws: Int)(using TensorTree[P]): Chain[P, NUTSInfo] =
    sampler.sample(key, position, numDraws)

/** The No-U-Turn Sampler, a variant of Hamiltonian Monte Carlo that chooses the
  * length of each trajectory itself, run by BlackJax.
  *
  * [[NUTS.adapt]] runs a warmup that tunes the step size and the inverse mass
  * matrix, and returns a sampler that uses them:
  * {{{
  * val (warmupKey, sampleKey) = key.split2()
  * val adapted = NUTS.adapt(latent, warmupKey, latent.init)
  * val chain = adapted.sample(sampleKey, 2000)
  * chain(Sigma)
  * }}}
  *
  * Several chains are run one per starting position, e.g. each in a `Future`,
  * as shown at [[Chain]]. For a `latent`, the warmup and the sampling loop are
  * compiled once, and shared by all chains.
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
    kernel(latent).sampler(stepSize, inverseMassMatrix)

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
    Kernel[P](PyBridge.toPyFn(logDensity)).sampler(stepSize, inverseMassMatrix)

  /** Runs Stan's window adaptation for the posterior of `latent`, starting
    * from `initial`: `numSteps` NUTS steps, during which the step size is
    * tuned by dual averaging towards `targetAcceptanceRate`, and the inverse
    * mass matrix is estimated from the draws in windows of growing length.
    *
    * The mass matrix is diagonal, or `dense` over all parameter elements. A
    * dense one captures correlations, but it needs more warmup steps to be
    * estimated well.
    *
    * The warmup is compiled on the first call for `latent`, and reused by
    * later ones with the same settings, e.g. for further chains; so is the
    * sampling loop of the adapted sampler.
    */
  def adapt(
      latent: Latent,
      key: Key,
      initial: latent.Params,
      numSteps: Int = 1000,
      dense: Boolean = false,
      targetAcceptanceRate: Double = 0.8
  ): NUTSAdaptation[latent.Params] =
    kernel(latent).adapt(key, initial, numSteps, dense, targetAcceptanceRate)

  /** Runs Stan's window adaptation for `logDensity`, for any parameter type.
    * See the other `adapt`. Each call compiles the warmup anew: for several
    * chains, prefer the `adapt` for a [[Latent]].
    */
  def adapt[P: TensorTree](
      key: Key,
      initial: P,
      numSteps: Int,
      dense: Boolean,
      targetAcceptanceRate: Double
  )(
      logDensity: P => Tensor0[Float32]
  ): NUTSAdaptation[P] =
    Kernel[P](PyBridge.toPyFn(logDensity)).adapt(key, initial, numSteps, dense, targetAcceptanceRate)

  /** The kernel of `latent`, made once and kept with it. */
  private def kernel(latent: Latent): Kernel[latent.Params] =
    latent
      .cached("nuts")(Kernel[latent.Params](PyBridge.toPyFn(latent.logDensity)))
      .asInstanceOf[Kernel[latent.Params]]

  /** NUTS for one log-density, with the compiled parts that all its samplers
    * share: the step and the sampling loop, which take the step size and the
    * inverse mass matrix from the state, and the warmups.
    */
  private final class Kernel[P: TensorTree](pyLogDensity: py.Dynamic):
    private val blackjax = py.module("blackjax")
    private val loops = SamplingAlgorithm.Loops()
    private val warmups = scala.collection.concurrent.TrieMap.empty[(Int, Boolean, Double), py.Dynamic]

    private val step = jit((key: Key, state: NUTSState[P]) =>
      val nuts = blackjax.nuts(
        pyLogDensity,
        TensorTree[Tensor0[Float32]].toPyTree(state.stepSize),
        state.pyInverseMass
      )
      val result = nuts.step(key.jaxKey, state.toPy)
      (
        NUTSState.fromPy[P](result.bracketAccess(0), state.stepSize, state.inverseMass),
        NUTSInfo.fromPy(result.bracketAccess(1))
      )
    )

    def sampler(
        stepSize: Double,
        inverseMassMatrix: InverseMassMatrix[P]
    ): SamplingAlgorithm[P, NUTSState[P], NUTSInfo] =
      sampler(Tensor0(stepSize.toFloat), p => flatMass(inverseMassMatrix.toPy(p)))

    /** The sampler with the given settings; the inverse mass matrix for a
      * position, in [[NUTSState]]'s representation.
      */
    private def sampler(
        stepSize: Tensor0[Float32],
        inverseMass: P => Tensor1[MassElement, Float32]
    ): SamplingAlgorithm[P, NUTSState[P], NUTSInfo] =
      SamplingAlgorithm.sharing(
        init = position =>
          NUTSState.fromPy[P](
            blackjax.mcmc.hmc.init(TensorTree[P].toPyTree(position), pyLogDensity),
            stepSize,
            inverseMass(position)
          ),
        step = step,
        loops = loops
      )

    def adapt(
        key: Key,
        initial: P,
        numSteps: Int,
        dense: Boolean,
        targetAcceptanceRate: Double
    ): NUTSAdaptation[P] =
      val warmup = warmups.getOrElseUpdate(
        (numSteps, dense, targetAcceptanceRate),
        compiledWarmup(numSteps, dense, targetAcceptanceRate)
      )
      val result = warmup(key.jaxKey, TensorTree[P].toPyTree(initial))
      val pyPosition = result.bracketAccess(0)
      val pyStepSize = result.bracketAccess(1)
      val pyInverseMass = result.bracketAccess(2)
      val inverseMassMatrix: InverseMassMatrix[P] =
        if dense then InverseMassMatrix.Dense(pyInverseMass)
        else
          val unravel =
            py.module("jax.flatten_util").ravel_pytree(pyPosition).bracketAccess(1)
          InverseMassMatrix.Diagonal(TensorTree[P].fromPyTree(unravel(pyInverseMass)))
      val mass = flatMass(pyInverseMass)
      NUTSAdaptation(
        sampler = sampler(TensorTree[Tensor0[Float32]].fromPyTree(pyStepSize), _ => mass),
        position = TensorTree[P].fromPyTree(pyPosition),
        stepSize = pyStepSize.item().as[Double],
        inverseMassMatrix = inverseMassMatrix,
        numDivergent = result.bracketAccess(3).item().as[Int]
      )

    /** The window adaptation, compiled: it takes a key and a starting
      * position, and returns the final position, the step size, the inverse
      * mass matrix and the number of divergences.
      */
    private def compiledWarmup(
        numSteps: Int,
        dense: Boolean,
        targetAcceptanceRate: Double
    ): py.Dynamic =
      val warmup = windowAdaptation(pyLogDensity, dense, targetAcceptanceRate)
      val runOne = (k: py.Dynamic, position: py.Dynamic) =>
        val result = warmup.run(k, position, numSteps)
        val adapted = result.bracketAccess(0)
        py.Dynamic.global.tuple(
          Seq(
            adapted.state.position,
            adapted.parameters.bracketAccess("step_size"),
            adapted.parameters.bracketAccess("inverse_mass_matrix"),
            result.bracketAccess(1).info.is_divergent.sum()
          ).toPythonProxy
        )
      py.module("jax").jit(dimwit.jax.Jax.jax_helper.wrap_fn(runOne))

    /** BlackJax's inverse mass matrix, flattened as in [[NUTSState]]. */
    private def flatMass(raw: py.Dynamic): Tensor1[MassElement, Float32] =
      TensorTree[Tensor1[MassElement, Float32]].fromPyTree(py.module("jax.numpy").ravel(raw))

  /** BlackJax's window adaptation of NUTS, which keeps only the divergences
    * of the warmup steps, not their states.
    */
  private def windowAdaptation(
      pyLogDensity: py.Dynamic,
      dense: Boolean,
      targetAcceptanceRate: Double
  ): py.Dynamic =
    val blackjax = py.module("blackjax")
    val infoFn = blackjax.adaptation.base.get_filter_adapt_info_fn(
      py.Dynamic.global.set(),
      py.Dynamic.global.set(Seq("is_divergent").toPythonProxy),
      py.Dynamic.global.set()
    )
    blackjax.window_adaptation(
      blackjax.nuts,
      pyLogDensity,
      is_mass_matrix_diagonal = !dense,
      target_acceptance_rate = targetAcceptanceRate,
      adaptation_info_fn = infoFn
    )
