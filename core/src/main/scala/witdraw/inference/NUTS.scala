package witdraw
package inference

import dimwit.*
import dimwit.python.PyBridge
import dimwit.random.Random.Key
import dimwit.tensortree.TensorTree
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

/** The state of [[NUTS]]: the position, and the log-density and its gradient
  * there, as in BlackJax's `HMCState`.
  */
case class NUTSState[P](
    position: P,
    logDensity: Tensor0[Float32],
    logDensityGrad: P
) extends KernelState[P]:
  private[witdraw] def toPy(using tree: TensorTree[P]): py.Dynamic =
    py.module("blackjax.mcmc.hmc")
      .HMCState(
        tree.toPyTree(position),
        TensorTree[Tensor0[Float32]].toPyTree(logDensity),
        tree.toPyTree(logDensityGrad)
      )

object NUTSState:
  given [P: TensorTree]: TensorTree[NUTSState[P]] = TensorTree.derived

  private[witdraw] def fromPy[P](raw: py.Dynamic)(using
      tree: TensorTree[P]
  ): NUTSState[P] =
    NUTSState(
      tree.fromPyTree(raw.position),
      TensorTree[Tensor0[Float32]].fromPyTree(raw.logdensity),
      tree.fromPyTree(raw.logdensity_grad)
    )

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
  * which to start it.
  */
case class NUTSAdaptation[P](
    sampler: SamplingAlgorithm[P, NUTSState[P], NUTSInfo],
    position: P,
    stepSize: Double,
    inverseMassMatrix: InverseMassMatrix[P],
    numDivergent: Int
)

/** The window adaptation of several chains, from [[NUTS.chainsWarmup]]. It is
  * compiled on the first [[run]], and later runs reuse it, as long as they
  * have as many chains; so do the samplers of the adaptations it returns.
  */
final class NUTSChainsWarmup[P: TensorTree] private[inference] (
    pyLogDensity: py.Dynamic,
    numSteps: Int,
    dense: Boolean,
    targetAcceptanceRate: Double
):
  private lazy val warmup =
    NUTS.compiledWarmup(pyLogDensity, numSteps, dense, targetAcceptanceRate)
  private lazy val loop = NUTS.chainsLoop[P](pyLogDensity)

  /** Runs the warmup for one chain per starting position in `initials`. */
  def run(initials: Seq[P], key: Key): NUTSChainsAdaptation[P] =
    NUTS.runChainsWarmup(initials, key, dense, pyLogDensity, warmup, loop)

/** The result of [[NUTS.adaptChains]]: one [[NUTSAdaptation]] per chain, each
  * with its own step size and inverse mass matrix, as in Stan. [[sample]]
  * continues all chains from the ends of their warmups.
  */
final class NUTSChainsAdaptation[P: TensorTree] private[witdraw] (
    val chains: Seq[NUTSAdaptation[P]],
    loop: py.Dynamic,
    settings: Seq[py.Any]
):

  /** `numDraws` draws of each chain, after its warmup, in one compiled loop.
    * The loop is compiled on the first call, and shared by all adaptations
    * from the same [[NUTSChainsWarmup]].
    */
  def sample(key: Key, numDraws: Int): Seq[Seq[(P, NUTSInfo)]] =
    SamplingAlgorithm.sampleChains(loop, chains.map(_.position), settings, key, numDraws)

  /** The divergent transitions during the warmup, summed over the chains. */
  def numDivergent: Int = chains.map(_.numDivergent).sum

/** The No-U-Turn Sampler, a variant of Hamiltonian Monte Carlo that chooses the
  * length of each trajectory itself, run by BlackJax.
  *
  * [[NUTS.adapt]] runs a warmup that tunes the step size and the inverse mass
  * matrix, and returns a sampler that uses them:
  * {{{
  * val (warmupKey, sampleKey) = key.split2()
  * val adapted = NUTS.adapt(latent, latent.init, warmupKey)
  * val draws = adapted.sampler
  *   .sample(adapted.position, sampleKey, 2000)
  *   .map(_._1)
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
    sampler(PyBridge.toPyFn(logDensity), py.Any.from(stepSize), inverseMassMatrix.toPy)

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
    val pyLogDensity = PyBridge.toPyFn(logDensity)
    val warmup = windowAdaptation(pyLogDensity, dense, targetAcceptanceRate)
    val result = warmup.run(key.jaxKey, TensorTree[P].toPyTree(initial), numSteps)
    val adapted = result.bracketAccess(0)
    adaptation[P](
      pyLogDensity,
      adapted.state.position,
      adapted.parameters.bracketAccess("step_size"),
      adapted.parameters.bracketAccess("inverse_mass_matrix"),
      dense,
      result.bracketAccess(1).info.is_divergent.sum()
    )

  /** Runs Stan's window adaptation for the posterior of `latent` for several
    * chains, one per starting position in `initials`, side by side in one
    * compiled loop. Each chain is tuned on its own, so that differences
    * between them show up in diagnostics such as [[Diagnostics.rhat]]. See
    * [[adapt]] for the parameters.
    *
    * {{{
    * val Seq(initKey, warmupKey, sampleKey) = key.split(3)
    * val starts = initKey.split(4).map(latent.initRandom(_))
    * val adapted = NUTS.adaptChains(latent, starts, warmupKey)
    * val chains = adapted.sample(sampleKey, 1000).map(_.map(_._1))
    * Diagnostics.rhat(chains.map(_.map(_(Sigma))))
    * }}}
    *
    * Each call compiles the warmup anew. To run it several times, e.g. for
    * other starting positions, create it once with [[chainsWarmup]].
    *
    * The chains should start from different positions, e.g. from
    * [[Latent.initRandom]]: from one position, they only differ by their
    * keys, which makes problems harder to detect.
    */
  def adaptChains(
      latent: Latent,
      initials: Seq[latent.Params],
      key: Key,
      numSteps: Int = 1000,
      dense: Boolean = false,
      targetAcceptanceRate: Double = 0.8
  ): NUTSChainsAdaptation[latent.Params] =
    chainsWarmup(latent, numSteps, dense, targetAcceptanceRate).run(initials, key)

  /** Runs Stan's window adaptation for several chains of `logDensity`, for
    * any parameter type. See the other `adaptChains`.
    */
  def adaptChains[P: TensorTree](
      initials: Seq[P],
      key: Key,
      numSteps: Int,
      dense: Boolean,
      targetAcceptanceRate: Double
  )(
      logDensity: P => Tensor0[Float32]
  ): NUTSChainsAdaptation[P] =
    chainsWarmup[P](numSteps, dense, targetAcceptanceRate)(logDensity).run(initials, key)

  /** The window adaptation of [[adaptChains]] for the posterior of `latent`,
    * to run several times: it is compiled once, on its first run.
    * {{{
    * val warmup = NUTS.chainsWarmup(latent)
    * val adapted = warmup.run(starts, warmupKey)
    * }}}
    */
  def chainsWarmup(
      latent: Latent,
      numSteps: Int = 1000,
      dense: Boolean = false,
      targetAcceptanceRate: Double = 0.8
  ): NUTSChainsWarmup[latent.Params] =
    chainsWarmup[latent.Params](numSteps, dense, targetAcceptanceRate)(latent.logDensity)

  /** The window adaptation of [[adaptChains]] for `logDensity`, for any
    * parameter type. See the other `chainsWarmup`.
    */
  def chainsWarmup[P: TensorTree](
      numSteps: Int,
      dense: Boolean,
      targetAcceptanceRate: Double
  )(
      logDensity: P => Tensor0[Float32]
  ): NUTSChainsWarmup[P] =
    NUTSChainsWarmup(PyBridge.toPyFn(logDensity), numSteps, dense, targetAcceptanceRate)

  /** Runs `warmup`, from [[compiledWarmup]], for one chain per starting
    * position; the adapted chains sample with `loop`, from [[chainsLoop]].
    */
  private[inference] def runChainsWarmup[P: TensorTree](
      initials: Seq[P],
      key: Key,
      dense: Boolean,
      pyLogDensity: py.Dynamic,
      warmup: py.Dynamic,
      loop: py.Dynamic
  ): NUTSChainsAdaptation[P] =
    require(initials.nonEmpty, "the warmup needs at least one starting position")
    val keys = py.module("jax.random").split(key.jaxKey, initials.size)
    val starts = SamplingAlgorithm.stack(initials.map(TensorTree[P].toPyTree(_)))
    val results = warmup(keys, starts)
    val itemgetter = py.module("operator").itemgetter
    val perChain = initials.indices.map: c =>
      val row = py.module("jax.tree_util").tree_map(itemgetter(c), results)
      (row.bracketAccess(0), row.bracketAccess(1), row.bracketAccess(2), row.bracketAccess(3))
    val chains = perChain.map: (position, stepSize, inverseMass, divergent) =>
      adaptation[P](pyLogDensity, position, stepSize, inverseMass, dense, divergent)
    val settings = perChain.map: (_, stepSize, inverseMass, _) =>
      py.Dynamic.global.tuple(Seq(stepSize, inverseMass).toPythonProxy): py.Any
    NUTSChainsAdaptation(chains, loop, settings)

  /** The window adaptation of several chains, side by side and compiled: it
    * takes the keys and the starting positions of the chains, stacked, and
    * returns their positions, step sizes, inverse mass matrices and numbers
    * of divergences, stacked.
    */
  private[inference] def compiledWarmup(
      pyLogDensity: py.Dynamic,
      numSteps: Int,
      dense: Boolean,
      targetAcceptanceRate: Double
  ): py.Dynamic =
    val jax = py.module("jax")
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
    jax.jit(jax.vmap(dimwit.jax.Jax.jax_helper.wrap_fn(runOne)))

  /** The compiled sampling loop of adapted chains, whose settings are their
    * step sizes and inverse mass matrices.
    */
  private[inference] def chainsLoop[P: TensorTree](pyLogDensity: py.Dynamic): py.Dynamic =
    SamplingAlgorithm.chainsLoop[P, NUTSState[P], NUTSInfo]: s =>
      sampler[P](pyLogDensity, s.bracketAccess(0), _ => s.bracketAccess(1))

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

  /** The [[NUTSAdaptation]] for the results of a warmup, given as BlackJax
    * returns them.
    */
  private def adaptation[P: TensorTree](
      pyLogDensity: py.Dynamic,
      pyPosition: py.Dynamic,
      pyStepSize: py.Dynamic,
      pyInverseMass: py.Dynamic,
      dense: Boolean,
      pyNumDivergent: py.Dynamic
  ): NUTSAdaptation[P] =
    val inverseMassMatrix: InverseMassMatrix[P] =
      if dense then InverseMassMatrix.Dense(pyInverseMass)
      else
        val unravel =
          py.module("jax.flatten_util").ravel_pytree(pyPosition).bracketAccess(1)
        InverseMassMatrix.Diagonal(TensorTree[P].fromPyTree(unravel(pyInverseMass)))
    NUTSAdaptation(
      sampler = sampler(pyLogDensity, pyStepSize, inverseMassMatrix.toPy),
      position = TensorTree[P].fromPyTree(pyPosition),
      stepSize = pyStepSize.item().as[Double],
      inverseMassMatrix = inverseMassMatrix,
      numDivergent = pyNumDivergent.item().as[Int]
    )

  /** The NUTS sampler for `pyLogDensity`, with the inverse mass matrix in
    * BlackJax's representation for a position. Its step is compiled, and it is
    * traceable, so [[SamplingAlgorithm.sample]] can run it in a loop.
    */
  private[witdraw] def sampler[P: TensorTree](
      pyLogDensity: py.Dynamic,
      stepSize: py.Any,
      inverseMassMatrix: P => py.Dynamic
  ): SamplingAlgorithm[P, NUTSState[P], NUTSInfo] =
    val blackjax = py.module("blackjax")
    val step = jit((key: Key, state: NUTSState[P]) =>
      val nuts = blackjax.nuts(
        pyLogDensity,
        stepSize,
        inverseMassMatrix(state.position)
      )
      val result = nuts.step(key.jaxKey, state.toPy)
      (
        NUTSState.fromPy[P](result.bracketAccess(0)),
        NUTSInfo.fromPy(result.bracketAccess(1))
      )
    )
    SamplingAlgorithm(
      init = position =>
        NUTSState.fromPy[P](
          blackjax.mcmc.hmc.init(TensorTree[P].toPyTree(position), pyLogDensity)
        ),
      step = step
    )
