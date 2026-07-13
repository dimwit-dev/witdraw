package witdraw
package mcmc

import dimwit.*
import dimwit.autodiff.TensorTree
import dimwit.python.PyBridge
import dimwit.random.Random
import me.shadaj.scalapy.py
import witdraw.mcmc.{KernelState, SamplingAlgorithm}

/** Kernel state wrapping a Python-side BlackJax state object.
  *
  * `position` is decoded on-demand via [[TensorTree.fromPyTree]]. `raw` is
  * package-private so the step closure can pass it directly to Python.
  */
case class PyKernelState[P: TensorTree](private[witdraw] val raw: py.Dynamic)
    extends KernelState[P]:
  def position: P = TensorTree[P].fromPyTree(raw.position)

/** Typed wrapper around a BlackJax NUTS info NamedTuple.
  *
  * Fields are decoded lazily from the Python object — only pay the crossing
  * cost for values you actually inspect.
  */
case class NUTSInfo(private val raw: py.Dynamic):
  lazy val acceptanceRate: Double = raw.acceptance_rate.item().as[Double]
  lazy val numIntegrationSteps: Int = raw.num_integration_steps.item().as[Int]
  lazy val isDivergent: Boolean = raw.is_divergent.item().as[Boolean]
  lazy val energy: Double = raw.energy.item().as[Double]

/** Generic factory for any BlackJax MCMC sampling algorithm.
  *
  * `build(pyLogProb, initPyTree)` constructs the BlackJax algo object once —
  * inside `SamplingAlgorithm.init`. The JAX-JIT-compiled step function is also
  * created at that point and reused for all subsequent steps.
  *
  * The mutable `algo` and `jittedStep` vars are captured in this closure only —
  * they are never visible on `SamplingAlgorithm`'s public surface. Parallel
  * independent chains should each be constructed via a separate
  * `BlackJaxAlgorithm.apply` call.
  *
  * Adding a new BlackJax algorithm is a single `build` lambda:
  * {{{
  * val hmcSampler = BlackJaxAlgorithm[Params, HMCInfo](
  *   build       = (logProb, initPyTree) => blackjax.hmc(logProb, stepSize, mass(initPyTree), numSteps),
  *   extractInfo = raw => HMCInfo(raw)
  * )(logDensity)
  * }}}
  */
object BlackJaxAlgorithm:
  def apply[P: TensorTree, I](
      build: (py.Dynamic, py.Any) => py.Dynamic,
      extractInfo: py.Dynamic => I
  )(
      logDensity: P => Tensor0[Float32]
  ): SamplingAlgorithm[P, PyKernelState[P], I] =
    val tt = TensorTree[P]
    val jax = py.module("jax")
    val pyLogProb = PyBridge.toPyFn(logDensity)

    var algo: py.Dynamic = null.asInstanceOf[py.Dynamic]
    var jittedStep: py.Dynamic = null.asInstanceOf[py.Dynamic]

    SamplingAlgorithm(
      init = position =>
        val initPyTree = tt.toPyTree(position)
        algo = build(pyLogProb, initPyTree)
        jittedStep = jax.jit(algo.step)
        PyKernelState(algo.init(initPyTree))
      ,
      step = (key, state) =>
        val result = jittedStep(key.jaxKey, state.raw)
        (
          PyKernelState(result.bracketAccess(0)),
          extractInfo(result.bracketAccess(1))
        )
    )

/** BlackJax NUTS sampler.
  *
  * Produces a [[SamplingAlgorithm]] directly — logdensity is baked in at
  * construction, matching how BlackJax's own algorithm objects work.
  *
  * {{{
  * val sampler = NUTSAlgorithm[Params](stepSize = 1e-3)(logDensity)
  * val chain   = sampler.run(init, key).drop(200).take(1000)
  *                 .map { case (pos, info) => pos }.toList
  *
  * // With a pre-conditioned diagonal mass matrix:
  * val sampler = NUTSAlgorithm[Params](stepSize = 1e-3, inverseMassMatrix = Some(myMass))(logDensity)
  * }}}
  *
  * @param stepSize
  *   Leapfrog integration step size.
  * @param inverseMassMatrix
  *   Optional 1D JAX array (one entry per parameter). Defaults to identity,
  *   inferred from the shape of the initial position.
  */
object NUTSAlgorithm:
  def apply[P: TensorTree](
      stepSize: Double,
      inverseMassMatrix: Option[py.Dynamic] = None
  )(
      logDensity: P => Tensor0[Float32]
  ): SamplingAlgorithm[P, PyKernelState[P], NUTSInfo] =
    val jax = py.module("jax")
    val blackjax = py.module("blackjax")
    BlackJaxAlgorithm[P, NUTSInfo](
      build = (logProb, initPyTree) =>
        val mass = inverseMassMatrix.getOrElse:
          val leaves = jax.tree_util.tree_leaves(initPyTree.as[py.Dynamic])
          val flat = jax.numpy.concatenate(
            jax.tree_util.tree_map(jax.numpy.ravel, leaves)
          )
          jax.numpy.ones_like(flat)
        blackjax.nuts(logProb, stepSize, mass)
      ,
      extractInfo = NUTSInfo(_)
    )(logDensity)
