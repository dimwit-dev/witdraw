package witdraw
package inference

import dimwit.*
import dimwit.autodiff.Autodiff
import dimwit.linalg.LinearAlgebra.inv
import dimwit.random.Random.Key
import dimwit.stats.MVNormal
import dimwit.tensortree.TensorTree

/** The axis of the flattened parameters of [[Laplace]]. */
private[witdraw] trait LaplaceDim derives Label

/** The state of [[Laplace]]: the last draw, and the Gaussian the draws come
  * from, over the flattened parameters.
  */
case class LaplaceState[P](
    position: P,
    private[witdraw] val mean: Tensor1[LaplaceDim, Float32],
    private[witdraw] val covariance: Tensor2[LaplaceDim, Prime[LaplaceDim], Float32]
) extends KernelState[P]

object LaplaceState:
  given [P: TensorTree]: TensorTree[LaplaceState[P]] = TensorTree.derived

/** The Laplace approximation, Rethinking's `quap`: a Gaussian centred at the
  * mode of the log-density, whose covariance is the inverse of the negative
  * Hessian there. Each step is an independent draw from it, so there is no
  * burn-in.
  * {{{
  * val laplace = Laplace(latent)
  * val draws = laplace.sample(latent.init, key, 2000).map(_._1)
  * draws.map(_(Sigma))
  * }}}
  *
  * `init` does the expensive part: it finds the mode, and computes and inverts
  * the Hessian there. Each step is a cheap Gaussian draw.
  */
object Laplace:
  /** The Laplace approximation of the posterior of `latent`. It approximates
    * [[Latent.logDensity]], so it lives in the unconstrained space and every
    * draw lies inside the constraints. `init` searches the mode from the
    * starting point it is given.
    */
  def apply(
      latent: Latent,
      maxSteps: Int = 10000,
      learningRate: Float = 0.05f
  ): SamplingAlgorithm[latent.Params, LaplaceState[latent.Params], Unit] =
    val atMode = apply[latent.Params](latent.logDensity)
    atMode.copy(init = start =>
      atMode.init(
        maximise("Laplace", latent.logDensity, start, maxSteps, learningRate)
      )
    )

  /** The Laplace approximation of `logDensity`, for any parameter type. `init`
    * must be given the mode of `logDensity`.
    */
  def apply[P: TensorTree](
      logDensity: P => Tensor0[Float32]
  ): SamplingAlgorithm[P, LaplaceState[P], Unit] =
    val step = jit((key: Key, state: LaplaceState[P]) =>
      val unflatten = TensorTree.ravel(state.position, Axis[LaplaceDim]).unflatten
      val draw = MVNormal(state.mean, state.covariance).sample(key)
      (state.copy(position = unflatten(draw)), ())
    )
    SamplingAlgorithm(
      init = mode =>
        val (flatten, unflatten) = TensorTree.ravel(mode, Axis[LaplaceDim])
        val hessian = Autodiff.hessian((v: Tensor1[LaplaceDim, Float32]) =>
          logDensity(unflatten(v))
        )(flatten(mode))
        val covariance = inv(hessian *! Tensor0(-1f))
        LaplaceState(mode, flatten(mode), covariance.transpose)
      ,
      step = step
    )
