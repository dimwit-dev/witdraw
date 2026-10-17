package witdraw
package inference

import dimwit.*
import dimwit.autodiff.Autodiff
import dimwit.linalg.LinearAlgebra.inv
import dimwit.random.Random.Key
import dimwit.stats.MVNormal
import dimwit.tensortree.TensorTree

/** The state of [[Laplace]]: the last draw, and how to make the next
  * one.
  */
case class LaplaceState[P](
    position: P,
    private[witdraw] val draw: Key => P
) extends KernelState[P]

/** The Laplace approximation, Rethinking's `quap`: a Gaussian centred at the
  * mode of the log-density, whose covariance is the inverse of the negative
  * Hessian there. Each step is an independent draw from it, so there is no
  * burn-in.
  * {{{
  * val laplace = Laplace(latent)
  * val draws = laplace.run(latent.init, key).take(2000).map(_._1).toList
  * draws.map(_(Sigma))
  * }}}
  *
  * `init` does the expensive part: it finds the mode, and computes and inverts
  * the Hessian there. Each step is a cheap Gaussian draw.
  */
object Laplace:
  private trait LaplaceDim derives Label

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
    SamplingAlgorithm(
      init = mode =>
        val (flatten, unflatten) = TensorTree.ravel(mode, Axis[LaplaceDim])
        val hessian = Autodiff.hessian((v: Tensor1[LaplaceDim, Float32]) =>
          logDensity(unflatten(v))
        )(flatten(mode))
        val covariance = inv(hessian *! Tensor0(-1f))
        val gaussian = MVNormal(flatten(mode), covariance.transpose)
        LaplaceState(mode, key => unflatten(gaussian.sample(key)))
      ,
      step = (key, state) =>
        (state.copy(position = state.draw(key)), ())
    )
