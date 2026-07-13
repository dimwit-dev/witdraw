package witdraw.vi

import dimwit.stats.MVNormal
import dimwit.tensor.Label
import dimwit.Float32
import dimwit.autodiff.TensorTree
import dimwit.autodiff.Autodiff
import dimwit.tensor.Tensor0
import dimwit.*
import dimwit.random.Random.Key
import dimwit.linalg.LinearAlgebra.inv
import witdraw.mcmc.{KernelState, SamplingAlgorithm}

/** State for the QUAP sampling algorithm.
  *
  * The fitted Gaussian approximation is captured in `sampler`, a function from
  * a random key to a new i.i.d. draw. This keeps the internal flattened label
  * type private to [[QuapAlgorithm]].
  */
case class QuapState[P](
    position: P,
    private[witdraw] val sampler: Key => P
) extends KernelState[P]

/** Quadratic approximation (QUAP) as a [[SamplingAlgorithm]].
  *
  * Fits a multivariate Gaussian at the MAP by inverting the Hessian of the
  * log-density, then draws i.i.d. samples from that Gaussian. Because samples
  * are independent, there is no burn-in and no autocorrelation — `run` produces
  * an infinite stream of draws from the Laplace approximation.
  *
  * {{{
  * val sampler = QuapAlgorithm[Params](logDensity)
  * val samples = sampler.run(mapEstimate, Key(42)).take(1000).map(_._1).toList
  * }}}
  *
  * @note
  *   `init` does the expensive work: it computes and inverts the Hessian at the
  *   supplied starting point. Each subsequent `step` is a cheap Gaussian draw.
  */
object QuapAlgorithm:
  private trait QuapFlat derives Label

  def apply[P: TensorTree](
      logDensity: P => Tensor0[Float32]
  ): SamplingAlgorithm[P, QuapState[P], Unit] =
    SamplingAlgorithm(
      init = params =>
        val (flatten, unflatten) = TensorTree.ravel(params, Axis[QuapFlat])
        val hessF =
          Autodiff.hessian((v: Tensor1[QuapFlat, Float32]) =>
            logDensity(unflatten(v))
          )
        val h = hessF(flatten(params))
        val hinv = inv(h *! Tensor0(-1f))

        val dist = MVNormal(flatten(params), hinv.transpose)
        QuapState(params, key => unflatten(dist.sample(key)))
      ,
      step = (key, state) =>
        val next = state.sampler(key)
        (state.copy(position = next), ())
    )
