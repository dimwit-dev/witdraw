package witdraw.examples.stan

import dimwit.*
import witdraw.reparam.*

/** Wiring [[witdraw.reparam.Reparam]] to the [[Mixture]] model: `theta` on the
  * simplex, `mu` increasing, `sigma` positive — the three constrained
  * parameter kinds, mapped to unconstrained space for gradient-based MCMC.
  * The model itself is untouched.
  */
object ReparamExample:

  trait ThetaZ derives Label // K-1 axis for the stick-breaking simplex

  /** The unconstrained companion of `Mixture.Posterior`: identical except
    * `theta`, whose stick-breaking chart has K-1 coordinates.
    */
  type PosteriorZ = (
      theta: Tensor1[ThetaZ, Float32],
      mu: Tensor1[Mixture.Comp, Float32],
      sigma: Tensor1[Mixture.Comp, Float32]
  )

  @main
  def reparamDemo(): Unit =
    dimwit.initialize()
    val k = 3
    val cAxis = Axis[Mixture.Comp]

    val bijector = Reparam.build[Mixture.Posterior, PosteriorZ](
      (
        theta = Bijector.simplexStickBreaking[ThetaZ, Mixture.Comp](k),
        mu = Bijector.ordered[Mixture.Comp](k),
        sigma = Bijector.exp[Mixture.Comp *: EmptyTuple]
      )
    )

    val params: Mixture.Posterior = (
      theta = Tensor1(cAxis -> k).fromArray(Array(0.5f, 0.3f, 0.2f)),
      mu = Tensor1(cAxis -> k).fromArray(Array(-2f, 0f, 3f)),
      sigma = Tensor1(cAxis -> k).fromArray(Array(0.5f, 1f, 0.8f))
    )
    val y = Mixture.simulate(params, n = 40)(Key(3L))

    // constrained-space model  ->  unconstrained-space target for NUTS
    val target: PosteriorZ => Tensor0[Float32] =
      Reparam.pushForward(bijector)(Mixture.logDensity(y))

    val z0 = bijector.inverse(params) // start point in unconstrained space
    println(f"target at z0            = ${target(z0).item}%.4f")

    // round-trips through the chart
    val back = bijector.forward(z0)
    def maxAbs(a: Array[Float], b: Array[Float]) =
      a.zip(b).map((x, y) => math.abs(x - y)).max
    println(f"round-trip theta max|Δ| = ${maxAbs(params.theta.toArray, back.theta.toArray)}%.2e")
    println(f"round-trip mu    max|Δ| = ${maxAbs(params.mu.toArray, back.mu.toArray)}%.2e")
    println(f"round-trip sigma max|Δ| = ${maxAbs(params.sigma.toArray, back.sigma.toArray)}%.2e")

    // a NUTS run would then be:
    //   val nuts   = NUTSAlgorithm[PosteriorZ](stepSize = 0.05)(target)
    //   val drawsZ = nuts.run(z0, Key(0)).drop(burn).take(n).map(_._1).toList
    //   val draws  = drawsZ.map(bijector.forward)   // report in constrained space
