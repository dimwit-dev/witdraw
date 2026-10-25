package witdraw
package inference

import dimwit.*
import dimwit.tensor.Labels
import dimwit.tensortree.TensorTree
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

/** Convergence diagnostics for the draws of several chains, as from
  * [[SamplingAlgorithm.sampleChains]] or [[NUTSChainsAdaptation.sample]].
  *
  * Each takes the draws of one quantity, e.g. a site, per chain, and returns
  * one value per element of the quantity:
  * {{{
  * val chains = adapted.sample(key, 1000).map(_.map(_._1))
  * val rhat = Diagnostics.rhat(chains.map(_.map(_(Beta))))   // like Beta
  * val ess = Diagnostics.ess(chains.map(_.map(_(Beta))))
  * }}}
  */
object Diagnostics:

  /** The split R-hat (Gelman et al., as in Stan): each chain is split into
    * halves, and the variance between the halves is compared with the
    * variance within them. Values close to 1 indicate that the chains agree;
    * above about 1.01, they have not converged, or need more draws.
    *
    * Splitting the chains detects a chain that still drifts, so this works
    * for one chain too. Each chain needs at least 4 draws; with an odd number,
    * the last draw is left out.
    */
  def rhat[T <: Tuple: Labels](
      chains: Seq[Seq[Tensor[T, Float32]]]
  ): Tensor[T, Float32] =
    val draws = stacked(chains)
    val numDraws = chains.head.size
    require(numDraws >= 4, s"rhat needs at least 4 draws per chain, but got $numDraws")
    val half = numDraws / 2
    val elementShape = shapeOf(draws).drop(2)
    val firstDraws = py.Dynamic.global.tuple(
      Seq(
        py.Dynamic.global.slice(py.None),
        py.Dynamic.global.slice(2 * half)
      ).toPythonProxy
    )
    val split = draws
      .bracketAccess(firstDraws)
      .reshape(tuple(Seq(2 * chains.size, half) ++ elementShape))
    val value = py.module("blackjax.diagnostics").potential_scale_reduction(split)
    result(value, elementShape)

  /** The effective sample size, over all chains (Geyer's initial monotone
    * sequence, as in Stan): about how many independent draws the correlated
    * draws are worth, for estimating the mean.
    */
  def ess[T <: Tuple: Labels](
      chains: Seq[Seq[Tensor[T, Float32]]]
  ): Tensor[T, Float32] =
    val draws = stacked(chains)
    val numDraws = chains.head.size
    require(numDraws >= 2, s"ess needs at least 2 draws per chain, but got $numDraws")
    val value = py.module("blackjax.diagnostics").effective_sample_size(draws)
    result(value, shapeOf(draws).drop(2))

  /** The draws as one array, with a chain and a draw axis in front. */
  private def stacked[T <: Tuple: Labels](
      chains: Seq[Seq[Tensor[T, Float32]]]
  ): py.Dynamic =
    require(chains.nonEmpty, "no chains given")
    val sizes = chains.map(_.size).distinct
    require(sizes.size == 1, s"all chains must have the same number of draws, but have ${sizes.mkString(", ")}")
    val jnp = py.module("jax.numpy")
    val tree = TensorTree[Tensor[T, Float32]]
    jnp.stack(chains.map(c => jnp.stack(c.map(tree.toPyTree(_)).toPythonProxy)).toPythonProxy)

  /** The diagnostic with the shape of one element, which BlackJax may have
    * squeezed.
    */
  private def result[T <: Tuple: Labels](value: py.Dynamic, shape: Seq[Int]): Tensor[T, Float32] =
    val jnp = py.module("jax.numpy")
    TensorTree[Tensor[T, Float32]].fromPyTree(
      jnp.reshape(value, tuple(shape)).astype(jnp.float32)
    )

  private def shapeOf(array: py.Dynamic): Seq[Int] =
    py.Dynamic.global.list(array.shape).as[Seq[Int]]

  private def tuple(values: Seq[Int]): py.Dynamic =
    py.Dynamic.global.tuple(values.toPythonProxy)
