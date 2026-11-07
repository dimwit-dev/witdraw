package witdraw
package inference

import dimwit.*

/** The draws of one or more chains of a sampler, as from
  * [[SamplingAlgorithm.sampleChains]] or [[NUTSChainsAdaptation.sample]]: per
  * chain, a sequence of `(position, info)` pairs. There is at least one chain,
  * and all chains have the same number of draws, at least one.
  * {{{
  * val draws = adapted.sample(key, 1000)
  * draws(Sigma)                    // the values of Sigma, per chain
  * draws.values(p => p(Mu).sum)    // a value computed from each draw, per chain
  * draws.numDivergent              // the divergent transitions of NUTS
  * Diagnostics.rhat(draws, Sigma)
  * }}}
  */
final class Draws[P, I] private (val chains: Seq[Seq[(P, I)]]):

  def numChains: Int = chains.size

  /** The number of draws of each chain. */
  def numDraws: Int = chains.head.size

  /** The positions, per chain. */
  def positions: Seq[Seq[P]] = chains.map(_.map(_._1))

  /** The infos of the sampler, per chain. */
  def infos: Seq[Seq[I]] = chains.map(_.map(_._2))

  /** The positions of all chains, one chain after the other, e.g. to estimate
    * a posterior mean.
    */
  def all: Seq[P] = chains.flatMap(_.map(_._1))

  /** The value of `f` at each position, per chain. */
  def values[Q](f: P => Q): Seq[Seq[Q]] = chains.map(_.map(d => f(d._1)))

  /** The values of `site`, per chain, e.g. for [[Diagnostics.rhat]]. */
  def apply[T <: Tuple, V](site: SiteDef[T, V])(using reader: SiteReader[P]): Seq[Seq[Tensor[T, V]]] =
    values(p => reader.values(p)(site))

  /** Whether each draw was a divergent transition, per chain. */
  def divergent(using info: SamplerInfo[I]): Seq[Seq[Boolean]] =
    chains.map(_.map(d => info.isDivergent(d._2)))

  /** The number of divergent transitions, over all chains. */
  def numDivergent(using SamplerInfo[I]): Int = divergent.map(_.count(identity)).sum

  override def toString: String = s"Draws($numChains chains of $numDraws draws)"

object Draws:

  /** The draws of several chains, one sequence of `(position, info)` pairs per
    * chain.
    */
  def apply[P, I](chains: Seq[Seq[(P, I)]]): Draws[P, I] =
    require(chains.nonEmpty, "Draws needs at least one chain")
    val sizes = chains.map(_.size).distinct
    require(sizes.size == 1, s"all chains must have the same number of draws, but have ${sizes.mkString(", ")}")
    require(sizes.head > 0, "the chains have no draws")
    new Draws(chains)

  /** The draws of one chain, as from [[SamplingAlgorithm.sample]]. */
  def single[P, I](chain: Seq[(P, I)]): Draws[P, I] = apply(Seq(chain))

/** What a sampler reports about each of its draws, that generic code, such as
  * diagnostics and plots, can use.
  */
trait SamplerInfo[I]:
  /** Whether the draw was a divergent transition, which only Hamiltonian
    * samplers such as [[NUTS]] can detect.
    */
  def isDivergent(info: I): Boolean

object SamplerInfo:
  /** No info, as for [[Laplace]]: no draw is divergent. */
  given SamplerInfo[Unit] = _ => false
