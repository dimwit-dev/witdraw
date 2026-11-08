package witdraw
package inference

import dimwit.*

/** The draws of one chain of a sampler, as from [[SamplingAlgorithm.sample]]:
  * a sequence of `(position, info)` pairs.
  * {{{
  * val chain = adapted.sample(key, 1000)
  * chain(Sigma)                    // the values of Sigma
  * chain.values(p => p(Mu).sum)    // a value computed from each draw
  * chain.numDivergent              // the divergent transitions of NUTS
  * }}}
  *
  * Several chains are a `Seq[Chain]`, e.g. for [[Diagnostics.rhat]]. A chain
  * is ordinary code, e.g. a warmup followed by sampling, so to run several at
  * the same time, run each in a `Future`:
  * {{{
  * def chain(key: Key, start: latent.Params) =
  *   val (warmupKey, sampleKey) = key.split2()
  *   val adapted = NUTS.adapt(latent, warmupKey, start)
  *   adapted.sample(sampleKey, 1000)
  *
  * val runs = keys.zip(starts).map((key, start) => Future(blocking(chain(key, start))))
  * val chains = Await.result(Future.sequence(runs), Duration.Inf)
  * Diagnostics.rhat(chains, Sigma)
  * }}}
  * A chain mostly waits for compiled JAX code, and `blocking` lets the global
  * pool give each chain a thread of its own. Chains on the CPU then run at the
  * same time; on one GPU, JAX runs them one after the other.
  */
final class Chain[P, I](val draws: Seq[(P, I)]):

  /** The number of draws. */
  def size: Int = draws.size

  def positions: Seq[P] = draws.map(_._1)

  /** What the sampler reported about each draw. */
  def infos: Seq[I] = draws.map(_._2)

  /** The value of `f` at each position. */
  def values[Q](f: P => Q): Seq[Q] = positions.map(f)

  /** The values of `site`. */
  def apply[T <: Tuple, V](site: SiteDef[T, V])(using reader: SiteReader[P]): Seq[Tensor[T, V]] =
    values(p => reader.values(p)(site))

  /** Whether each draw was a divergent transition. */
  def divergent(using info: SamplerInfo[I]): Seq[Boolean] = infos.map(info.isDivergent)

  def numDivergent(using SamplerInfo[I]): Int = divergent.count(identity)

  override def toString: String = s"Chain($size draws)"

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
