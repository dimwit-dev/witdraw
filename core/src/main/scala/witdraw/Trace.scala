package witdraw

import dimwit.*
import dimwit.stats.Distribution
import scala.collection.mutable

/** What happened at one site during a run of a model. */
case class TraceEntry[T <: Tuple, V](
    site: SiteDef[T, V],
    distribution: Distribution[T, V],
    value: Tensor[T, V],
    observed: Boolean
):
  /** The log-density of the value under the site's distribution. */
  def logProb: Tensor0[Float32] = distribution.logProb(value).asFloat

/** The sites of one run of a model, in the order in which they were sampled.
  *
  * {{{
  * val (result, tr) = ForwardSampler.run(key)(withTrace(model(x)))
  * tr.latentSites   // the sites that were not observed
  * tr(Sigma)        // the value of Sigma in this run, a Sigma.Value
  * }}}
  */
final class Trace(val entries: Seq[TraceEntry[?, ?]]):
  private val bySite: Map[SiteDef[?, ?], TraceEntry[?, ?]] =
    entries.map(e => e.site -> e).toMap

  def sites: Seq[SiteDef[?, ?]] = entries.map(_.site)
  def latentSites: Seq[SiteDef[?, ?]] =
    entries.filterNot(_.observed).map(_.site)
  def observedSites: Seq[SiteDef[?, ?]] = entries.filter(_.observed).map(_.site)

  /** The value of `site` in this run. */
  def apply[T <: Tuple, V](site: SiteDef[T, V]): Tensor[T, V] =
    bySite.get(site) match
      case Some(e) => e.value.asInstanceOf[Tensor[T, V]]
      case None    =>
        throw java.util.NoSuchElementException(
          s"Site '${site.name}' is not in the trace. Sites: ${sites.map(_.name).mkString(", ")}"
        )

  override def toString: String =
    entries
      .map(e => s"${e.site.name}${if e.observed then " (observed)" else ""}")
      .mkString("Trace(", ", ", ")")

/** Runs `model` and records every site it samples, together with its
  * distribution, value and whether it was observed. Returns the model's result
  * and the [[Trace]].
  *
  * A model that samples a site more than once is an error.
  *
  * Like every handler, `withTrace` sees the statements of its block after the
  * handlers further in have changed them, so put it around `observe` to record
  * which sites are observed:
  * {{{
  * ForwardSampler.run(key):
  *   withTrace:
  *     observe(Y := yData)(model(x))
  * }}}
  */
def withTrace[A](model: ProbProg[A])(using outer: ProbContext): (A, Trace) =
  val entries = mutable.LinkedHashMap.empty[SiteDef[?, ?], TraceEntry[?, ?]]
  val result = model(using
    new ProbContext:
      def sample[T <: Tuple, V](
          statement: SampleStatement[T, V]
      ): Tensor[T, V] =
        if entries.contains(statement.site) then
          throw sampledTwice(statement.site)
        val value = outer.sample(statement)
        entries(statement.site) = TraceEntry(
          statement.site,
          statement.distribution,
          value,
          observed = statement.observed.isDefined
        )
        value
  )
  (result, Trace(entries.values.toSeq))
