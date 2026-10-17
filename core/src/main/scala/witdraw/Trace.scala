package witdraw

import dimwit.*
import scala.collection.mutable

/** What happened at one site during a run of a model. */
case class TraceEntry[Ten <: Tensor[?, ?]](
    site: Site[Ten],
    distribution: DistributionOf[Ten],
    value: Ten,
    observed: Boolean
):
  /** The log-density of the value under the site's distribution. */
  def logProb: Tensor0[Float32] =
    distribution
      .logProb(value.asInstanceOf[Tensor[ShapeOf[Ten], ElemOf[Ten]]])
      .asFloat

/** The sites of one run of a model, in the order in which they were sampled.
  *
  * {{{
  * val (result, tr) = ForwardSampler.run(key)(withTrace(model(x)))
  * tr.latentSites   // the sites that were not observed
  * tr(Sigma)        // the value of Sigma in this run, a Sigma.Value
  * }}}
  */
final class Trace(val entries: Seq[TraceEntry[?]]):
  private val bySite: Map[Site[?], TraceEntry[?]] =
    entries.map(e => e.site -> e).toMap

  def sites: Seq[Site[?]] = entries.map(_.site)
  def latentSites: Seq[Site[?]] = entries.filterNot(_.observed).map(_.site)
  def observedSites: Seq[Site[?]] = entries.filter(_.observed).map(_.site)

  def contains(site: Site[?]): Boolean = bySite.contains(site)

  /** Everything recorded for `site`. */
  def entry[Ten <: Tensor[?, ?]](site: Site[Ten]): TraceEntry[Ten] =
    bySite.get(site) match
      case Some(e) => e.asInstanceOf[TraceEntry[Ten]]
      case None    =>
        throw java.util.NoSuchElementException(
          s"Site '${site.name}' is not in the trace. Sites: ${sites.map(_.name).mkString(", ")}"
        )

  /** The value of `site` in this run. */
  def apply[Ten <: Tensor[?, ?]](site: Site[Ten]): Ten = entry(site).value

  override def toString: String =
    entries
      .map(e => s"${e.site.name}${if e.observed then " (observed)" else ""}")
      .mkString("Trace(", ", ", ")")

/** Runs `model` and records every site it samples, together with its
  * distribution, value and whether it was observed. Returns the model's result
  * and the [[Trace]].
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
def withTrace[A](model: ProbProg[A])(using ctx: ProbContext): (A, Trace) =
  val entries = mutable.ArrayBuffer.empty[TraceEntry[?]]
  val result = model(using
    new Handler(ctx):
      override def sample[Ten <: Tensor[?, ?]](
          statement: SampleStatement[Ten]
      ): Ten =
        val value = outer.sample(statement)
        entries += TraceEntry(
          statement.site,
          statement.distribution,
          value,
          observed = statement.observed.isDefined
        )
        value
  )
  (result, Trace(entries.toSeq))
