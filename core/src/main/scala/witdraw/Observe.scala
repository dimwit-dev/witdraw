package witdraw

import dimwit.*
import scala.collection.mutable
import scala.compiletime.constValueTuple
import scala.deriving.Mirror

/** Runs `model` with the given sites fixed to observed values.
  *
  * {{{
  * ForwardSampler.run(key)(observe(Sigma := Tensor0(1f))(model(x)))
  *
  * ModelScorer.run:
  *   observe(Y := yData, Sigma := sigma):
  *     model(x)
  * }}}
  *
  * Observing a site that the model never samples is an error, as is observing a
  * site that a handler further in has already observed.
  */
def observe[A](observations: Observation[?]*)(model: ProbProg[A])(using
    outer: ProbContext
): A =
  val bySite: Map[Site[?], Any] =
    observations.map(o => o.site -> o.value).toMap
  require(
    bySite.size == observations.size,
    s"A site is observed more than once in the same observe: ${observations.map(_.site.name).mkString(", ")}"
  )
  val observer = Observer(outer, bySite.get)
  val result = model(using observer)
  val unmatched = bySite.keySet -- observer.sampled
  require(
    unmatched.isEmpty,
    s"Observed ${quoted(unmatched.map(_.name))}, but the model has no such site. " +
      s"Sites in the model: ${observer.sampled.map(_.name).mkString(", ")}."
  )
  result

/** Runs `model` with every field of `values` (a named tuple or case class)
  * observed. A field observes the site of the same name, compared with the
  * first letter lower-cased: field `sigma` observes `object Sigma`.
  *
  * {{{
  * type Params = (alpha: Alpha.Value, sigma: Sigma.Value)
  * val logDensity = (params: Params) =>
  *   ModelScorer.run:
  *     observe(Y := yData):
  *       observeAll(params):
  *         model(x)
  * }}}
  *
  * A field that matches no site of the model is an error, so a misspelled name
  * cannot silently leave a site free.
  */
inline def observeAll[P, A](values: P)(model: ProbProg[A])(using
    m: Mirror.ProductOf[P],
    outer: ProbContext
): A =
  val names = constValueTuple[m.MirroredElemLabels].toList.map(_.toString)
  val fields = values.asInstanceOf[Product].productIterator.toList
  observeByName(names.zip(fields).toMap)(model)

private[witdraw] def observeByName[A](byName: Map[String, Any])(
    model: ProbProg[A]
)(using outer: ProbContext): A =
  def fieldName(site: Site[?]) = site.name.updated(0, site.name(0).toLower)
  val observer = Observer(outer, site => byName.get(fieldName(site)))
  val result = model(using observer)
  val unmatched = byName.keySet -- observer.sampled.map(fieldName)
  require(
    unmatched.isEmpty,
    s"Observed ${quoted(unmatched)}, but the model has no site of that name. " +
      s"Sites in the model: ${observer.sampled.map(_.name).mkString(", ")}. " +
      "Names are matched with the first letter lower-cased (object Sigma <-> field sigma)."
  )
  result

/** The handler behind [[observe]] and [[observeAll]]: fixes every site for
  * which `valueOf` has a value, and records which sites the model sampled.
  */
private class Observer(outer: ProbContext, valueOf: Site[?] => Option[Any])
    extends Handler(outer):
  val sampled = mutable.LinkedHashSet.empty[Site[?]]

  override def sample[Ten <: Tensor[?, ?]](
      statement: SampleStatement[Ten]
  ): Ten =
    sampled += statement.site
    valueOf(statement.site) match
      case None        => outer.sample(statement)
      case Some(value) =>
        require(
          statement.observed.isEmpty,
          s"Site '${statement.site.name}' is observed twice."
        )
        outer.sample(statement.copy(observed = Some(value.asInstanceOf[Ten])))

private def quoted(names: Iterable[String]): String =
  names.map(n => s"'$n'").mkString(", ")
