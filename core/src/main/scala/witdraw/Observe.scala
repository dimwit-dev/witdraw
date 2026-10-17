package witdraw

import dimwit.*
import scala.collection.mutable

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
def observe[A](observations: Observation[?, ?]*)(model: ProbProg[A])(using
    outer: ProbContext
): A =
  val bySite: Map[SiteDef[?, ?], Any] =
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

/** The handler behind [[observe]]: fixes every site for which `valueOf` has a
  * value, and records which sites the model sampled.
  */
private class Observer(outer: ProbContext, valueOf: SiteDef[?, ?] => Option[Any])
    extends Handler(outer):
  val sampled = mutable.LinkedHashSet.empty[SiteDef[?, ?]]

  override def sample[T <: Tuple, V](
      statement: SampleStatement[T, V]
  ): Tensor[T, V] =
    sampled += statement.site
    valueOf(statement.site) match
      case None        => outer.sample(statement)
      case Some(value) =>
        require(
          statement.observed.isEmpty,
          s"Site '${statement.site.name}' is observed twice."
        )
        outer.sample(statement.copy(observed = Some(value.asInstanceOf[Tensor[T, V]])))

private def quoted(names: Iterable[String]): String =
  names.map(n => s"'$n'").mkString(", ")
