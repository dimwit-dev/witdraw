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
  * It is an error to observe a site twice, to observe a site the model never
  * samples, or to run a model that samples a site more than once.
  */
def observe[A](observations: Observation[?, ?]*)(model: ProbProg[A])(using
    ProbContext
): A =
  intercept("observed", "observe", observations.map(o => o.site -> o.value))(
    [T <: Tuple, V] =>
      (statement: SampleStatement[T, V], value: Tensor[?, ?]) =>
        if statement.observed.isDefined then
          throw IllegalArgumentException(
            s"Site '${statement.site.name}' is observed twice, by nested observe blocks."
          )
        statement.copy(observed = Some(value.asInstanceOf[Tensor[T, V]]))
  )(model)

/** Runs `model`, and changes the statement of each site in `sites` with
  * `change`, e.g. to observe it, before passing it on. `what` and `handler`
  * name the change and the handler in errors: it is an error to give a site
  * twice, to give a site the model never samples, or to run a model that
  * samples a site more than once.
  */
private[witdraw] def intercept[A, X](
    what: String,
    handler: String,
    sites: Seq[(SiteDef[?, ?], X)]
)(
    change: [T <: Tuple, V] => (SampleStatement[T, V], X) => SampleStatement[T, V]
)(model: ProbProg[A])(using outer: ProbContext): A =
  for (site, xs) <- sites.groupBy(_._1) if xs.size > 1 do
    throw IllegalArgumentException(
      s"Site '${site.name}' is $what more than once in the same $handler."
    )
  val bySite: Map[SiteDef[?, ?], X] = sites.toMap
  val sampled = mutable.LinkedHashSet.empty[SiteDef[?, ?]]

  val result = model(using
    new ProbContext:
      def sample[T <: Tuple, V](
          statement: SampleStatement[T, V]
      ): Tensor[T, V] =
        val site = statement.site
        if !sampled.add(site) then throw sampledTwice(site)
        bySite.get(site) match
          case None    => outer.sample(statement)
          case Some(x) => outer.sample(change[T, V](statement, x))
  )

  for site <- bySite.keys if !sampled.contains(site) do
    throw IllegalArgumentException(
      s"Site '${site.name}' is $what, but the model never samples it. " +
        s"Sites in the model: ${sampled.map(_.name).mkString(", ")}."
    )
  result
