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
    outer: ProbContext
): A =
  for (site, obs) <- observations.groupBy(_.site) if obs.size > 1 do
    throw IllegalArgumentException(
      s"Site '${site.name}' is observed more than once in the same observe."
    )
  val valueOf: Map[SiteDef[?, ?], Any] =
    observations.map(o => o.site -> o.value).toMap
  val sampled = mutable.LinkedHashSet.empty[SiteDef[?, ?]]

  val result = model(using
    new ProbContext:
      def sample[T <: Tuple, V](
          statement: SampleStatement[T, V]
      ): Tensor[T, V] =
        val site = statement.site
        if !sampled.add(site) then throw sampledTwice(site)
        valueOf.get(site) match
          case None        => outer.sample(statement)
          case Some(value) =>
            if statement.observed.isDefined then
              throw IllegalArgumentException(
                s"Site '${site.name}' is observed twice, by nested observe blocks."
              )
            val observed = value.asInstanceOf[Tensor[T, V]]
            outer.sample(statement.copy(observed = Some(observed)))
  )

  for site <- valueOf.keys if !sampled.contains(site) do
    throw IllegalArgumentException(
      s"Site '${site.name}' is observed, but the model never samples it. " +
        s"Sites in the model: ${sampled.map(_.name).mkString(", ")}."
    )
  result
