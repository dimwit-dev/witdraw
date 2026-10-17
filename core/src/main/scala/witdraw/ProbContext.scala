package witdraw

import dimwit.*
import dimwit.stats.Distribution
import scala.annotation.implicitNotFound

/** A probabilistic program: code that needs a [[ProbContext]] to run.
  *
  * Models are usually written as methods with a `ProbContext` parameter, which
  * is a `ProbProg` once its other arguments are applied:
  * {{{
  * def model(x: Tensor2[Sample, Feature, Float32])(using ProbContext) =
  *   val beta = Beta ~ Normal.isotropic(...)
  *   val y = Y ~ Normal.isotropic(x.dot(Axis[Feature])(beta), 1f)
  *   (beta = beta, y = y)
  *
  * ForwardSampler.run(key)(model(xData))
  * }}}
  *
  * Use the type explicitly where a program is passed around or stored without
  * being run, e.g. `val m: ProbProg[Result] = model(x)` or a parameter
  * `prior: ProbProg[Params]`. Inside another program, a sub-program is run like
  * any other expression: `val params = prior`.
  */
type ProbProg[+A] = ProbContext ?=> A

/** The capability to run probabilistic statements.
  *
  * Every `site ~ dist` in a model becomes a call to [[sample]]. What happens
  * then depends on who provides the context:
  *
  *   - an interpreter ([[ForwardSampler]], [[ScoredSampler]], [[ModelScorer]])
  *     decides the value of the site and what to compute with it, and
  *   - a [[Handler]] (e.g. [[observe]]) wraps the context it is given and
  *     changes the statements of a block before passing them on.
  *
  * A handler sees every statement executed inside its block. Handlers that are
  * nested further in act first.
  */
@implicitNotFound(
  "Sampling a site needs a ProbContext. Run the model with an interpreter, e.g. ForwardSampler.run(key)(model) or ModelScorer.run(model), or give the method a (using ProbContext) parameter."
)
trait ProbContext:
  /** Run the statement `statement.site ~ statement.distribution` and return the
    * value of the site.
    */
  def sample[T <: Tuple, V](statement: SampleStatement[T, V]): Tensor[T, V]

/** One `site ~ distribution` statement, as seen by handlers and interpreters.
  *
  * @param observed
  *   the value of the site, if a handler has fixed it (e.g. [[observe]]).
  *   `None` means the site is free: the interpreter chooses its value.
  */
case class SampleStatement[T <: Tuple, V](
    site: SiteDef[T, V],
    distribution: Distribution[T, V],
    observed: Option[Tensor[T, V]]
)

/** Base class for handlers: a context that passes every statement on to the
  * `outer` context. Override [[sample]] to change statements, then pass them on
  * with `outer.sample(...)`.
  *
  * A handler is a function that runs a block with the handler as its context:
  * {{{
  * def logSites[A](block: ProbProg[A])(using ctx: ProbContext): A =
  *   block(using new Handler(ctx):
  *     override def sample[T <: Tuple, V](s: SampleStatement[T, V]): Tensor[T, V] =
  *       println(s"sampling ${s.site.name}")
  *       outer.sample(s)
  *   )
  * }}}
  */
abstract class Handler(protected val outer: ProbContext) extends ProbContext:
  def sample[T <: Tuple, V](statement: SampleStatement[T, V]): Tensor[T, V] =
    outer.sample(statement)
