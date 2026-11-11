package witdraw

import dimwit.*
import dimwit.stats.Distribution
import scala.annotation.implicitNotFound

/** A model that has not been run yet. Given a context, it runs and produces a
  * value of type A.
  */
type ProbProg[+A] = ProbContext ?=> A

/** The probabilistic context handles the sample calls of a model. Every
  * `site ~ dist` in the model becomes a call to `sample`, which returns the
  * site's value.
  *
  * An interpreter (e.g. ForwardSampler) is the outermost context: it decides
  * the value.
  *
  * A handler (e.g. observe, withTrace) is a function that takes a block of
  * model code and runs it with a new context. That context wraps the one the
  * handler was given: it sees every sample call of the block, may change or
  * record it, and passes it on. In
  * `ForwardSampler.run(key)(withTrace(observe(Y := y)(model)))`, a call goes
  * from the model to observe, then withTrace, then ForwardSampler.
  */
@implicitNotFound(
  "Sampling a site needs a ProbContext. Run the model with an interpreter, e.g. ForwardSampler.run(key)(model) or ModelScorer.run(model), or give the method a (using ProbContext) parameter."
)
trait ProbContext:
  def sample[T <: Tuple, V](statement: SampleStatement[T, V]): Tensor[T, V]

/** One `site ~ distribution` call, with the site's value if a handler has
  * observed it.
  *
  * @param support
  *   the [[Support]] of the distribution
  */
case class SampleStatement[T <: Tuple, V](
    site: SiteDef[T, V],
    distribution: Distribution[T, V],
    support: Support[T, V],
    observed: Option[Tensor[T, V]]
)

/** The error for a model that samples `site` more than once, which makes
  * observing or tracing that site ambiguous.
  */
private[witdraw] def sampledTwice(
    site: SiteDef[?, ?]
): IllegalArgumentException =
  IllegalArgumentException(
    s"Site '${site.name}' is sampled more than once. Each site may appear only once in a model."
  )
