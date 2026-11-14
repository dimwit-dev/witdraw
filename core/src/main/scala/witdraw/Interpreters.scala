package witdraw

import dimwit.*
import dimwit.random.Random

/** Runs a model forward: free sites are drawn from their distributions,
  * sites whose value a handler fixed, e.g. observed ones, keep that value. The
  * result is the model's own return value.
  */
object ForwardSampler:
  def run[A](key: Random.Key)(model: ProbProg[A]): A =
    model(using
      new ProbContext:
        private var nextKey = key

        def sample[T <: Tuple, V](
            statement: SampleStatement[T, V]
        ): Tensor[T, V] =
          statement.value.getOrElse:
            val (keep, use) = nextKey.split2()
            nextKey = keep
            statement.distribution.sample(use)
    )

  /** `n` independent runs of `model`, each with its own key, split from
    * `key`, e.g. draws from the prior predictive distribution:
    * {{{
    * val priorPredictive = ForwardSampler.sample(key, 1000)(model(x))
    * }}}
    */
  def sample[A](key: Random.Key, n: Int)(model: ProbProg[A]): Seq[A] =
    require(n > 0, s"n must be positive, but is $n")
    key.split(n).map(k => run(k)(model))

/** Computes the log-density of a model in which every site has a value, e.g.
  * the unnormalised log posterior of a model observed on data, with the
  * parameters set by a [[Latent]]. A free site is an error.
  */
object ModelScorer:
  def run[A](model: ProbProg[A]): Tensor0[Float32] = runWithResult(model)._2

  /** The model's result, and its log-density. */
  def runWithResult[A](model: ProbProg[A]): (A, Tensor0[Float32]) =
    var logProb = Tensor0(0f)
    val result = model(using
      new ProbContext:
        def sample[T <: Tuple, V](
            statement: SampleStatement[T, V]
        ): Tensor[T, V] =
          val value = statement.value.getOrElse:
            throw IllegalArgumentException(
              s"Site '${statement.site.name}' is free, but ModelScorer needs every site to have a value (use observe, or Latent for the parameters)."
            )
          logProb = logProb + statement.distribution.logProb(value).asFloat
          value
    )
    (result, logProb)
