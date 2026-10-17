package witdraw

import dimwit.*
import dimwit.random.Random

/** Runs a model forward: free sites are drawn from their distributions,
  * observed sites keep their observed value. The result is the model's own
  * return value.
  */
object ForwardSampler:
  def run[A](key: Random.Key)(model: ProbProg[A]): A =
    model(using
      new ProbContext:
        private var nextKey = key

        def sample[T <: Tuple, V](
            statement: SampleStatement[T, V]
        ): Tensor[T, V] =
          statement.observed.getOrElse:
            val (keep, use) = nextKey.split2()
            nextKey = keep
            statement.distribution.sample(use)
    )

/** Computes the log-density of a model in which every site is observed, e.g.
  * the unnormalised log posterior of a model observed on data and parameters. A
  * free site is an error.
  */
object ModelScorer:
  def run[A](model: ProbProg[A]): Tensor0[Float32] =
    var logProb = Tensor0(0f)
    model(using
      new ProbContext:
        def sample[T <: Tuple, V](
            statement: SampleStatement[T, V]
        ): Tensor[T, V] =
          val value = statement.observed.getOrElse:
            throw IllegalArgumentException(
              s"Site '${statement.site.name}' is free, but ModelScorer needs every site to be observed (use observe, or Latent for the parameters)."
            )
          logProb = logProb + statement.distribution.logProb(value).asFloat
          value
    )
    logProb
