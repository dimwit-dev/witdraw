package witdraw

import dimwit.*
import dimwit.random.Random
import scala.collection.mutable

/** Base class for interpreters: the outermost context, which decides the value
  * of each site and what to compute with it.
  *
  * Implement [[interpret]]. This class checks that no site is sampled twice in
  * one run, which would make observations of that site ambiguous.
  */
abstract class Interpreter extends ProbContext:
  private val sampled = mutable.Set.empty[Site[?]]

  final def sample[Ten <: Tensor[?, ?]](statement: SampleStatement[Ten]): Ten =
    if !sampled.add(statement.site) then
      throw IllegalArgumentException(
        s"Site '${statement.site.name}' is sampled more than once. Each site may appear only once in a model."
      )
    interpret(statement)

  protected def interpret[Ten <: Tensor[?, ?]](
      statement: SampleStatement[Ten]
  ): Ten

/** Runs a model forward: free sites are drawn from their distributions,
  * observed sites keep their observed value. The result is the model's own
  * return value.
  */
object ForwardSampler:
  def run[A](key: Random.Key)(model: ProbProg[A]): A =
    model(using
      new Interpreter:
        private var nextKey = key

        protected def interpret[Ten <: Tensor[?, ?]](
            statement: SampleStatement[Ten]
        ): Ten =
          statement.observed.getOrElse:
            val (keep, use) = nextKey.split2()
            nextKey = keep
            statement.distribution.sample(use).asInstanceOf[Ten]
    )

/** Like [[ForwardSampler]], but also returns the log-density of all sites, free
  * and observed. Suitable for guides in variational inference and for
  * importance sampling, which need log q(z) alongside the sample.
  */
object ScoredSampler:
  def run[A](key: Random.Key)(model: ProbProg[A]): (A, Tensor0[Float32]) =
    var logProb = Tensor0(0f)
    val result = model(using
      new Interpreter:
        private var nextKey = key

        protected def interpret[Ten <: Tensor[?, ?]](
            statement: SampleStatement[Ten]
        ): Ten =
          val value = statement.observed.getOrElse:
            val (keep, use) = nextKey.split2()
            nextKey = keep
            statement.distribution.sample(use).asInstanceOf[Ten]
          logProb = logProb + logProbOf(statement, value)
          value
    )
    (result, logProb)

/** Computes the log-density of a model in which every site is observed, e.g.
  * the unnormalised log posterior of a model observed on data and parameters. A
  * free site is an error.
  */
object ModelScorer:
  def run[A](model: ProbProg[A]): Tensor0[Float32] =
    var logProb = Tensor0(0f)
    model(using
      new Interpreter:
        protected def interpret[Ten <: Tensor[?, ?]](
            statement: SampleStatement[Ten]
        ): Ten =
          val value = statement.observed.getOrElse:
            throw IllegalArgumentException(
              s"Site '${statement.site.name}' is free, but ModelScorer needs every site to be observed (use observe, or Latent for the parameters)."
            )
          logProb = logProb + logProbOf(statement, value)
          value
    )
    logProb

private def logProbOf[Ten <: Tensor[?, ?]](
    statement: SampleStatement[Ten],
    value: Ten
): Tensor0[Float32] =
  statement.distribution
    .logProb(value.asInstanceOf[Tensor[ShapeOf[Ten], ElemOf[Ten]]])
    .asFloat
