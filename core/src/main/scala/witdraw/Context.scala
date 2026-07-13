package witdraw

import dimwit.random.Random
import dimwit.*
import dimwit.stats.*

/** A probabilistic program with context-dependent interpretation.
  *
  * `ProbProg` wraps a context function `ProbContext ?=> Val` in a class to
  * prevent Scala 3 from auto-applying the context function when a `ProbContext`
  * is in scope. This is essential for `.observe(...)` to work correctly —
  * otherwise the receiver would be eagerly evaluated before the observation
  * wrapper can intercept `assume` calls.
  *
  * The type parameter [[Val]] represents the value this program yields during
  * composition (e.g., for passing between sub-programs). When executed by a
  * context:
  *   - [[ForwardSampler]] returns [[Val]] directly (the model's return value)
  *   - [[ModelScorer]] returns [[Tensor0]][Float] (log-probability)
  *
  * @tparam Val
  *   the value this program fragment yields for composition; [[Unit]] for
  *   effect-only programs
  */
class ProbProg[+Val](private[witdraw] val underlying: ProbContext ?=> Val):

  /** Execute this program under the given context. */
  inline def run(using ctx: ProbContext): Val = underlying(using ctx)

  /** Condition this program by fixing one or more sites to observed values.
    *
    * Returns a new `ProbProg` that intercepts `RandomVar` calls: if the site
    * matches an observation, it is treated as observed; otherwise it passes
    * through as free.
    *
    * @example
    *   {{{
    * val W = Site.auto[Tensor0[Float]]
    * val Y = Site.auto[Tensor1[Sample, Float]]
    *
    * def model: ProbProg[Result] = ProbProg:
    *   val w = RandomVar(W, Normal(Tensor0(0f), Tensor0(1f)))
    *   val y = RandomVar(Y, Normal.isotropic(w *! xData, Tensor0(1f)))
    *   Result(w, y)
    *
    * // Prior predictive — everything free
    * draw(model, key)
    *
    * // Training — condition Y on observed data
    * draw(model.observe(Y := targetData), key)
    *   }}}
    */
  def observe(observations: ObservedVariable[?]*): ProbProg[Val] =
    val obsMap: Map[Site[?], ObservedVariable[?]] =
      observations.map(o => o.site -> o).toMap
    ProbProg:
      val outerCtx = summon[ProbContext]
      val innerCtx = new ProbContext:
        override def randomVar[Ten <: Tensor[?, ?]](
            variable: SiteVariable[Ten],
            dist: Distribution[ShapeOf[Ten], ElemOf[Ten]]
        ): Ten =
          obsMap.get(variable.site) match
            case Some(obs) =>
              outerCtx.randomVar(obs.asInstanceOf[ObservedVariable[Ten]], dist)
            case None =>
              outerCtx.randomVar(variable, dist)
      ProbProg.this.run(using innerCtx)

/** Context that executes a probabilistic program. Each context type returns its
  * own result type, independent of the program's [[Val]] type parameter.
  */
trait ProbContext:
  def randomVar[Ten <: Tensor[?, ?]](
      variable: SiteVariable[Ten],
      dist: Distribution[ShapeOf[Ten], ElemOf[Ten]]
  ): Ten

def RandomVar[Ten <: Tensor[?, ?]](
    site: SiteVariable[Ten],
    dist: Distribution[ShapeOf[Ten], ElemOf[Ten]]
)(using ctx: ProbContext): Ten = ctx.randomVar(site, dist)

/** Forward-sampling interpreter — result is the model’s return value.
  *
  *   - Free [[Site]]: sampled from the distribution.
  *   - [[ObservedVariable]]: fixed value, scored.
  */

class ForwardSampler(initKey: Random.Key) extends ProbContext:

  private var logProbAcc = Tensor0(0f)
  private var key: Random.Key = initKey

  override def randomVar[Ten <: Tensor[?, ?]](
      variable: SiteVariable[Ten],
      dist: Distribution[ShapeOf[Ten], ElemOf[Ten]]
  ): Ten =
    variable match
      case ObservedVariable(parentSite, value) =>
        logProbAcc = logProbAcc + dist
          .logProb(value.asInstanceOf[Tensor[ShapeOf[Ten], ElemOf[Ten]]])
          .asFloat
        value
      case site =>
        val (k1, k2) = key.split2()
        key = k1
        dist.sample(k2).asInstanceOf[Ten]

  def logProb: Tensor0[Float32] = logProbAcc

object ForwardSampler:
  def run[Val](key: Random.Key)(program: ProbProg[Val]): Val =
    val ctx = new ForwardSampler(key)
    program.run(using ctx)

/** Sampling interpreter that scores all sites — both free and observed.
  *
  * Unlike [[ForwardSampler]] (which only scores observed sites), this
  * accumulates the full log-density of the execution, making it suitable for
  * guides in VI, importance sampling, and any algorithm that needs log q(z)
  * alongside the sample.
  *
  *   - Free [[Site]]: sampled from the distribution and scored.
  *   - [[ObservedVariable]]: fixed value, scored.
  */
class ScoredSampler(initKey: Random.Key) extends ProbContext:

  private var logProbAcc = Tensor0(0f)
  private var key: Random.Key = initKey

  override def randomVar[Ten <: Tensor[?, ?]](
      variable: SiteVariable[Ten],
      dist: Distribution[ShapeOf[Ten], ElemOf[Ten]]
  ): Ten =
    val value: Tensor[ShapeOf[Ten], ElemOf[Ten]] = variable match
      case ObservedVariable(_, v) =>
        v.asInstanceOf[Tensor[ShapeOf[Ten], ElemOf[Ten]]]
      case _ =>
        val (k1, k2) = key.split2()
        key = k1
        dist.sample(k2)
    logProbAcc = logProbAcc + dist.logProb(value).asFloat
    value.asInstanceOf[Ten]

  def logProb: Tensor0[Float32] = logProbAcc

object ScoredSampler:
  def run[Val](key: Random.Key)(
      program: ProbProg[Val]
  ): (Val, Tensor0[Float32]) =
    val ctx = new ScoredSampler(key)
    val value = program.run(using ctx)
    (value, ctx.logProb)

/** Scoring interpreter — result is the total log-probability.
  *
  * Every site in the model must be an [[ObservedVariable]] (via `:=`). The
  * scorer accumulates the log-probability of each observed value under its
  * distribution.
  */
private class ModelScorer extends ProbContext:

  private var logProbAcc: Tensor0[Float32] = Tensor0(0f)

  override def randomVar[Ten <: Tensor[?, ?]](
      variable: SiteVariable[Ten],
      dist: Distribution[ShapeOf[Ten], ElemOf[Ten]]
  ): Ten =
    val value: Tensor[ShapeOf[Ten], ElemOf[Ten]] = variable match
      case ObservedVariable(_, v) =>
        v.asInstanceOf[Tensor[ShapeOf[Ten], ElemOf[Ten]]]
      case site =>
        throw new IllegalArgumentException(
          s"Site '${site.name}' is free but ModelScorer requires all sites to be observed (use site := value)"
        )
    logProbAcc = logProbAcc + dist.logProb(value).asFloat
    value.asInstanceOf[Ten]

  def result: Tensor0[Float32] = logProbAcc

object ModelScorer:
  def run[Val](program: ProbProg[Val]): Tensor0[Float32] =
    val ctx = new ModelScorer
    program.run(using ctx)
    ctx.result
