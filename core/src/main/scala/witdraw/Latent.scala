package witdraw

import dimwit.*
import dimwit.random.Random
import dimwit.tensortree.{TensorTree, TreeOf}
import witdraw.inference.Chain

/** The axis of [[Latent.Params]]: the unconstrained values of all latent sites,
  * one after the other.
  */
trait LatentDim derives Label

/** The log-density of the model, and the log-density with the log-Jacobian of
  * the bijectors.
  */
private trait ScorePart derives Label

/** The draws of a chain, completed at once by [[Latent.recover]]. */
private trait RecoveredDraw derives Label

/** The latent sites of a model, found by tracing one forward run.
  *
  * Inference moves in `latent.Params`: the unconstrained values of all latent
  * sites, in one vector, e.g. a starting value or a draw of NUTS. What you
  * read is what the model returns: `latent.result(params)` at given
  * parameters, and `chain.samples` at every draw of a chain, the same type as
  * a forward run of the model.
  *
  * {{{
  * def model(using ProbContext) =
  *   val sigma = Sigma ~ HalfNormal(0f, 1f)
  *   val y = Y ~ Normal.isotropic(mu, sigma)
  *   (sigma = sigma, y = y)
  *
  * val latent = Latent.discover(observe(Y := yData)(model))
  * val chain = NUTS.adapt(latent, warmupKey, latent.init).sample(key, 1000)
  * chain.samples.map(_.sigma)     // always positive
  * latent.result(latent.init).sigma
  * }}}
  *
  * The model is interpreted on two latent spaces: the continuous sites, laid
  * out in the parameters by a [[ParamLayout]], and the sites summed out with
  * [[marginalize]], whose combinations of values an [[Enumeration]] lists. A
  * point of both, a parameter vector and a combination, determines every
  * site. The log-density sums over the combinations; a model with no sites
  * summed out has a single one.
  */
sealed abstract class Latent private (
    model: ProbProg[Any],
    layout: ParamLayout,
    enumeration: Enumeration,
    resultTree: TensorTree[?]
):
  import Latent.{logSumExp, stacked, traceOf}

  /** What inference algorithms keep for this latent, e.g. compiled kernels,
    * so that later calls, e.g. for further chains, reuse them. It lives with
    * the latent, as what it keeps refers to the latent's log-density, and is
    * dropped with it.
    */
  private val caches = scala.collection.concurrent.TrieMap.empty[Latent.CacheKey[?], Any]

  /** The value kept under `key`, made by `make` on first use. */
  private[witdraw] def cached[A](key: Latent.CacheKey[A])(make: => A): A =
    caches.getOrElseUpdate(key, make).asInstanceOf[A]

  /** The type of what the model returns, e.g. a named tuple of tensors. It is
    * inferred from the model, so it only needs a name where it is written
    * down, e.g. `(r: latent.Result) => r.sigma`.
    *
    * A function that returns a latent keeps it only if its result type is
    * inferred: declared as plain `Latent`, the result's fields are lost.
    */
  type Result

  /** Values for the latent sites of the model: their unconstrained values, in
    * one vector. This is where inference algorithms move, e.g. a starting value
    * or a draw, not something to read: [[result]] gives the model's result at
    * them.
    */
  opaque type Params = Tensor1[LatentDim, Float32]

  object Params:
    given TensorTree[Params] = TensorTree.tensor[Tuple1[LatentDim], Float32]
    given TreeOf[Params, Float32] = TreeOf.tensor[Tuple1[LatentDim], Float32]
    given ModelResult[Params, Result] = result(_)

  private given TensorTree[Result] = resultTree.asInstanceOf[TensorTree[Result]]

  /** The latent sites, in the order in which the model samples them, without
    * the ones summed out with [[marginalize]], see [[marginalizedSites]].
    */
  def sites: Seq[SiteDef[?, ?]] = layout.sites

  /** The sites summed out with [[marginalize]], in the order in which the
    * model samples them.
    */
  def marginalizedSites: Seq[SiteDef[?, ?]] = enumeration.sites.map(_.site)

  /** The model with every latent site set: the continuous ones to their values
    * at `params`, and the ones summed out to their values in the combination
    * `k`. Returns the model's result and the log-Jacobian of the bijectors.
    */
  private def at(params: Flat, k: Tensor0[Int32]): ProbProg[(Any, Tensor0[Float32])] =
    layout.substitute(params)(enumeration.condition(k)(model))

  /** The values of the latent sites at `params`, in the model's own space, by
    * site, e.g. for tests of the supports. Users read [[result]] instead.
    */
  private[witdraw] def values(params: Params): SiteValues =
    traceOf(layout.substitute(params)(model))

  /** The model's result at `params`: the model run with the latent sites set to
    * their values at `params`, and the observed sites to their data. The same
    * type as a forward run of the model, e.g. a prior sample, so prior and
    * posterior samples are read the same way. Nothing is drawn.
    *
    * With sites summed out by [[marginalize]], the parameters do not determine
    * the result, which may depend on those sites: it is an error. Complete the
    * draws of a chain with [[recover]] first, and read the samples of the
    * recovered chain.
    */
  def result(params: Params): Result =
    if !enumeration.isTrivial then
      val names = marginalizedSites.map(_.name).mkString(", ")
      throw IllegalStateException(
        s"The parameters do not hold the sites summed out with marginalize ($names), so they do not determine " +
          "the model's result. Draw those sites with latent.recover(chains, key), and read the samples of the " +
          "recovered chains, or get their posterior with latent.posteriorOf(site)(params)."
      )
    compiledResult(params, Tensor0(0))

  /** The model's result at the parameters and a combination, compiled once. */
  private val compiledResult: (Flat, Tensor0[Int32]) => Result =
    jit((p: Flat, k: Tensor0[Int32]) =>
      ForwardSampler.run(Random.Key(0))(at(p, k))._1.asInstanceOf[Result]
    )

  /** A draw of a chain completed by [[recover]]: the parameters, and values of
    * the sites summed out, drawn from their posterior given the parameters.
    * Read it as a draw of a chain: `chain.samples` gives the model's results.
    */
  final class Draw private[Latent] (
      val params: Params,
      private[Latent] val combination: Tensor0[Int32]
  )

  object Draw:
    given ModelResult[Draw, Result] = resultOf(_)

  /** The model's result at a completed draw. */
  def resultOf(draw: Draw): Result =
    compiledResult(draw.params, draw.combination)

  /** The values of `site`, summed out with [[marginalize]], along [[Choice]],
    * e.g. the years of a switchpoint, in the order of [[posteriorOf]].
    */
  def valuesOf[V](site: SiteDef[EmptyTuple, V]): Tensor1[Choice, V] =
    enumeration.enumeratedOf(site).values.asInstanceOf[Tensor1[Choice, V]]

  /** The posterior of `site`, summed out with [[marginalize]], given the other
    * sites at `params`: `p(z | theta, x)`, along the values of [[valuesOf]].
    * It is exact, so its average over the draws of a chain estimates the
    * posterior `p(z | x)` with less noise than counting recovered draws.
    */
  def posteriorOf(site: SiteDef[EmptyTuple, ?])(
      params: Params
  ): Tensor1[Choice, Float32] =
    val e = enumeration.enumeratedOf(site)
    compiledLogJoint(params)
      .softmax(Axis[Combination])
      .dot(Axis[Combination])(e.oneHot)

  /** The posterior `p(z | x)` of `site`, summed out with [[marginalize]],
    * estimated from chains of the other sites: the average of the exact
    * posterior `p(z | theta, x)`, see [[posteriorOf]], over all draws. It
    * estimates the same posterior as the frequencies of the draws of
    * [[recover]], with less noise (Rao-Blackwellisation).
    */
  def posteriorOf(
      site: SiteDef[EmptyTuple, ?],
      chains: Seq[Chain[Params, ?]]
  ): Tensor1[Choice, Float32] =
    val e = enumeration.enumeratedOf(site)
    val draws = chains.flatMap(_.positions)
    require(draws.nonEmpty, "posteriorOf needs at least one draw")
    val probs = compiledBatchLogJoint(stacked(draws)).softmax(Axis[Combination])
    probs.mean(Axis[RecoveredDraw]).dot(Axis[Combination])(e.oneHot)

  /** The chain with every draw completed: the sites summed out with
    * [[marginalize]] drawn from their posterior `p(z | theta, x)` given the
    * draw's parameters `theta`. Together, the recovered draws are draws of the
    * joint posterior, as `p(theta, z | x) = p(theta | x) p(z | theta, x)`.
    */
  def recover[I](chain: Chain[Params, I], key: Random.Key): Chain[Draw, I] =
    recover(Seq(chain), key).head

  /** [[recover]] for several chains, all at once. */
  def recover[I](
      chains: Seq[Chain[Params, I]],
      key: Random.Key
  ): Seq[Chain[Draw, I]] =
    if enumeration.isTrivial then
      throw IllegalStateException(
        "The model has no sites summed out with marginalize, so its draws are complete: there is nothing to recover."
      )
    val draws = chains.flatMap(_.draws)
    if draws.isEmpty then chains.map(_ => Chain(Seq.empty))
    else
      // all draws at once, by the Gumbel-max trick: the argmax of the
      // log-probabilities plus Gumbel noise is a draw of the categorical
      val logits = compiledBatchLogJoint(stacked(draws.map(_._1)))
      val u = dimwit.stats
        .Uniform(Tensor.like(logits).fill(0f), Tensor.like(logits).fill(1f))
        .sample(key)
      val gumbel = -(-u.log).log
      val combinations = (logits + gumbel).argmax(Axis[Combination]).toArray
      val completed = draws.zip(combinations).map { case ((p, info), k) =>
        (Draw(p, Tensor0(k)), info)
      }
      val ends = chains.map(_.size).scanLeft(0)(_ + _)
      ends.zip(ends.tail).map((from, until) => Chain(completed.slice(from, until)))

  /** `log p(z, theta, x)` for every combination `z` of values of the sites
    * summed out, up to a constant, at the parameters `theta`.
    */
  private val compiledLogJoint: Flat => Tensor1[Combination, Float32] =
    jit((p: Flat) => logJoint(p)._1)

  /** [[compiledLogJoint]] for many parameters at once. */
  private val compiledBatchLogJoint
      : Tensor2[RecoveredDraw, LatentDim, Float32] => Tensor2[RecoveredDraw, Combination, Float32] =
    jit((ps: Tensor2[RecoveredDraw, LatentDim, Float32]) =>
      ps.vmap(Axis[RecoveredDraw])(p => logJoint(p)._1)
    )

  /** For every combination of values of the sites summed out, the log-density
    * of the model at `params` with those values, without and with the
    * log-Jacobian of the bijectors, from one run of the model each,
    * vectorized.
    */
  private def logJoint(
      params: Flat
  ): (Tensor1[Combination, Float32], Tensor1[Combination, Float32]) =
    val parts = enumeration.combinations.vmap(Axis[Combination]): k =>
      val ((_, logJacobian), logDensity) = ModelScorer.runWithResult(at(params, k))
      stack(Seq(logDensity, logDensity + logJacobian), Axis[ScorePart])
    (
      parts.select(Axis[ScorePart].at(0)),
      parts.select(Axis[ScorePart].at(1))
    )

  /** Starting values for inference: the values of the traced run. */
  def init: Params = initWith()

  /** Starting values for inference: the given values for some sites, e.g.
    * `latent.initWith(Sigma := Tensor0(1f))`, and the values of the traced run
    * for the others.
    *
    * Every value must lie inside its support, e.g. be positive for a
    * `HalfNormal` site or increasing for an `Ordered` one. A traced value can
    * lie outside when given values move the support of another site, e.g.
    * `X ~ Uniform(0, sigma)` with a smaller given sigma, and then that site
    * needs a given value too.
    */
  def initWith(values: Observation[?, ?]*): Params =
    if values.isEmpty then layout.initial
    else
      val givenValues: Map[SiteDef[?, ?], Tensor[?, ?]] =
        values.map(o => layout.slotOf(o.site).site -> o.value).toMap
      def startValue(slot: ParamLayout.Slot[?]): Tensor[?, ?] =
        givenValues.getOrElse(slot.site, slot.traced)
      // the supports at the starting values, as a support may depend on
      // other sites
      val atStart = traceOf(observe(layout.slots.map(s => s.observation(startValue(s)))*)(model))
      layout.pack(
        startValue,
        slot => atStart.entry(slot.site).get.support,
        slot => if givenValues.contains(slot.site) then "given" else "traced"
      )

  /** Random starting values for inference, as in Stan: every unconstrained
    * value is drawn uniformly between `-radius` and `radius`. Since they are
    * unconstrained, the values always lie inside the supports.
    *
    * Use it to start several chains from different positions, e.g.
    * `key.split(4).map(latent.initRandom(_))`.
    */
  def initRandom(key: Random.Key, radius: Float = 2f): Params =
    layout.random(key, radius)

  /** The log-density to sample from, e.g. with NUTS: the density of the
    * unconstrained values, which is the density of the model times the
    * Jacobian of the bijectors, summed over the combinations of values of the
    * sites summed out.
    */
  def logDensity(params: Params): Tensor0[Float32] = logSumExp(logJoint(params)._2)

  /** The log-density of the model, observed on the sites' values at `params`,
    * summed over the combinations of values of the sites summed out. Maximise
    * this one to find the posterior mode: unlike [[logDensity]], its maximum
    * does not depend on the bijectors.
    */
  def modelLogDensity(params: Params): Tensor0[Float32] = logSumExp(logJoint(params)._1)

object Latent:

  /** The key of something an inference algorithm keeps with a latent, of type
    * `A`, e.g. NUTS's compiled kernel. Keys are compared by identity, so each
    * algorithm declares its own.
    */
  private[witdraw] final class CacheKey[A]

  /** Runs `model` forward once, and collects its latent sites: the sites that
    * are not observed. The values of that run are the default starting point,
    * [[Latent.init]].
    *
    * The run uses a fixed key: which sites are latent cannot depend on the
    * values drawn, so the key only decides the default starting point. Use
    * [[Latent.initWith]] to start elsewhere.
    *
    * The parameters are meant for gradient-based inference, so every latent
    * site must be continuous (`Float32`); a discrete latent site is an error,
    * unless it is summed out with [[marginalize]].
    *
    * Each latent site is sampled through the [[Bijector]] of the [[Support]]
    * of its distribution, e.g. on the log scale for `HalfNormal`, by
    * stick-breaking for `Dirichlet`, or as increasing values for `Ordered`.
    * The bijector is built anew in every run of the model, so a support may
    * depend on other sites, e.g. `X ~ Uniform(0, sigma)`. Increasing values
    * come from the distribution `Ordered`, e.g.
    * `Mu ~ Ordered(Normal(0f, 10f), Axis[Comp] -> k)`. A latent site whose
    * support is unknown is an error.
    *
    * The model must return a tensor tree, e.g. a named tuple of tensors: its
    * result is a sample of the posterior, see [[Latent.result]]. The result is
    * compiled, so reading the samples of a chain is fast. As in all of
    * inference, the model is traced: it may not read a value that depends on a
    * latent site with `.item`, but must use tensor operations instead, e.g.
    * `where` for a branch.
    */
  def discover[A](model: ProbProg[A])(using resultTree: TensorTree[A]): Latent {
    type Result = A
  } =
    val tr = traceOf(model)
    // a site observed outside its marginalize block is observed: the
    // observation is the more specific condition
    val (marginalized, latentSites) =
      tr.latentSites.partition(tr.entry(_).get.marginalized)

    for site <- latentSites if tr(site).dtype != DType.Float32 do
      throw IllegalArgumentException(
        s"Site '${site.name}' is ${tr(site).dtype}, but latent sites must be continuous (Float32). " +
          s"Observe it, or sum it out with marginalize(${site.name})."
      )
    // a second run, with other values, shows whether the values of the sites
    // summed out depend on the other sites
    val enumeration = Enumeration.of(
      marginalized.map(tr.entry(_).get),
      traceOf(model, Random.Key(1))
    )
    // latent sites are Float32, as checked above
    val layout = ParamLayout.of(
      latentSites.map(tr.entry(_).get.asInstanceOf[TraceEntry[Tuple, Float32]])
    )
    new Latent(model, layout, enumeration, resultTree):
      type Result = A

  /** `log(sum(exp(l)))`, shifted by the maximum to avoid overflow. When every
    * entry is `-inf`, it is `-inf`; for a single entry, it is that entry.
    */
  private def logSumExp[L: Label](l: Tensor1[L, Float32]): Tensor0[Float32] =
    val max = l.max
    val m = where(max.isfinite, max, Tensor0(0f))
    m + (l -! m).exp.sum.log

  /** `stack(draws, Axis[RecoveredDraw])`, but fast for many draws: JAX's
    * `stack` of thousands of arrays is slow, NumPy's on the host is not.
    */
  private def stacked(draws: Seq[Flat]): Tensor2[RecoveredDraw, LatentDim, Float32] =
    import me.shadaj.scalapy.py
    import me.shadaj.scalapy.py.SeqConverters
    val (np, jnp) = (py.module("numpy"), py.module("jax.numpy"))
    val tree = TensorTree[Flat]
    val host = draws.map(d => np.asarray(tree.toPyTree(d))).toPythonCopy
    TensorTree[Tensor2[RecoveredDraw, LatentDim, Float32]].fromPyTree(
      jnp.asarray(np.stack(host))
    )

  /** The trace of a forward run of `model`. When every latent site is filled
    * in, as by `observe` or `substitute`, nothing is drawn, and the key does
    * not matter.
    */
  private def traceOf(model: ProbProg[Any], key: Random.Key = Random.Key(0)) =
    ForwardSampler.run(key)(withTrace(model))._2
