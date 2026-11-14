package witdraw

import dimwit.*
import dimwit.random.Random
import dimwit.tensortree.{TensorTree, TreeOf}
import witdraw.inference.Chain

/** The axis of [[Latent.Params]]: the unconstrained values of all latent sites,
  * one after the other.
  */
trait LatentDim derives Label

/** The axis along the values of a site summed out with [[marginalize]], e.g.
  * of [[Latent.valuesOf]] and [[Latent.posteriorOf]].
  */
trait Choice derives Label

/** The axis along all combinations of values of the sites summed out with
  * [[marginalize]]. Users never see it: [[Latent.posteriorOf]] gives the
  * posterior of one site along [[Choice]].
  */
private[witdraw] trait Combination derives Label

/** The log-density of the model, and the log-Jacobian of the bijectors. */
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
  */
sealed abstract class Latent private (
    model: ProbProg[Any],
    layout: ParamLayout,
    enumeration: Option[Latent.Enumeration],
    resultTree: TensorTree[?]
):
  import Latent.{Enumerated, Enumeration, logSumExp, traceOf}

  private val enumeratedBySite: Map[SiteDef[?, ?], Enumerated[?]] =
    enumeration.fold(Map.empty)(_.sites.map(e => e.site -> e).toMap)

  /** What inference algorithms keep for this latent, e.g. compiled functions,
    * so that later calls, e.g. for further chains, reuse them.
    */
  private val caches = scala.collection.concurrent.TrieMap.empty[String, AnyRef]

  /** The value kept under `key`, made by `make` on first use. */
  private[witdraw] def cached(key: String)(make: => AnyRef): AnyRef =
    caches.getOrElseUpdate(key, make)

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

  private val compiled = Compiled()

  /** The latent sites, in the order in which the model samples them, without
    * the ones summed out with [[marginalize]], see [[marginalizedSites]].
    */
  def sites: Seq[SiteDef[?, ?]] = layout.sites

  /** The sites summed out with [[marginalize]], in the order in which the
    * model samples them.
    */
  def marginalizedSites: Seq[SiteDef[?, ?]] =
    enumeration.fold(Seq.empty)(_.sites.map(_.site))

  /** The model with the sites summed out observed on their values in the
    * combination `k`, an index along [[Combination]], which may be traced.
    */
  private def conditionedOn(k: Tensor0[Int32]): ProbProg[Any] =
    observe(enumeration.get.observationsAt(k)*)(model)

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
    if enumeration.isDefined then
      val names = marginalizedSites.map(_.name).mkString(", ")
      throw IllegalStateException(
        s"The parameters do not hold the sites summed out with marginalize ($names), so they do not determine " +
          "the model's result. Draw those sites with latent.recover(chains, key), and read the samples of the " +
          "recovered chains, or get their posterior with latent.posteriorOf(site)(params)."
      )
    compiledResult(params)

  /** The model's result at `params`, run with the sites summed out observed as
    * in `m`. Compiled once; a model that cannot be compiled, e.g. calls `.item`
    * on a sampled value, runs eagerly.
    */
  private def resultIn(m: ProbProg[Any], params: Flat): Result =
    ForwardSampler.run(Random.Key(0))(layout.substitute(params)(m))._1.asInstanceOf[Result]

  private val compiledResult: Flat => Result =
    compiled((p: Flat) => resultIn(model, p))

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
    compiledResultAt(draw.params, draw.combination)

  private val compiledResultAt: (Flat, Tensor0[Int32]) => Result =
    compiled((p: Flat, k: Tensor0[Int32]) => resultIn(conditionedOn(k), p))

  /** The values of `site`, summed out with [[marginalize]], along [[Choice]],
    * e.g. the years of a switchpoint, in the order of [[posteriorOf]].
    */
  def valuesOf[V](site: SiteDef[EmptyTuple, V]): Tensor1[Choice, V] =
    enumeratedOf(site).values.asInstanceOf[Tensor1[Choice, V]]

  /** The posterior of `site`, summed out with [[marginalize]], given the other
    * sites at `params`: `p(z | theta, x)`, along the values of [[valuesOf]].
    * It is exact, so its average over the draws of a chain estimates the
    * posterior `p(z | x)` with less noise than counting recovered draws.
    */
  def posteriorOf(site: SiteDef[EmptyTuple, ?])(
      params: Params
  ): Tensor1[Choice, Float32] =
    val e = enumeratedOf(site)
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
    val e = enumeratedOf(site)
    val draws = chains.flatMap(_.positions)
    require(draws.nonEmpty, "posteriorOf needs at least one draw")
    val probs = compiledBatchLogJoint(stacked(draws)).softmax(Axis[Combination])
    probs.mean(Axis[RecoveredDraw]).dot(Axis[Combination])(e.oneHot)

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
    if enumeration.isEmpty then
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

  private def enumeratedOf(site: SiteDef[?, ?]): Enumerated[?] =
    enumeratedBySite.getOrElse(
      site,
      throw java.util.NoSuchElementException(
        s"Site '${site.name}' is not summed out with marginalize. Sites summed out: ${marginalizedSites.map(_.name).mkString(", ")}"
      )
    )

  /** `log p(z, theta, x)` for every combination `z` of values of the sites
    * summed out, up to a constant, at the parameters `theta`.
    */
  private val compiledLogJoint: Flat => Tensor1[Combination, Float32] =
    compiled((p: Flat) => logJoint(p)._1)

  /** [[compiledLogJoint]] for many parameters at once. */
  private val compiledBatchLogJoint: Tensor2[RecoveredDraw, LatentDim, Float32] => Tensor2[
    RecoveredDraw,
    Combination,
    Float32
  ] =
    compiled((ps: Tensor2[RecoveredDraw, LatentDim, Float32]) =>
      ps.vmap(Axis[RecoveredDraw])(p => logJoint(p)._1)
    )

  /** For every combination of values of the sites summed out, the log-density
    * of the model with those values and the log-Jacobian of the bijectors,
    * from one run of the model each, vectorized.
    */
  private def logJoint(
      params: Flat
  ): (Tensor1[Combination, Float32], Tensor1[Combination, Float32]) =
    val parts = enumeration.get.combinations.vmap(Axis[Combination]): k =>
      val (logDensity, logJacobian) = scoreIn(conditionedOn(k), params)
      stack(Seq(logDensity, logJacobian), Axis[ScorePart])
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
    * unconstrained values, which is [[modelLogDensity]] plus the log-Jacobian
    * of the bijectors.
    */
  def logDensity(params: Params): Tensor0[Float32] =
    val (modelLogDensity, logJacobian) = score(params)
    modelLogDensity + logJacobian

  /** The log-density of the model, observed on the sites' values at `params`.
    * Maximise this one to find the posterior mode: unlike [[logDensity]], its
    * maximum does not depend on the bijectors.
    */
  def modelLogDensity(params: Params): Tensor0[Float32] = score(params)._1

  /** The log-density of the model at `params`, and the log-Jacobian of the
    * bijectors, from one run of the model, or, with sites summed out, from one
    * run for every combination of their values: then the log-density is the
    * log of the sum of the densities, and the log-Jacobian is the one of the
    * sum of the densities with the Jacobians.
    */
  private def score(params: Flat): (Tensor0[Float32], Tensor0[Float32]) =
    enumeration match
      case None    => scoreIn(model, params)
      case Some(_) =>
        val (logDensities, logJacobians) = logJoint(params)
        val modelLogDensity = logSumExp(logDensities)
        (modelLogDensity, logSumExp(logDensities + logJacobians) - modelLogDensity)

  /** The log-density of `m` at `params`, and the log-Jacobian of the bijectors,
    * from one run.
    */
  private def scoreIn(
      m: ProbProg[Any],
      params: Flat
  ): (Tensor0[Float32], Tensor0[Float32]) =
    val ((_, logJacobian), logDensity) =
      ModelScorer.runWithResult(layout.substitute(params)(m))
    (logDensity, logJacobian)

object Latent:

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
    * compiled, so reading the samples of a chain is fast.
    */
  def discover[A](model: ProbProg[A])(using resultTree: TensorTree[A]): Latent {
    type Result = A
  } =
    val tr = traceOf(model)
    for e <- tr.entries if e.observed && e.marginalized do
      throw IllegalArgumentException(
        s"Site '${e.site.name}' is both observed and marginalized. Only latent sites can be summed out."
      )
    val (marginalized, latentSites) =
      tr.latentSites.partition(tr.entry(_).get.marginalized)

    for site <- latentSites if tr(site).dtype != DType.Float32 do
      throw IllegalArgumentException(
        s"Site '${site.name}' is ${tr(site).dtype}, but latent sites must be continuous (Float32). " +
          s"Observe it, or sum it out with marginalize(${site.name})."
      )
    // A second run, with other values, to check that the values of the sites
    // summed out do not depend on the other sites.
    val enumeration = Option.when(marginalized.nonEmpty)(
      enumerationOf(marginalized.map(tr.entry(_).get), traceOf(model, Random.Key(1)))
    )
    // latent sites are Float32, as checked above
    val layout = ParamLayout.of(
      latentSites.map(tr.entry(_).get.asInstanceOf[TraceEntry[Tuple, Float32]])
    )
    new Latent(model, layout, enumeration, resultTree):
      type Result = A

  /** `log(sum(exp(l)))`, shifted by the maximum to avoid overflow. */
  private def logSumExp[L: Label](l: Tensor1[L, Float32]): Tensor0[Float32] =
    val m = l.max
    m + (l -! m).exp.sum.log

  /** The most combinations of values of sites summed out. */
  private val maxCombinations = 1_000_000

  /** The sites summed out, `entries`, traced as in `other` in another run, and
    * all combinations of their values, the first site varying slowest.
    */
  private def enumerationOf(
      entries: Seq[TraceEntry[?, ?]],
      other: Trace
  ): Enumeration =
    val values = entries.map: e =>
      val (description, tensor) = valuesOf(e)
      val otherDescription = other.entry(e.site).map(valuesOf(_)._1)
      if !otherDescription.contains(description) then
        throw IllegalArgumentException(
          s"Site '${e.site.name}' is marginalized, but its values depend on other sites " +
            s"($description in one run, ${otherDescription.getOrElse("not sampled")} in another). " +
            "The values of a site summed out must be fixed."
        )
      tensor
    val sizes = values.map(_.shape(Axis[Choice]))
    val total = sizes.foldLeft(1L)(_ * _)
    if total > maxCombinations then
      throw IllegalArgumentException(
        s"The sites summed out (${entries.map(_.site.name).mkString(", ")}) have $total combinations of values, " +
          s"more than $maxCombinations: every log-density evaluation would run the model for each of them."
      )
    val n = total.toInt
    val strides = sizes.scanRight(1)(_ * _).tail
    val enumerated = entries.zip(values).zip(sizes.zip(strides)).map:
      case ((e, v), (size, stride)) =>
        val index = Array.tabulate(n)(k => (k / stride) % size)
        Enumerated(
          e.site,
          v,
          v.select(Axis[Choice].at(Tensor1(Axis[Combination]).fromArray(index))),
          Tensor2(Axis[Combination], Axis[Choice]).fromArray(
            Array.tabulate(n, size)((k, j) => if index(k) == j then 1f else 0f)
          )
        )
    Enumeration(enumerated.toVector, n)

  /** The values of a site to sum out, traced as `e`: a description to compare
    * them between runs, and the values along [[Choice]].
    */
  private def valuesOf(e: TraceEntry[?, ?]): (Any, Tensor1[Choice, ?]) =
    val name = e.site.name
    if e.value.shape.dimensions.nonEmpty then
      throw IllegalArgumentException(
        s"Site '$name' is marginalized, but it is not scalar (shape ${e.value.shape}). Only scalar sites can be summed out."
      )
    e.support match
      case Support.Integers(lo, hi) if e.value.dtype == DType.Int32 =>
        val (l, h) = (
          lo.asInstanceOf[Tensor0[Int32]].item,
          hi.asInstanceOf[Tensor0[Int32]].item
        )
        if l >= h then
          throw IllegalArgumentException(
            s"Site '$name' is marginalized, but it has no values: the integers from $l until $h."
          )
        (s"the integers from $l until $h", Tensor1(Axis[Choice]).fromArray(Array.range(l, h)))
      case Support.Booleans() =>
        ("false and true", Tensor1(Axis[Choice]).fromArray(Array(false, true)))
      case support =>
        val distribution = e.distribution.getClass.getSimpleName
        throw IllegalArgumentException(
          s"Site '$name' is marginalized, but the values of its distribution ($distribution) cannot be listed " +
            s"(support $support). Sites with a DiscreteUniform, Categorical or Bernoulli distribution can be summed out."
        )

  /** All combinations of values of the sites summed out, along
    * [[Combination]].
    */
  private final class Enumeration(
      val sites: Vector[Enumerated[?]],
      val size: Int
  ):
    val combinations: Tensor1[Combination, Int32] =
      Tensor1(Axis[Combination]).fromArray(Array.range(0, size))

    /** The values of all sites summed out in the combination `k`, which may
      * be traced, to condition the model on with [[observe]].
      */
    def observationsAt(k: Tensor0[Int32]): Seq[Observation[?, ?]] =
      sites.map(_.observationAt(k))

  /** A site summed out.
    *
    * @param values
    *   its values
    * @param valuesByCombination
    *   its value in every combination
    * @param oneHot
    *   which of its values each combination has
    */
  private final class Enumerated[V](
      val site: SiteDef[?, ?],
      val values: Tensor1[Choice, V],
      valuesByCombination: Tensor1[Combination, V],
      val oneHot: Tensor2[Combination, Choice, Float32]
  ):
    /** Its value in the combination `k`, which may be traced. */
    def observationAt(k: Tensor0[Int32]): Observation[EmptyTuple, V] =
      Observation(
        site.asInstanceOf[SiteDef[EmptyTuple, V]],
        valuesByCombination.select(Axis[Combination].at(k))
      )

  /** The trace of a forward run of `model`. When every latent site is filled
    * in, as by `observe` or `substitute`, nothing is drawn, and the key does
    * not matter.
    */
  private def traceOf(model: ProbProg[Any], key: Random.Key = Random.Key(0)) =
    ForwardSampler.run(key)(withTrace(model))._2
