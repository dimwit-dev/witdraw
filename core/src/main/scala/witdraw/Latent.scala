package witdraw

import dimwit.*
import dimwit.jax.Jax
import dimwit.random.Random
import dimwit.tensor.Labels
import dimwit.tensortree.{TensorTree, TreeOf}
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import witdraw.reparam.Bijector

/** A change of variables for a latent site: inference algorithms move freely in
  * the unconstrained space of `bijector`, and the model sees
  * `bijector.forward(z)`. For example, `Constraint(Sigma, Bijector.exp)` keeps
  * `Sigma` positive.
  */
final class Constraint[Ten <: Tensor[?, ?]] private (
    val site: Site[Ten],
    private[witdraw] val bijector: Bijector[Any, Any],
    private[witdraw] val unconstrainedTree: TensorTree[Any]
)

object Constraint:
  def apply[TZ <: Tuple: Labels, VZ, Ten <: Tensor[?, ?]](
      site: Site[Ten],
      bijector: Bijector[Tensor[TZ, VZ], Ten]
  ): Constraint[Ten] =
    new Constraint(
      site,
      bijector.asInstanceOf[Bijector[Any, Any]],
      TensorTree.tensor[TZ, VZ].asInstanceOf[TensorTree[Any]]
    )

/** The latent sites of a model, found by tracing one forward run, and a
  * parameter type for them.
  *
  * {{{
  * val latent = Latent.discover(Key(0), Constraint(Sigma, Bijector.exp)):
  *   observe(Y := yData)(model(x))
  *
  * val nuts = NUTSAlgorithm[latent.Params](stepSize = 0.05)(latent.logDensity)
  * val draws = nuts.run(latent.init, key).drop(500).take(2000).map(_._1).toList
  * draws.map(_(Sigma))   // List[Sigma.Value], always positive
  * }}}
  *
  * `latent.Params` holds one value per latent site. Inference algorithms see it
  * as a tensor tree in unconstrained space; `params(site)` returns the value in
  * the model's own space.
  */
final class Latent private (
    model: ProbProg[Any],
    slots: Vector[Latent.Slot],
    initial: Vector[Any]
):
  private val indexOf: Map[Site[?], Int] = slots.map(_.site).zipWithIndex.toMap

  /** Values for the latent sites of the model. */
  final class Params private[Latent] (
      private[Latent] val unconstrained: Vector[Any]
  ):
    /** The value of `site`, in the model's own (constrained) space. */
    def apply[Ten <: Tensor[?, ?]](site: Site[Ten]): Ten =
      val slot = slotOf(site)
      slots(slot).bijector.forward(unconstrained(slot)).asInstanceOf[Ten]

    override def toString: String =
      slots
        .map(s => s"${s.site.name} = ${apply(s.site)}")
        .mkString("Params(", ", ", ")")

  object Params:
    given TensorTree[Params] = ParamsTree
    given TreeOf[Params, Float32] = new TreeOf[Params, Float32] {}

  /** The latent sites, in the order in which the model samples them. */
  def sites: Seq[Site[?]] = slots.map(_.site)

  /** Starting values for inference: the values of the traced run. */
  def init: Params = initWith()

  /** Starting values for inference: the given values for some sites, e.g.
    * `latent.initWith(Sigma := Tensor0(1f))`, and the values of the traced run
    * for the others.
    *
    * Every value must lie inside its constraint, e.g. be positive for
    * `Bijector.exp` or increasing for `Bijector.ordered`. A traced value can
    * lie outside when the constraint is stricter than the model, and then the
    * site needs a given value.
    */
  def initWith(values: Observation[?]*): Params =
    val start = values.foldLeft(initial): (acc, o) =>
      val slot = slotOf(o.site)
      acc.updated(slot, slots(slot).bijector.inverse(o.value))
    val overridden = values.map(_.site).toSet
    for i <- slots.indices if !slots(i).isFinite(start(i)) do
      val site = slots(i).site
      val origin =
        if overridden(site) then "The given value" else "The traced value"
      throw IllegalArgumentException(
        s"$origin of '${site.name}' lies outside its constraint (${site.name} = ${slots(i).bijector.forward(start(i))}). " +
          s"Give a starting value inside it with latent.initWith(${site.name} := ...)."
      )
    new Params(start)

  /** The log-density to sample from, e.g. with NUTS: the density of the
    * unconstrained values, which is [[modelLogDensity]] plus the log-Jacobian
    * of the constraints.
    */
  def logDensity(params: Params): Tensor0[Float32] =
    val zs = params.unconstrained
    val logJacobian = slots.indices
      .map(i => slots(i).bijector.logDetJac(zs(i)))
      .foldLeft(Tensor0(0f))(_ + _)
    modelLogDensity(params) + logJacobian

  /** The log-density of the model, observed on the values `params(site)`.
    * Maximise this one to find the posterior mode: unlike [[logDensity]], its
    * maximum does not depend on the constraints.
    */
  def modelLogDensity(params: Params): Tensor0[Float32] =
    val observations =
      slots.indices.map(i => slots(i).observation(params.unconstrained(i)))
    ModelScorer.run(observe(observations*)(model))

  private def slotOf(site: Site[?]): Int =
    indexOf.getOrElse(
      site,
      throw java.util.NoSuchElementException(
        s"Site '${site.name}' is not a latent site of the model. Latent sites: ${sites.map(_.name).mkString(", ")}"
      )
    )

  /** Params as a tensor tree: one leaf per latent site, as a Python tuple. */
  private object ParamsTree extends TensorTree[Params]:
    type Fn = [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => Tensor[T, V])
    type NamedFn =
      [T <: Tuple, V] => (Labels[T]) ?=> (
          (String, Tensor[T, V]) => Tensor[T, V]
      )
    type PairFn = [T <: Tuple, V] => (Labels[T]) ?=> (
        (Tensor[T, V], Tensor[T, V]) => Tensor[T, V]
    )

    private def build(leaf: (Latent.Slot, Int) => Any): Params =
      new Params(slots.zipWithIndex.map(leaf.tupled))
    private def pathOf(path: String, s: Latent.Slot) =
      if path.isEmpty then s.site.name else s"$path.${s.site.name}"

    def map(p: Params, f: Fn): Params =
      build((s, i) => s.tree.map(p.unconstrained(i), f))
    def mapWithName(p: Params, f: NamedFn, path: String = ""): Params =
      build((s, i) =>
        s.tree.mapWithName(p.unconstrained(i), f, pathOf(path, s))
      )
    def mapLeaves[A](
        p: Params,
        f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => A)
    ): Iterator[A] =
      slots.iterator
        .zip(p.unconstrained)
        .flatMap((s, z) => s.tree.mapLeaves(z, f))
    def foreach(
        p: Params,
        f: [T <: Tuple, V] => (Labels[T]) ?=> (Tensor[T, V] => Unit)
    ): Unit =
      slots.zip(p.unconstrained).foreach((s, z) => s.tree.foreach(z, f))
    def foreachWithName(
        p: Params,
        f: [T <: Tuple, V] => (Labels[T]) ?=> ((String, Tensor[T, V]) => Unit),
        path: String = ""
    ): Unit =
      slots
        .zip(p.unconstrained)
        .foreach((s, z) => s.tree.foreachWithName(z, f, pathOf(path, s)))
    def zipMap(p1: Params, p2: Params, f: PairFn): Params =
      build((s, i) =>
        s.tree.zipMap(p1.unconstrained(i), p2.unconstrained(i), f)
      )

    def toPyTree(p: Params): Jax.PyAny =
      py.Dynamic.global.tuple(
        slots
          .zip(p.unconstrained)
          .map((s, z) => s.tree.toPyTree(z))
          .toPythonProxy
      )
    def fromPyTree(pyVal: Jax.PyAny): Params =
      build((s, i) => s.tree.fromPyTree(pyVal.as[py.Dynamic].bracketAccess(i)))
    def toNumpyTree(p: Params): Jax.PyAny =
      py.Dynamic.global.tuple(
        slots
          .zip(p.unconstrained)
          .map((s, z) => s.tree.toNumpyTree(z))
          .toPythonProxy
      )
    def fromNumpyTree(pyVal: Jax.PyAny): Params =
      build((s, i) =>
        s.tree.fromNumpyTree(pyVal.as[py.Dynamic].bracketAccess(i))
      )

object Latent:

  /** Runs `model` forward once with `key`, and collects its latent sites: the
    * sites that are not observed. The values of that run are the default
    * starting point, [[Latent.init]].
    *
    * The parameters are meant for gradient-based inference, so every latent
    * site must be continuous (`Float32`); a discrete latent site is an error.
    * `constraints` gives a change of variables for some of the latent sites.
    */
  def discover(key: Random.Key, constraints: Constraint[?]*)(
      model: ProbProg[Any]
  ): Latent =
    val (_, tr) = ForwardSampler.run(key)(withTrace(model))
    val bySite: Map[Site[?], Constraint[?]] =
      constraints.map(c => c.site -> c).toMap
    require(
      bySite.size == constraints.size,
      "A site has more than one constraint."
    )
    val notLatent = bySite.keySet -- tr.latentSites
    require(
      notLatent.isEmpty,
      s"Constraints for ${notLatent.map(s => s"'${s.name}'").mkString(", ")}, which are not latent sites of the model. " +
        s"Latent sites: ${tr.latentSites.map(_.name).mkString(", ")}"
    )
    val notContinuous = tr.latentSites.filter(_.dtype != DType.Float32)
    require(
      notContinuous.isEmpty,
      s"Latent sites must be continuous (Float32), but ${notContinuous.map(s => s"'${s.name}' is ${s.dtype}").mkString(", ")}. " +
        "Observe these sites, or marginalise them out of the model."
    )
    val slots = tr.latentSites.toVector.map: site =>
      bySite.get(site) match
        case Some(c) => Slot(site, c.bijector, c.unconstrainedTree)
        case None    => Slot(site, Bijector.identityErased, treeOf(site))
    val initial = slots.map(s => s.bijector.inverse(tr(s.site)))
    new Latent(model, slots, initial)

  /** One latent site: how to map its unconstrained value to the model's space,
    * and how to treat the unconstrained value as a tensor tree.
    */
  private[witdraw] final case class Slot(
      site: Site[?],
      bijector: Bijector[Any, Any],
      tree: TensorTree[Any]
  ):
    /** Whether every element of the unconstrained value is finite. */
    def isFinite(z: Any): Boolean =
      tree
        .mapLeaves(
          z,
          [T <: Tuple, V] =>
            (labels: Labels[T]) ?=>
              (t: Tensor[T, V]) =>
                java.lang.Float
                  .isFinite(t.asInstanceOf[Tensor[T, Float32]].sum.item)
        )
        .forall(identity)

    def observation(z: Any): Observation[?] =
      Observation(
        site.asInstanceOf[Site[Tensor[Tuple, Any]]],
        bijector.forward(z).asInstanceOf[Tensor[Tuple, Any]]
      )

  private def treeOf[Ten <: Tensor[?, ?]](site: Site[Ten]): TensorTree[Any] =
    TensorTree
      .tensor[ShapeOf[Ten], ElemOf[Ten]](using site.labels)
      .asInstanceOf[TensorTree[Any]]
