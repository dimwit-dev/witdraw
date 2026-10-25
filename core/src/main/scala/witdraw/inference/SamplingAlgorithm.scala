package witdraw
package inference

import dimwit.*
import dimwit.random.Random.Key
import dimwit.tensortree.TensorTree
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

/** The state carried between steps. Every state must expose its position. */
trait KernelState[P]:
  def position: P

/** An inference method that produces a sequence of draws, such as [[NUTS]] or
  * [[Laplace]]: `init` builds the state from a starting position, and `step`
  * moves it on.
  */
case class SamplingAlgorithm[P, S <: KernelState[P], I](
    init: P => S,
    step: (Key, S) => (S, I)
):
  /** Infinite iterator of `(position, info)` pairs, driven by key splitting.
    * Each step is a separate call; [[sample]] runs many steps at once.
    */
  def run(initial: P, key: Key): Iterator[(P, I)] =
    Iterator.unfold((init(initial), key)) { case (state, rng) =>
      val (k1, k2) = rng.split2()
      val (next, info) = step(k1, state)
      Some(((next.position, info), (next, k2)))
    }

  /** The first `numDraws` `(position, info)` pairs of [[run]], computed in one
    * compiled loop (`scan`), which is much faster than stepping one by one.
    * `step` must therefore be traceable: it may not read values with `.item`.
    *
    * The loop is compiled on the first call, and reused by later calls with
    * the same `numDraws`, e.g. for further chains.
    */
  def sample(initial: P, key: Key, numDraws: Int)(using
      TensorTree[P],
      TensorTree[S],
      TensorTree[I]
  ): Seq[(P, I)] =
    require(numDraws > 0, s"numDraws must be positive, but is $numDraws")
    // the first step runs on its own: it gives the structure of a draw
    val (k1, k2) = key.split2()
    val (first, firstInfo) = step(k1, init(initial))
    if numDraws == 1 then Seq((first.position, firstInfo))
    else
      val steps = Tensor(Shape(Axis[Draw] -> (numDraws - 1))).fill(0)
      val flats = compiledLoop(first, k2, steps)
      (first.position, firstInfo) +: unflattenAll((first.position, firstInfo), flats)

  /** `numDraws` draws of each of several chains, one chain per starting
    * position in `initials`, all run in one compiled loop. The chains are
    * independent: each gets its own key, split from `key`. As in [[sample]],
    * `step` must be traceable.
    *
    * Running several chains, from different starting positions, makes it
    * possible to check whether they agree, e.g. with [[Diagnostics.rhat]].
    * They run side by side (`vmap`), so a few chains take little longer than
    * one. For NUTS, though, every step lasts as long as the longest
    * trajectory among the chains.
    */
  def sampleChains(initials: Seq[P], key: Key, numDraws: Int)(using
      TensorTree[P],
      TensorTree[S],
      TensorTree[I]
  ): Seq[Seq[(P, I)]] =
    val loop = loopCache.synchronized:
      loopCache
        .getOrElseUpdate("chains", SamplingAlgorithm.chainsLoop[P, S, I](_ => this))
        .asInstanceOf[py.Dynamic]
    SamplingAlgorithm.sampleChains(loop, initials, Seq.empty, key, numDraws)

  /** The loop of [[sample]] from a state, compiled once. scan's outputs must
    * be tensors, so each draw is flattened into a vector.
    */
  private def compiledLoop(using
      TensorTree[P],
      TensorTree[S],
      TensorTree[I]
  ): (S, Key, Tensor1[Draw, Int32]) => Tensor2[Draw, DrawElement, Float32] =
    loopCache.synchronized:
      loopCache
        .getOrElseUpdate(
          "loop",
          jit((start: S, rng: Key, steps: Tensor1[Draw, Int32]) =>
            scan(Axis[Draw])((start, rng), steps) { case ((state, rng), _) =>
              val (k1, k2) = rng.split2()
              val (next, info) = step(k1, state)
              val draw = (next.position, info)
              ((next, k2), TensorTree.ravel(draw, Axis[DrawElement]).flatten(draw))
            }._2
          )
        )
        .asInstanceOf[(S, Key, Tensor1[Draw, Int32]) => Tensor2[Draw, DrawElement, Float32]]

  private val loopCache = scala.collection.mutable.Map.empty[String, AnyRef]

  /** The draws in the rows of `flats`, with the structure of `reference`.
    * They are unflattened together, and split into draws in a few Python
    * calls: slicing them out one by one from Scala is much slower.
    */
  private def unflattenAll(
      reference: (P, I),
      flats: Tensor2[Draw, DrawElement, Float32]
  )(using tree: TensorTree[(P, I)]): Seq[(P, I)] =
    val referencePy = tree.toPyTree(reference)
    val unravel = py.module("jax.flatten_util").ravel_pytree(referencePy).bracketAccess(1)
    val flatsPy = TensorTree[Tensor2[Draw, DrawElement, Float32]].toPyTree(flats)
    val stacked = py.module("jax").vmap(unravel)(flatsPy)
    SamplingAlgorithm.unstack(stacked, flats.shape(Axis[Draw])).map(tree.fromPyTree(_))

object SamplingAlgorithm:

  /** A compiled function that runs one chain per row of its inputs, side by
    * side (`vmap`). It takes the starting positions, the keys and the
    * settings of the chains, each stacked along a leading axis, and a vector
    * with one entry per draw after the first; it returns the draws, as a tree
    * whose leaves have a chain and a draw axis in front.
    *
    * `build` makes the algorithm of a chain from its settings, e.g. its step
    * size; without settings it is given `None`.
    */
  private[witdraw] def chainsLoop[P, S <: KernelState[P], I](
      build: py.Dynamic => SamplingAlgorithm[P, S, I]
  )(using
      pTree: TensorTree[P],
      sTree: TensorTree[S],
      iTree: TensorTree[I]
  ): py.Dynamic =
    val jax = py.module("jax")
    val helper = dimwit.jax.Jax.jax_helper
    val chain = (start: py.Dynamic, key: py.Dynamic, settings: py.Dynamic, steps: py.Dynamic) =>
      val algorithm = build(settings)
      val (k1, k2) = Key(key).split2()
      // the first step runs on its own, so that the loop's state has the
      // types of a stepped state
      val (first, firstInfo) = algorithm.step(k1, algorithm.init(pTree.fromPyTree(start)))
      val body = (carry: py.Dynamic, _: py.Dynamic) =>
        val state = sTree.fromPyTree(carry.bracketAccess(0))
        val (k1, k2) = Key(carry.bracketAccess(1)).split2()
        val (next, info) = algorithm.step(k1, state)
        py.Dynamic.global.tuple(
          Seq(
            py.Dynamic.global.tuple(Seq(sTree.toPyTree(next), k2.jaxKey).toPythonProxy),
            drawToPy(next.position, info)
          ).toPythonProxy
        )
      val init = py.Dynamic.global.tuple(Seq(sTree.toPyTree(first), k2.jaxKey).toPythonProxy)
      val rest = helper.scan(body, init, steps).bracketAccess(1)
      val prepend = (head: py.Dynamic, tail: py.Dynamic) =>
        py.module("jax.numpy").concatenate(Seq(head.bracketAccess(py.None), tail).toPythonProxy)
      py.module("jax.tree_util").tree_map(
        helper.wrap_fn(prepend),
        drawToPy(first.position, firstInfo),
        rest
      )
    val inAxes = py.Dynamic.global.tuple(Seq[py.Any](py.Any.from(0), py.Any.from(0), py.Any.from(0), py.None).toPythonProxy)
    jax.jit(jax.vmap(helper.wrap_fn(chain), in_axes = inAxes))

  private def drawToPy[P, I](position: P, info: I)(using
      pTree: TensorTree[P],
      iTree: TensorTree[I]
  ): py.Dynamic =
    py.Dynamic.global.tuple(Seq(pTree.toPyTree(position), iTree.toPyTree(info)).toPythonProxy)

  /** Runs `loop`, from [[chainsLoop]], for one chain per starting position,
    * with the given settings per chain (none, or one per chain).
    */
  private[witdraw] def sampleChains[P, I](
      loop: py.Dynamic,
      initials: Seq[P],
      settings: Seq[py.Any],
      key: Key,
      numDraws: Int
  )(using pTree: TensorTree[P], iTree: TensorTree[I]): Seq[Seq[(P, I)]] =
    require(initials.nonEmpty, "sampleChains needs at least one starting position")
    require(numDraws > 0, s"numDraws must be positive, but is $numDraws")
    require(
      settings.isEmpty || settings.size == initials.size,
      s"${settings.size} settings for ${initials.size} chains"
    )
    val numChains = initials.size
    val keys = py.module("jax.random").split(key.jaxKey, numChains)
    val starts = stack(initials.map(pTree.toPyTree(_)))
    val stackedSettings = if settings.isEmpty then py.None else stack(settings)
    val steps = py.module("jax.numpy").zeros(numDraws - 1)
    val draws = loop(starts, keys, stackedSettings, steps)
    val treeUtil = py.module("jax.tree_util")
    val itemgetter = py.module("operator").itemgetter
    (0 until numChains).map: c =>
      val chain = treeUtil.tree_map(itemgetter(c), draws)
      unstack(chain, numDraws).map: d =>
        (pTree.fromPyTree(d.bracketAccess(0)), iTree.fromPyTree(d.bracketAccess(1)))

  /** Trees of the same structure, stacked leaf by leaf along a new leading
    * axis.
    */
  private[witdraw] def stack(trees: Seq[py.Any]): py.Dynamic =
    val treeUtil = py.module("jax.tree_util")
    val jnp = py.module("jax.numpy")
    val flat = trees.map(t => treeUtil.tree_flatten(t))
    val structure = flat.head.bracketAccess(1)
    for f <- flat.tail do
      require(
        py.module("operator").applyDynamic("eq")(f.bracketAccess(1), structure).as[Boolean],
        s"all chains must have the same structure, but got ${f.bracketAccess(1)} and $structure"
      )
    val leaves = flat.map(_.bracketAccess(0).as[Seq[py.Dynamic]])
    val stacked = leaves.transpose.map(column => jnp.stack(column.toPythonProxy))
    treeUtil.tree_unflatten(structure, stacked.toPythonProxy)

  /** The `n` trees in the rows of `stacked`, a tree whose leaves have a
    * leading axis of length `n`, split in a few Python calls.
    */
  private[witdraw] def unstack(stacked: py.Dynamic, n: Int): Seq[py.Dynamic] =
    val treeUtil = py.module("jax.tree_util")
    val lists = treeUtil.tree_map(py.Dynamic.global.list, stacked)
    val draws = treeUtil.tree_transpose(
      treeUtil.tree_structure(stacked),
      treeUtil.tree_structure(py.Dynamic.global.list(py.Dynamic.global.range(n))),
      lists
    )
    draws.as[Seq[py.Dynamic]]

/** The axis of the draws in [[SamplingAlgorithm.sample]]. */
private trait Draw derives Label

/** The axis of one flattened draw in [[SamplingAlgorithm.sample]]. */
private trait DrawElement derives Label
