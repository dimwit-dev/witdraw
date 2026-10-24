package witdraw
package inference

import dimwit.*
import dimwit.random.Random.Key
import dimwit.tensortree.TensorTree
import me.shadaj.scalapy.py

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
    val jax = py.module("jax")
    val treeUtil = jax.tree_util
    val referencePy = tree.toPyTree(reference)
    val unravel = py.module("jax.flatten_util").ravel_pytree(referencePy).bracketAccess(1)
    val flatsPy = TensorTree[Tensor2[Draw, DrawElement, Float32]].toPyTree(flats)
    // a tree of stacked leaves, then a tree of lists of rows, then a list of trees
    val stacked = jax.vmap(unravel)(flatsPy)
    val lists = treeUtil.tree_map(py.Dynamic.global.list, stacked)
    val numRows = flats.shape(Axis[Draw])
    val draws = treeUtil.tree_transpose(
      treeUtil.tree_structure(referencePy),
      treeUtil.tree_structure(py.Dynamic.global.list(py.Dynamic.global.range(numRows))),
      lists
    )
    draws.as[Seq[py.Dynamic]].map(tree.fromPyTree(_))

/** The axis of the draws in [[SamplingAlgorithm.sample]]. */
private trait Draw derives Label

/** The axis of one flattened draw in [[SamplingAlgorithm.sample]]. */
private trait DrawElement derives Label
