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
  *
  * To run several chains, run one per starting position, e.g. each in a
  * `Future`, as shown at [[Chain]].
  */
final class SamplingAlgorithm[P, S <: KernelState[P], I] private (
    val init: P => S,
    val step: (Key, S) => (S, I),
    loops: SamplingAlgorithm.Loops
):
  /** This algorithm, starting with `init` instead. It shares the compiled loop
    * of [[sample]] with this one, since the loop depends only on `step`.
    */
  def withInit(init: P => S): SamplingAlgorithm[P, S, I] =
    new SamplingAlgorithm(init, step, loops)

  /** Infinite iterator of `(position, info)` pairs, driven by key splitting,
    * e.g. to step through a chain one draw at a time. Each step is a separate
    * call; [[sample]] runs many steps at once, and is much faster.
    */
  def iterate(key: Key, initial: P): Iterator[(P, I)] =
    Iterator.unfold((init(initial), key)) { case (state, rng) =>
      val (k1, k2) = rng.split2()
      val (next, info) = step(k1, state)
      Some(((next.position, info), (next, k2)))
    }

  /** The chain of the first `numDraws` `(position, info)` pairs of
    * [[iterate]], computed in one compiled loop (`scan`), which is much faster
    * than stepping one by one.
    * `step` must therefore be traceable: it may not read values with `.item`.
    *
    * The loop is compiled on the first call, and reused by later calls, also
    * by those of algorithms made with [[withInit]], e.g. for further chains.
    * Whatever differs between the chains must therefore be in the state.
    */
  def sample(key: Key, initial: P, numDraws: Int)(using
      TensorTree[P],
      TensorTree[S],
      TensorTree[I]
  ): Chain[P, I] =
    require(numDraws > 0, s"numDraws must be positive, but is $numDraws")
    // the first step runs on its own: it gives the structure of a draw
    val (k1, k2) = key.split2()
    val (first, firstInfo) = step(k1, init(initial))
    if numDraws == 1 then Chain(Seq((first.position, firstInfo)))
    else
      val steps = Tensor(Shape(Axis[Draw] -> (numDraws - 1))).fill(0)
      val flats = compiledLoop(first, k2, steps)
      Chain((first.position, firstInfo) +: unflattenAll((first.position, firstInfo), flats))

  /** The loop of [[sample]] from a state, compiled once. scan's outputs must
    * be tensors, so each draw is flattened into a vector.
    */
  private def compiledLoop(using
      TensorTree[P],
      TensorTree[S],
      TensorTree[I]
  ): (S, Key, Tensor1[Draw, Int32]) => Tensor2[Draw, DrawElement, Float32] =
    loops.synchronized:
      loops.cache
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

  /** The algorithm with the given `init` and `step`. */
  def apply[P, S <: KernelState[P], I](
      init: P => S,
      step: (Key, S) => (S, I)
  ): SamplingAlgorithm[P, S, I] =
    new SamplingAlgorithm(init, step, Loops())

  /** The algorithm with the given `init` and `step`, which compiles its loop
    * into `loops`. Algorithms with the same `step` can share them.
    */
  private[inference] def sharing[P, S <: KernelState[P], I](
      init: P => S,
      step: (Key, S) => (S, I),
      loops: Loops
  ): SamplingAlgorithm[P, S, I] =
    new SamplingAlgorithm(init, step, loops)

  /** The compiled loops of an algorithm, shared with those made by
    * [[SamplingAlgorithm.withInit]].
    */
  private[inference] final class Loops:
    val cache = scala.collection.mutable.Map.empty[String, AnyRef]

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
