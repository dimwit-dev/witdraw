package witdraw
package inference

import dimwit.random.Random.Key

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
  /** Infinite iterator of `(position, info)` pairs, driven by key splitting. */
  def run(initial: P, key: Key): Iterator[(P, I)] =
    Iterator.unfold((init(initial), key)) { case (state, rng) =>
      val (k1, k2) = rng.split2()
      val (next, info) = step(k1, state)
      Some(((next.position, info), (next, k2)))
    }
