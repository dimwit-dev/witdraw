package witdraw.vi

import dimwit.*

/** Low-level variational inference algorithm.
  *
  * Mirrors BlackJax's lower-level VI API. The log-density and optimizer are
  * passed explicitly at each `step` call, keeping the interface open for
  * adaptive schemes.
  *
  * {{{
  * val algo: VIAlgorithm[P, S, Opt, I] = ...
  * val state             = algo.init(position, optimizer)
  * val (next, info)      = algo.step(key, state, logDensity, optimizer)
  * val samples: List[P]  = algo.sample(key, next, numSamples)
  * }}}
  */
trait VIAlgorithm[P, S, Opt, I]:
  def init(position: P, optimizer: Opt): S
  def step(
      key: Key,
      state: S,
      logDensity: P => Tensor0[Float],
      optimizer: Opt
  ): (S, I)
  def sample(key: Key, state: S, numSamples: Int): List[P]
