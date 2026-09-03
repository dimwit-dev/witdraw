package witdraw.reparam

import dimwit.*
import dimwit.tensor.Labels
import dimwit.nn.ActivationFunctions

/** A differentiable bijection `Z <-> X` with a tractable log-det-Jacobian.
  *
  * A *constraining transform* (this file's factories) is the fixed,
  * zero-parameter case: `Z` is an unconstrained ℝ-valued tensor and `X` lives
  * in some constrained support. A *normalizing flow* is the same interface
  * with learnable parameters and many layers composed via [[andThen]].
  *
  * Convention: we sample in `Z`-space, so the pushed-forward target density is
  * `p_X(forward(z)) * |det J_forward(z)|`, i.e.
  * `logDensity(forward(z)) + logDetJac(z)` — see `Reparam.pushForward`.
  */
trait Bijector[Z, X]:

  /** unconstrained/base value -> value in the constrained support */
  def forward(z: Z): X

  /** constrained value -> unconstrained/base value */
  def inverse(x: X): Z

  /** `log |det ∂forward/∂z|` evaluated at `z`, as a scalar. */
  def logDetJac(z: Z): Tensor0[Float32]

  /** Compose: `this` then `that`. Jacobians add along the chain. */
  infix def andThen[Y](that: Bijector[X, Y]): Bijector[Z, Y] =
    val self = this
    new Bijector[Z, Y]:
      def forward(z: Z): Y = that.forward(self.forward(z))
      def inverse(y: Y): Z = self.inverse(that.inverse(y))
      def logDetJac(z: Z): Tensor0[Float32] =
        self.logDetJac(z) + that.logDetJac(self.forward(z))

object Bijector:

  /** Untyped identity — used by `Reparam` for fields with no declared
    * constraint.
    */
  private[reparam] val identityErased: Bijector[Any, Any] =
    new Bijector[Any, Any]:
      def forward(z: Any): Any = z
      def inverse(x: Any): Any = x
      def logDetJac(z: Any): Tensor0[Float32] = Tensor0(0f)

  /** `x = z` (an unconstrained real parameter). */
  def identity[T <: Tuple: Labels]
      : Bijector[Tensor[T, Float32], Tensor[T, Float32]] =
    new Bijector[Tensor[T, Float32], Tensor[T, Float32]]:
      def forward(z: Tensor[T, Float32]) = z
      def inverse(x: Tensor[T, Float32]) = x
      def logDetJac(z: Tensor[T, Float32]) = Tensor0(0f)

  /** `x = exp(z)`, so `x > 0` — Stan's `<lower=0>`. */
  def exp[T <: Tuple: Labels]
      : Bijector[Tensor[T, Float32], Tensor[T, Float32]] =
    new Bijector[Tensor[T, Float32], Tensor[T, Float32]]:
      def forward(z: Tensor[T, Float32]) = z.exp
      def inverse(x: Tensor[T, Float32]) = x.log
      def logDetJac(z: Tensor[T, Float32]) = z.sum // ∑ log e^{z_i} = ∑ z_i

  /** `x = shift + scale ⊙ z` (element-wise; `scale > 0`). */
  def affine[T <: Tuple: Labels](
      shift: Tensor[T, Float32],
      scale: Tensor[T, Float32]
  ): Bijector[Tensor[T, Float32], Tensor[T, Float32]] =
    new Bijector[Tensor[T, Float32], Tensor[T, Float32]]:
      def forward(z: Tensor[T, Float32]) = shift + scale * z
      def inverse(x: Tensor[T, Float32]) = (x - shift) / scale
      def logDetJac(z: Tensor[T, Float32]) = scale.log.sum

  /** `x = lo + (hi - lo)·σ(z)`, so `x ∈ (lo, hi)` element-wise — Stan's
    * `<lower=a, upper=b>`. `lo`/`hi` are full tensors (broadcast yourself for
    * scalar bounds).
    */
  def interval[T <: Tuple: Labels](
      lo: Tensor[T, Float32],
      hi: Tensor[T, Float32]
  ): Bijector[Tensor[T, Float32], Tensor[T, Float32]] =
    val width = hi - lo
    new Bijector[Tensor[T, Float32], Tensor[T, Float32]]:
      def forward(z: Tensor[T, Float32]) =
        lo + width * ActivationFunctions.sigmoid(z)
      def inverse(x: Tensor[T, Float32]) =
        val u = (x - lo) / width
        (u / -(u -! Tensor0(1f))).log // logit(u) = log(u / (1 - u))
      def logDetJac(z: Tensor[T, Float32]) =
        val s = ActivationFunctions.sigmoid(z)
        (width.log + (s * -(s -! Tensor0(1f))).log).sum // ∑ log[width·σ(1-σ)]

  /** Strictly increasing along `axis` — Stan's `ordered[k]`.
    * `x_0 = z_0`, `x_j = x_{j-1} + exp(z_j)`.
    */
  def ordered[L: Label](
      k: Int
  ): Bijector[Tensor1[L, Float32], Tensor1[L, Float32]] =
    val axis = Axis[L]
    new Bijector[Tensor1[L, Float32], Tensor1[L, Float32]]:
      def forward(z: Tensor1[L, Float32]) =
        val out = scala.collection.mutable.ArrayBuffer[Tensor0[Float32]]()
        var acc = z.slice(axis.at(0))
        out += acc
        var j = 1
        while j < k do
          acc = acc + z.slice(axis.at(j)).exp
          out += acc
          j += 1
        stack(out.toSeq, axis)
      def inverse(x: Tensor1[L, Float32]) =
        val out = scala.collection.mutable.ArrayBuffer[Tensor0[Float32]]()
        out += x.slice(axis.at(0))
        var j = 1
        while j < k do
          out += (x.slice(axis.at(j)) - x.slice(axis.at(j - 1))).log
          j += 1
        stack(out.toSeq, axis)
      def logDetJac(z: Tensor1[L, Float32]) =
        if k <= 1 then Tensor0(0f)
        else z.slice(axis.at(1 until k)).sum // ∑_{j≥1} z_j

  /** Stick-breaking simplex map — Stan's `simplex[k]`. `Z` is `k-1`
    * unconstrained reals (`LZ` axis), `X` is a `k`-vector on the simplex
    * (`LC` axis).
    */
  def simplexStickBreaking[LZ: Label, LC: Label](
      k: Int
  ): Bijector[Tensor1[LZ, Float32], Tensor1[LC, Float32]] =
    val zAxis = Axis[LZ]
    val cAxis = Axis[LC]
    def offset(i: Int): Tensor0[Float32] =
      Tensor0(math.log(1.0 / (k - 1 - i)).toFloat) // logit shift -> uniform default

    new Bijector[Tensor1[LZ, Float32], Tensor1[LC, Float32]]:
      def forward(z: Tensor1[LZ, Float32]) =
        val xs = scala.collection.mutable.ArrayBuffer[Tensor0[Float32]]()
        var remaining = Tensor0(1f)
        var i = 0
        while i < k - 1 do
          val zeta = ActivationFunctions.sigmoid(z.slice(zAxis.at(i)) + offset(i))
          val xi = remaining * zeta
          xs += xi
          remaining = remaining - xi
          i += 1
        xs += remaining
        stack(xs.toSeq, cAxis)

      def inverse(x: Tensor1[LC, Float32]) =
        val zs = scala.collection.mutable.ArrayBuffer[Tensor0[Float32]]()
        var remaining = Tensor0(1f)
        var i = 0
        while i < k - 1 do
          val xi = x.slice(cAxis.at(i))
          val zeta = xi / remaining
          zs += (zeta / (Tensor0(1f) - zeta)).log - offset(i) // logit(zeta) - shift
          remaining = remaining - xi
          i += 1
        stack(zs.toSeq, zAxis)

      def logDetJac(z: Tensor1[LZ, Float32]) =
        var acc = Tensor0(0f)
        var remaining = Tensor0(1f)
        var i = 0
        while i < k - 1 do
          val zeta = ActivationFunctions.sigmoid(z.slice(zAxis.at(i)) + offset(i))
          // ∑ [ log ζ_i + log(1-ζ_i) + log(stick remaining before i) ]
          acc = acc + (zeta * (Tensor0(1f) - zeta)).log + remaining.log
          remaining = remaining - remaining * zeta
          i += 1
        acc
