package witdraw

import dimwit.*
import dimwit.tensor.Labels

/** A differentiable bijection between an unconstrained value `z` and a value
  * `x` in a constrained space, with the log-determinant of its Jacobian.
  * [[Constraint]] attaches one to a site.
  *
  * Inference moves in `z`, so the density of `z` is
  * `logDensity(forward(z)) + logDetJac(z)`, see [[Latent.logDensity]].
  */
trait Bijector[Z, X]:
  /** The unconstrained value `z` to the constrained value `x`. */
  def forward(z: Z): X

  /** The constrained value `x` to the unconstrained value `z`. */
  def inverse(x: X): Z

  /** `log |det ∂forward/∂z|` at `z`. */
  def logDetJac(z: Z): Tensor0[Float32]

object Bijector:

  /** `x = z`: no constraint. */
  def identity[T <: Tuple, V]: Bijector[Tensor[T, V], Tensor[T, V]] =
    new Bijector[Tensor[T, V], Tensor[T, V]]:
      def forward(z: Tensor[T, V]) = z
      def inverse(x: Tensor[T, V]) = x
      def logDetJac(z: Tensor[T, V]) = Tensor0(0f)

  /** `x = exp(z)`, so `x > 0` — Stan's `<lower=0>`. */
  def exp[T <: Tuple: Labels]
      : Bijector[Tensor[T, Float32], Tensor[T, Float32]] =
    new Bijector[Tensor[T, Float32], Tensor[T, Float32]]:
      def forward(z: Tensor[T, Float32]) = z.exp
      def inverse(x: Tensor[T, Float32]) = x.log
      def logDetJac(z: Tensor[T, Float32]) = z.sum // ∑ log e^{z_i} = ∑ z_i

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
        lo + width * Tensor.sigmoid(z)
      def inverse(x: Tensor[T, Float32]) =
        val u = (x - lo) / width
        (u / -(u -! Tensor0(1f))).log // logit(u) = log(u / (1 - u))
      def logDetJac(z: Tensor[T, Float32]) =
        val s = Tensor.sigmoid(z)
        (width.log + (s * -(s -! Tensor0(1f))).log).sum // ∑ log[width·σ(1-σ)]

  /** Strictly increasing along the axis `L` — Stan's `ordered[k]`.
    * `x_0 = z_0`, `x_j = x_{j-1} + exp(z_j)`, i.e. `x = cumsum([z_0, exp(z_1),
    * …, exp(z_{k-1})])`.
    */
  def ordered[L: Label]: Bijector[Tensor1[L, Float32], Tensor1[L, Float32]] =
    val axis = Axis[L]
    new Bijector[Tensor1[L, Float32], Tensor1[L, Float32]]:
      def forward(z: Tensor1[L, Float32]) =
        val k = z.shape(axis)
        if k <= 1 then z
        else
          val incrs = concatenate(
            Seq(z.slice(axis.at(0 until 1)), z.slice(axis.at(1 until k)).exp),
            axis
          )
          incrs.cumsum(axis)
      def inverse(x: Tensor1[L, Float32]) =
        val k = x.shape(axis)
        if k <= 1 then x
        else
          val gaps =
            x.slice(axis.at(1 until k)) - x.slice(axis.at(0 until k - 1))
          concatenate(Seq(x.slice(axis.at(0 until 1)), gaps.log), axis)
      def logDetJac(z: Tensor1[L, Float32]) =
        val k = z.shape(axis)
        if k <= 1 then Tensor0(0f)
        else z.slice(axis.at(1 until k)).sum // ∑_{j≥1} z_j

  /** Stick-breaking simplex map — Stan's `simplex[k]`. `Z` is `k-1`
    * unconstrained reals (`LZ` axis), `X` is a `k`-vector on the simplex (`LC`
    * axis).
    */
  def simplexStickBreaking[LZ: Label, LC: Label]
      : Bijector[Tensor1[LZ, Float32], Tensor1[LC, Float32]] =
    val zAxis = Axis[LZ]
    val cAxis = Axis[LC]
    // logit shift, so that z = 0 gives the uniform simplex
    def offset(k: Int, i: Int): Tensor0[Float32] =
      Tensor0(math.log(1.0 / (k - 1 - i)).toFloat)

    new Bijector[Tensor1[LZ, Float32], Tensor1[LC, Float32]]:
      def forward(z: Tensor1[LZ, Float32]) =
        val k = z.shape(zAxis) + 1
        val xs = scala.collection.mutable.ArrayBuffer[Tensor0[Float32]]()
        var remaining = Tensor0(1f)
        var i = 0
        while i < k - 1 do
          val zeta =
            Tensor.sigmoid(z.slice(zAxis.at(i)) + offset(k, i))
          val xi = remaining * zeta
          xs += xi
          remaining = remaining - xi
          i += 1
        xs += remaining
        stack(xs.toSeq, cAxis)

      def inverse(x: Tensor1[LC, Float32]) =
        val k = x.shape(cAxis)
        val zs = scala.collection.mutable.ArrayBuffer[Tensor0[Float32]]()
        var remaining = Tensor0(1f)
        var i = 0
        while i < k - 1 do
          val xi = x.slice(cAxis.at(i))
          val zeta = xi / remaining
          zs += (zeta / (Tensor0(1f) - zeta)).log - offset(k, i) // logit(zeta) - shift
          remaining = remaining - xi
          i += 1
        stack(zs.toSeq, zAxis)

      def logDetJac(z: Tensor1[LZ, Float32]) =
        val k = z.shape(zAxis) + 1
        var acc = Tensor0(0f)
        var remaining = Tensor0(1f)
        var i = 0
        while i < k - 1 do
          val zeta =
            Tensor.sigmoid(z.slice(zAxis.at(i)) + offset(k, i))
          // ∑ [ log ζ_i + log(1-ζ_i) + log(stick remaining before i) ]
          acc = acc + (zeta * (Tensor0(1f) - zeta)).log + remaining.log
          remaining = remaining - remaining * zeta
          i += 1
        acc

  /** Applies a vector bijector to every row of a matrix, e.g. a stick-breaking
    * simplex per row for a row-stochastic transition matrix (Stan's
    * `array[K] simplex[K]`). The log-det-Jacobian is the sum over the rows.
    */
  def rowwise[R: Label, LZ: Label, LX: Label](
      row: Bijector[Tensor1[LZ, Float32], Tensor1[LX, Float32]]
  ): Bijector[Tensor2[R, LZ, Float32], Tensor2[R, LX, Float32]] =
    new Bijector[Tensor2[R, LZ, Float32], Tensor2[R, LX, Float32]]:
      def forward(z: Tensor2[R, LZ, Float32]) = z.vmap(Axis[R])(row.forward)
      def inverse(x: Tensor2[R, LX, Float32]) = x.vmap(Axis[R])(row.inverse)
      def logDetJac(z: Tensor2[R, LZ, Float32]) =
        z.vmap(Axis[R])(row.logDetJac).sum
