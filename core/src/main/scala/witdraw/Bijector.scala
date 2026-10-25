package witdraw

import dimwit.*
import dimwit.linalg.LinearAlgebra
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

  /** `x = lo + exp(z)`, so `x > lo` element-wise — Stan's `<lower=a>`. */
  def lowerBound[T <: Tuple: Labels](
      lo: Tensor[T, Float32]
  ): Bijector[Tensor[T, Float32], Tensor[T, Float32]] =
    new Bijector[Tensor[T, Float32], Tensor[T, Float32]]:
      def forward(z: Tensor[T, Float32]) = lo + z.exp
      def inverse(x: Tensor[T, Float32]) = (x - lo).log
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

  /** Strictly increasing along the axis `L` — Stan's `ordered[k]`. `x_0 = z_0`,
    * `x_j = x_{j-1} + exp(z_j)`, i.e. `x = cumsum([z_0, exp(z_1), …,
    * exp(z_{k-1})])`.
    */
  def ordered[L: Label]: Bijector[Tensor1[L, Float32], Tensor1[L, Float32]] =
    val axis = Axis[L]
    new Bijector[Tensor1[L, Float32], Tensor1[L, Float32]]:
      def forward(z: Tensor1[L, Float32]) =
        val k = z.shape(axis)
        if k <= 1 then z
        else
          val incrs = concatenate(
            Seq(z.select(axis.at(0 until 1)), z.select(axis.at(1 until k)).exp),
            axis
          )
          incrs.cumsum(axis)
      def inverse(x: Tensor1[L, Float32]) =
        val k = x.shape(axis)
        if k <= 1 then x
        else
          val gaps =
            x.select(axis.at(1 until k)) - x.select(axis.at(0 until k - 1))
          concatenate(Seq(x.select(axis.at(0 until 1)), gaps.log), axis)
      def logDetJac(z: Tensor1[L, Float32]) =
        val k = z.shape(axis)
        if k <= 1 then Tensor0(0f)
        else z.select(axis.at(1 until k)).sum // ∑_{j≥1} z_j

  /** Stick-breaking simplex map — Stan's `simplex[k]`. `Z` is `k-1`
    * unconstrained reals (`LZ` axis), `X` is a `k`-vector on the simplex (`LC`
    * axis).
    *
    * Piece `i` takes the fraction `ζ_i = σ(z_i + offset_i)` of the stick left
    * over by the pieces before it, and the last piece is the rest. The offsets
    * make `z = 0` the uniform simplex.
    */
  def simplexStickBreaking[LZ: Label, LC: Label]
      : Bijector[Tensor1[LZ, Float32], Tensor1[LC, Float32]] =
    val zAxis = Axis[LZ]
    val cAxis = Axis[LC]
    val one = Tensor0(1f)

    // offset_i = log(1 / (k - 1 - i)), for the n = k - 1 coordinates of z
    def offsets(n: Int) =
      Tensor1(zAxis).fromArray(
        Array.tabulate(n)(i => math.log(1.0 / (n - i)).toFloat)
      )

    // ζ, and the stick left before each piece: 1, (1-ζ_0), (1-ζ_0)(1-ζ_1), …
    def fractions(z: Tensor1[LZ, Float32]) =
      val n = z.shape(zAxis)
      val zeta = Tensor.sigmoid(z + offsets(n))
      val leftAfter = (one -! zeta).cumprod(zAxis)
      val leftBefore = concatenate(
        Seq(
          Tensor1(zAxis -> 1).fill(1f),
          leftAfter.select(zAxis.at(0 until n - 1))
        ),
        zAxis
      )
      (zeta, leftBefore, leftAfter)

    new Bijector[Tensor1[LZ, Float32], Tensor1[LC, Float32]]:
      def forward(z: Tensor1[LZ, Float32]) =
        val n = z.shape(zAxis)
        val (zeta, leftBefore, leftAfter) = fractions(z)
        concatenate(
          Seq(leftBefore * zeta, leftAfter.select(zAxis.at(n - 1 until n)))
            .map(_.relabelTo(cAxis)),
          cAxis
        )

      def inverse(x: Tensor1[LC, Float32]) =
        val n = x.shape(cAxis) - 1
        val pieces = x.select(cAxis.at(0 until n)).relabelTo(zAxis)
        val leftBefore = (one -! pieces.cumsum(zAxis)) + pieces
        val zeta = pieces / leftBefore
        (zeta / (one -! zeta)).log - offsets(n) // logit(ζ) - offset

      def logDetJac(z: Tensor1[LZ, Float32]) =
        // ∑ [ log ζ_i + log(1 - ζ_i) + log(stick left before i) ]
        val (zeta, leftBefore, _) = fractions(z)
        ((zeta * (one -! zeta)).log + leftBefore.log).sum

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

  /** Correlation matrices — Stan's `corr_matrix[d]`. `Z` is the `d(d-1)/2`
    * unconstrained reals (`LZ` axis), one per entry below the diagonal, in row
    * order: `(1,0), (2,0), (2,1), (3,0), …`.
    *
    * Each `z` becomes a canonical partial correlation `c = tanh(z)` in (-1, 1).
    * Row `i` of the Cholesky factor `w` takes the fraction `c_ij` of the length
    * it has left: `w_ij = c_ij · sqrt(1 - Σ_{k<j} w_ik²)`, and the diagonal is
    * the rest, so every row has unit length. Then `x = w wᵀ`.
    */
  def correlationMatrix[LZ: Label, L: Label]
      : Bijector[Tensor1[LZ, Float32], Tensor2[L, Prime[L], Float32]] =
    val zAxis = Axis[LZ]
    val one = Tensor0(1f)
    val half = Tensor0(0.5f)

    // the size d of the matrix, from the number n = d(d-1)/2 of entries
    def size(n: Int) = ((1 + math.sqrt(1 + 8.0 * n)) / 2).round.toInt

    // The rows of the Cholesky factor, entries 0..i, and log |det J|.
    def cholesky(z: Tensor1[LZ, Float32]) =
      val d = size(z.shape(zAxis))
      val c = z.tanh
      // log(1 - tanh(z)²) = 2·(log 2 - |z| - log(1 + e^{-2|z|})), the
      // Jacobian of tanh, written to stay finite for large |z|
      val absZ = z.abs
      val log1pExp = ((absZ *! Tensor0(-2f)).exp +! one).log
      var logDet =
        ((Tensor0(math.log(2).toFloat) -! absZ - log1pExp) *! Tensor0(2f)).sum
      var index = 0
      val rows = Vector.newBuilder[Vector[Tensor0[Float32]]]
      val logDiag = Array.fill(d)(Tensor0(0f)) // log w_jj
      for i <- 0 until d do
        var left = one // 1 - Σ_{k<j} w_ik²
        val row = Vector.newBuilder[Tensor0[Float32]]
        for j <- 0 until i do
          val w = c.select(zAxis.at(index)) * left.sqrt
          index += 1
          // ∂w_ij/∂c_ij = sqrt(left), ∂x_ij/∂w_ij = w_jj
          logDet = logDet + half * left.log + logDiag(j)
          left = left - w * w
          row += w
        logDiag(i) = half * left.log
        row += left.sqrt
        rows += row.result()
      (rows.result(), logDet)

    new Bijector[Tensor1[LZ, Float32], Tensor2[L, Prime[L], Float32]]:
      def forward(z: Tensor1[LZ, Float32]) =
        val (w, _) = cholesky(z)
        val d = w.size
        def entry(i: Int, j: Int) =
          if i == j then one
          else (0 to math.min(i, j)).map(k => w(i)(k) * w(j)(k)).reduce(_ + _)
        stack(
          (0 until d).map(i =>
            stack((0 until d).map(j => entry(i, j)), Axis[Prime[L]])
          ),
          Axis[L]
        )

      def inverse(x: Tensor2[L, Prime[L], Float32]) =
        val d = x.shape(Axis[L])
        val w = LinearAlgebra.cholesky(x)
        def at(i: Int, j: Int) = w(Axis[L].at(i))(Axis[Prime[L]].at(j))
        val zs =
          for
            i <- 1 until d
            j <- 0 until i
          yield
            val left =
              (0 until j).foldLeft(one)((acc, k) => acc - at(i, k) * at(i, k))
            val c = at(i, j) / left.sqrt
            ((one + c) / (one - c)).log * half // atanh(c)
        stack(zs, zAxis)

      def logDetJac(z: Tensor1[LZ, Float32]) = cholesky(z)._2
