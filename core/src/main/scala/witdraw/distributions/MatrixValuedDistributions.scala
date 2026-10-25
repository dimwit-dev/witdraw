package witdraw.distributions

import dimwit.*
import dimwit.linalg.LinearAlgebra
import dimwit.stats.{Beta, Normal}
import witdraw.{Support, SupportOf}

/** LKJ distribution over d×d correlation matrices.
  *
  * The LKJ distribution (Lewandowski, Kurowicka, Joe 2009) is parameterised by
  * a scalar concentration η > 0. η = 1 is uniform over all correlation
  * matrices; η > 1 concentrates mass near the identity matrix.
  *
  * Sampling uses the Onion method (Lewandowski et al. 2009, Algorithm 3): the
  * correlation matrix is grown from 2×2 to d×d one row/column at a time, each
  * step using the Cholesky factor of the existing submatrix.
  */
class LKJ[L: Label, V: IsFloating](
    val concentration: Tensor0[V],
    val matrixShape: Shape2[L, Prime[L]]
) extends Distribution[(L, Prime[L]), V]:

  private val d: Int = matrixShape.dimensions(0)

  override def logProb(R: Tensor2[L, Prime[L], V]): Tensor0[LogProb] =
    val L_chol = LinearAlgebra.cholesky(R)
    val logDet: Tensor0[Float32] =
      (L_chol.diagonal(Axis[L]).log.sum * Tensor0(VType[V])(2f))
        .asType(VType[Float32])
    val eta = concentration.asType(VType[Float32])
    LogProb(((eta - Tensor0(1f)) * logDet - logNormaliser(eta)).asFloat32)

  /** `log c_d`, with `p(R) = det(R)^(η-1) / c_d` (Lewandowski et al. 2009, eq.
    * 16):
    * `log c_d = Σ_{k=1}^{d-1} (d-k) · [ (2η-2+d-k)·log 2 + log B(a_k, a_k) ]`,
    * where `a_k = η + (d-k-1)/2`.
    */
  private def logNormaliser(eta: Tensor0[Float32]): Tensor0[Float32] =
    (1 until d)
      .map: k =>
        val a = eta + Tensor0((d - k - 1) / 2f)
        val powerOf2 = eta * Tensor0(2f) + Tensor0((d - k - 2).toFloat)
        (powerOf2 * Tensor0(math.log(2).toFloat) + logBetaFunction(a)) *
          Tensor0((d - k).toFloat)
      .foldLeft(Tensor0(0f))(_ + _)

  /** `log B(a, a)`. dimwit has no log-gamma function, so this reads it off the
    * Beta(a, a) density at 1/2: `log p(1/2) = 2(a-1)·log(1/2) - log B(a, a)`.
    */
  private def logBetaFunction(a: Tensor0[Float32]): Tensor0[Float32] =
    val logDensityAtHalf = Beta(a, a).logProb(Tensor0(0.5f)).asFloat32
    (a - Tensor0(1f)) * Tensor0(2 * math.log(0.5).toFloat) - logDensityAtHalf

  override def sample(key: Random.Key): Tensor2[L, Prime[L], V] =

    // Initialise the full d×d result matrix to the identity.
    var R: Tensor2[L, Prime[L], V] = Tensor2(matrixShape).eye(VType[V])

    if d < 2 then return R

    // One key for the initialisation step, one key per loop iteration.
    val keys = key.split(d - 1)

    // ── Step 1: initialise the 2×2 top-left block ──────────────────────────
    // β = η + (d−2)/2;  r₁₂ = 2u − 1,  u ~ Beta(β, β)
    var beta: Tensor0[Float32] =
      concentration.asType(VType[Float32]) + Tensor0((d - 2) / 2.0f)
    val u0: Tensor0[Float32] = Beta(beta, beta).sample(keys(0))
    val r12: Tensor0[V] = (u0 * Tensor0(2f) - Tensor0(1f)).asType(VType[V])
    R = setEntry(R, 0, 1, r12)
    R = setEntry(R, 1, 0, r12)

    // ── Step 2: grow the block from 2×2 up to d×d ─────────────────────────
    for n <- 2 until d do
      beta = beta - Tensor0(0.5f)
      val (kb, ku) = keys(n - 1).split2()

      // y ~ Beta(n/2, β)
      val y: Tensor0[Float32] = Beta(Tensor0(n / 2.0f), beta).sample(kb)

      // u uniform on the n-sphere via normalised standard normals
      val zFull: Tensor1[Prime[L], Float32] =
        Normal.standardNormal(Shape1(Axis[Prime[L]] -> d)).sample(ku)
      val zSlice: Tensor1[Prime[L], Float32] =
        zFull.select(Axis[Prime[L]].at(0 until n))
      val uUnit: Tensor1[Prime[L], Float32] =
        zSlice /! LinearAlgebra.norm(zSlice, LinearAlgebra.VectorNormType.L2)

      // w = √y · u
      val w: Tensor1[Prime[L], V] = (uUnit *! y.sqrt).asType(VType[V])

      // A = chol(R[0:n, 0:n]),  z = A · w  (new off-diagonal column)
      val subR = R(Axis[L].at(0 until n))(Axis[Prime[L]].at(0 until n))
      val A = LinearAlgebra.cholesky(subR)
      val z: Tensor1[L, V] = A.dot(Axis[Prime[L]])(w)

      // Extend R symmetrically: new column then new row
      val column = R(Axis[Prime[L]].at(n))
      R = R
        .updated(Axis[Prime[L]].at(n))(column.updated(Axis[L].at(0 until n))(z))
      val zRow: Tensor1[Prime[L], V] = z.relabelTo(Axis[Prime[L]])
      val row = R(Axis[L].at(n))
      R = R.updated(Axis[L].at(n))(
        row.updated(Axis[Prime[L]].at(0 until n))(zRow)
      )

    R

  /** `m` with the entry in row `i` and column `j` set to `value`. */
  private def setEntry(
      m: Tensor2[L, Prime[L], V],
      i: Int,
      j: Int,
      value: Tensor0[V]
  ): Tensor2[L, Prime[L], V] =
    m.updated(Axis[L].at(i))(
      m(Axis[L].at(i)).updated(Axis[Prime[L]].at(j))(value)
    )

object LKJ:
  given [L: Label, V]: SupportOf[LKJ[L, V], (L, Prime[L]), V] =
    _ => Support.CorrelationMatrix()
