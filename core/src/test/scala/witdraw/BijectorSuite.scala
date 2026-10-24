package witdraw

import dimwit.*
import dimwit.autodiff.Autodiff
import dimwit.linalg.LinearAlgebra.det
import dimwit.stats.*

class BijectorSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  trait A derives Label
  trait C derives Label

  def randomZ(n: Int, seed: Int): Tensor1[A, Float32] =
    Normal.standardNormal(Shape(Axis[A] -> n)).sample(Key(seed))

  /** log |det J| of `f` at `z`, by autodiff. */
  def logAbsDetJacobian(f: Tensor1[A, Float32] => Tensor1[C, Float32])(
      z: Tensor1[A, Float32]
  ): Float =
    math.log(math.abs(det(Autodiff.jacobian(f)(z)).item)).toFloat

  /** `logDetJac` agrees with autodiff, and `inverse` undoes `forward`. */
  def check(
      b: Bijector[Tensor1[A, Float32], Tensor1[C, Float32]],
      n: Int,
      square: Tensor1[C, Float32] => Tensor1[C, Float32] = identity
  ): Unit =
    for seed <- 0 until 3 do
      val z = randomZ(n, seed)
      val expected = logAbsDetJacobian(z => square(b.forward(z)))(z)
      assertEqualsFloat(b.logDetJac(z).item, expected, 1e-3f, s"seed $seed")
      val roundTrip = (b.inverse(b.forward(z)) - z).abs.max.item
      assert(roundTrip < 1e-3f, s"inverse(forward(z)) differs by $roundTrip")

  /** The same bijector, with the axis of `x` renamed to `C`. */
  def onC(
      b: Bijector[Tensor1[A, Float32], Tensor1[A, Float32]]
  ): Bijector[Tensor1[A, Float32], Tensor1[C, Float32]] =
    new Bijector[Tensor1[A, Float32], Tensor1[C, Float32]]:
      def forward(z: Tensor1[A, Float32]) = b.forward(z).relabelTo(Axis[C])
      def inverse(x: Tensor1[C, Float32]) = b.inverse(x.relabelTo(Axis[A]))
      def logDetJac(z: Tensor1[A, Float32]) = b.logDetJac(z)

  test("exp"):
    check(onC(Bijector.exp[Tuple1[A]]), n = 4)

  test("interval"):
    val lo = Tensor1(Axis[A] -> 4).fill(-1f)
    val hi = Tensor1(Axis[A] -> 4).fill(2f)
    check(onC(Bijector.interval(lo, hi)), n = 4)

  test("ordered"):
    check(onC(Bijector.ordered[A]), n = 4)

  test("correlationMatrix, on the entries below the diagonal"):
    trait L derives Label
    val b = Bijector.correlationMatrix[A, L]
    for d <- Seq(2, 3, 4) do
      def below(x: Tensor2[L, Prime[L], Float32]): Tensor1[C, Float32] =
        stack(
          for
            i <- 1 until d
            j <- 0 until i
          yield x(Axis[L].at(i))(Axis[Prime[L]].at(j)),
          Axis[C]
        )
      for seed <- 0 until 3 do
        val z = randomZ(d * (d - 1) / 2, seed) *! Tensor0(2f)
        val x = b.forward(z)
        // symmetric, unit diagonal, positive definite
        def at(i: Int, j: Int) = x(Axis[L].at(i))(Axis[Prime[L]].at(j)).item
        for i <- 0 until d; j <- 0 until d do
          val expected = if i == j then 1f else at(j, i)
          assertEqualsFloat(at(i, j), expected, 1e-5f, s"d $d: x($i, $j)")
        assert(det(x).item > 0f, s"d $d seed $seed: det ${det(x).item}")
        val expected = logAbsDetJacobian(z => below(b.forward(z)))(z)
        assertEqualsFloat(b.logDetJac(z).item, expected, 1e-3f, s"d $d seed $seed")
        val roundTrip = (b.inverse(x) - z).abs.max.item
        assert(roundTrip < 1e-3f, s"d $d: inverse(forward(z)) differs by $roundTrip")

  test("simplexStickBreaking, on the first k-1 coordinates of the simplex"):
    for k <- Seq(2, 3, 5) do
      check(
        Bijector.simplexStickBreaking[A, C],
        n = k - 1,
        square = x => x.select(Axis[C].at(0 until k - 1))
      )
