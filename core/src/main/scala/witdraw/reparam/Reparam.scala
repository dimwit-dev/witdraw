package witdraw.reparam

import dimwit.*

import scala.compiletime.constValueTuple
import scala.deriving.Mirror

/** Builds a single [[Bijector]] over a whole parameter record from a
  * per-field spec, and pushes a constrained-space log-density into
  * unconstrained space for gradient-based MCMC.
  *
  * The spec is a product (named tuple or case class) whose field *names*
  * match the parameter record's, and whose field *values* are per-parameter
  * [[Bijector]]s. Fields the spec omits get [[Bijector.identity]]. `Z` (the
  * unconstrained record type) is supplied by the caller — it equals `C` when
  * every constraint is shape-preserving, and differs only in the fields whose
  * bijector changes shape (e.g. a stick-breaking simplex: `k-1` vs `k`).
  *
  * Models, handlers and interpreters are untouched: `Reparam` is a pure
  * adapter between the sampler and `logDensity`.
  *
  * @example
  *   {{{
  * // Posterior = (theta: T1[Comp], mu: T1[Comp], sigma: T1[Comp])
  * // PosteriorZ = (theta: T1[ThetaZ], mu: T1[Comp], sigma: T1[Comp])   // ThetaZ has K-1
  * val bij = Reparam.build[Posterior, PosteriorZ]((
  *   theta = Bijector.simplexStickBreaking[ThetaZ, Comp](k),
  *   mu    = Bijector.ordered[Comp](k),
  *   sigma = Bijector.exp[Comp *: EmptyTuple, Float32],
  * ))
  * val target = Reparam.pushForward(bij)(Mixture.logDensity(y))   // PosteriorZ => Tensor0
  * val z0     = bij.inverse(initConstrained)
  * val draws  = nuts.run(z0, key)....map(bij.forward)             // report constrained
  *   }}}
  */
object Reparam:

  /** Turn a constrained-space log-density into an unconstrained-space one:
    * `z => logDensity(forward(z)) + logDetJac(z)`.
    */
  def pushForward[Z, C](
      bijector: Bijector[Z, C]
  )(logDensity: C => Tensor0[Float32]): Z => Tensor0[Float32] =
    z => logDensity(bijector.forward(z)) + bijector.logDetJac(z)

  /** Field-wise bijector `Z <-> C` from a (possibly partial) `spec` of
    * per-field bijectors. Field order is taken from `C`; `Spec` is inferred
    * from the argument.
    *
    * {{{ Reparam.build[Posterior, PosteriorZ]((sigma = Bijector.exp, ...)) }}}
    */
  def build[C, Z]: Build[C, Z] = new Build[C, Z]

  final class Build[C, Z]:
    inline def apply[Spec](spec: Spec)(using
        mc: Mirror.ProductOf[C],
        mz: Mirror.ProductOf[Z],
        ms: Mirror.ProductOf[Spec]
    ): Bijector[Z, C] =
      assemble(
        constValueTuple[mc.MirroredElemLabels].toList.map(_.toString),
        constValueTuple[ms.MirroredElemLabels].toList.map(_.toString),
        spec.asInstanceOf[Product],
        mc,
        mz
      )

  private def assemble[C, Z](
      cNames: List[String],
      specNames: List[String],
      spec: Product,
      mc: Mirror.ProductOf[C],
      mz: Mirror.ProductOf[Z]
  ): Bijector[Z, C] =
    val specBijectors =
      spec.productIterator.toList.asInstanceOf[List[Bijector[Any, Any]]]
    val byName = specNames.iterator.zip(specBijectors.iterator).toMap
    val unknown = byName.keySet -- cNames.toSet
    require(
      unknown.isEmpty,
      s"Reparam spec has fields not in the parameter record: ${unknown.mkString(", ")}"
    )
    val perField: List[Bijector[Any, Any]] =
      cNames.map(n => byName.getOrElse(n, Bijector.identityErased))

    new Bijector[Z, C]:
      def forward(z: Z): C =
        val zs = z.asInstanceOf[Product].productIterator.toList
        val xs = perField.lazyZip(zs).map((b, zi) => b.forward(zi))
        mc.fromProduct(Tuple.fromArray(xs.toArray))
      def inverse(c: C): Z =
        val xs = c.asInstanceOf[Product].productIterator.toList
        val zs = perField.lazyZip(xs).map((b, xi) => b.inverse(xi))
        mz.fromProduct(Tuple.fromArray(zs.toArray))
      def logDetJac(z: Z): Tensor0[Float32] =
        val zs = z.asInstanceOf[Product].productIterator.toList
        perField
          .lazyZip(zs)
          .map((b, zi) => b.logDetJac(zi))
          .foldLeft(Tensor0(0f))(_ + _)

  /** Shape-preserving case — no constraint changes a tensor's shape, so the
    * unconstrained record has the same type as the constrained one. Just
    * `Reparam.build[C, C](spec)`.
    */
