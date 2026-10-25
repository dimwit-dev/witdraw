package witdraw

import dimwit.*
import dimwit.stats.*
import dimwit.tensor.Labels

/** The values a distribution can take.
  */
enum Support[T <: Tuple, V]:

  /** Any value: no change of variables. */
  case Real[T <: Tuple, V]() extends Support[T, V]

  /** Values greater than `lo`, element-wise. */
  case GreaterThan[T <: Tuple, V](lo: Tensor[T, V]) extends Support[T, V]

  /** Values in `(lo, hi)`, element-wise. */
  case Interval[T <: Tuple, V](lo: Tensor[T, V], hi: Tensor[T, V])
      extends Support[T, V]

  /** Vectors along `L` with positive entries that sum to one. */
  case Simplex[L, V]()(using val label: Label[L]) extends Support[Tuple1[L], V]

  /** Correlation matrices: symmetric, positive definite, ones on the diagonal.
    */
  case CorrelationMatrix[L, V]()(using val label: Label[L])
      extends Support[(L, Prime[L]), V]

  /** Matrices whose every row, a vector along `L`, lies in `row`. For example,
    * `Rowwise(Simplex())` are the row-stochastic matrices, such as the
    * transition matrix of a Markov chain.
    *
    * `row` may be [[Real]], [[GreaterThan]], [[Interval]] or [[Simplex]].
    */
  case Rowwise[R, L, V](row: Support[Tuple1[L], V])(using
      val rowLabel: Label[R],
      val label: Label[L]
  ) extends Support[(R, L), V]

  /** A support that brings its own bijector, from unconstrained values
    * `Tensor[TZ, VZ]` to its values, for a support witdraw does not know,
    * @param description
    *   for error messages
    */
  case Custom[T <: Tuple, V, TZ <: Tuple, VZ](
      description: String,
      bijector: Bijector[Tensor[TZ, VZ], Tensor[T, V]]
  )(using val zLabels: Labels[TZ]) extends Support[T, V]

  /** Integer or boolean values. Such a site cannot be latent. */
  case Discrete[T <: Tuple, V]() extends Support[T, V]

  /** No [[SupportOf]] instance was found, e.g. because the distribution was
    * passed around as a plain `Distribution`. A latent site with an unknown
    * support needs an explicit [[Constraint]].
    */
  case Unknown[T <: Tuple, V]() extends Support[T, V]

object Support:

  /** The axis of the unconstrained values of the default bijectors that change
    * the shape (simplex, correlation matrix). Users never see it: [[Latent]]
    * flattens all unconstrained values into one vector.
    */
  private[witdraw] trait Unconstrained derives Label

  /** The change of variables for continuous values in the support `s`, or
    * `None` for an unknown or discrete support, or a [[Rowwise]] support with
    * rows that have no bijector.
    */
  private[witdraw] def transform[T <: Tuple: Labels](
      s: Support[T, Float32]
  ): Option[Transform[T, Float32]] =
    s match
      case Real()           => Some(Transform.identity)
      case GreaterThan(lo)  => Some(Transform(Bijector.lowerBound(lo)))
      case Interval(lo, hi) => Some(Transform(Bijector.interval(lo, hi)))
      case c: Custom[T, Float32, tz, vz] @unchecked =>
        given Labels[tz] = c.zLabels
        Some(Transform(c.bijector))
      case r: Rowwise[rr, l, Float32] @unchecked =>
        given Label[rr] = r.rowLabel
        given Label[l] = r.label
        rowwiseTransform[rr, l](r.row)
      case s: Simplex[l, Float32] @unchecked =>
        given Label[l] = s.label
        Some(Transform(Bijector.simplexStickBreaking[Unconstrained, l]))
      case s: CorrelationMatrix[l, Float32] @unchecked =>
        given Label[l] = s.label
        Some(Transform(Bijector.correlationMatrix[Unconstrained, l]))
      case Discrete() | Unknown() => None

  /** A bijector per row, for a [[Rowwise]] support. */
  private def rowwiseTransform[R: Label, L: Label](
      row: Support[Tuple1[L], Float32]
  ): Option[Transform[(R, L), Float32]] =
    def rowwise[LZ: Label](
        b: Bijector[Tensor1[LZ, Float32], Tensor1[L, Float32]]
    ): Transform[(R, L), Float32] =
      Transform(Bijector.rowwise[R, LZ, L](b))
    row match
      case Real()           => Some(rowwise[L](Bijector.identity))
      case GreaterThan(lo)  => Some(rowwise[L](Bijector.lowerBound(lo)))
      case Interval(lo, hi) => Some(rowwise[L](Bijector.interval(lo, hi)))
      case Simplex()        =>
        Some(rowwise(Bijector.simplexStickBreaking[Unconstrained, L]))
      case _ => None

/** The support of the distributions of class `Dist`, whose values are
  * `Tensor[T, V]`. `site ~ dist` looks it up for the static type of `dist`.
  *
  * To give a distribution of your own a support, put an instance in its
  * companion object:
  * {{{
  * class LogNormal[T <: Tuple: Labels](...) extends IndependentDistribution[T, Float32]
  * object LogNormal:
  *   given [T <: Tuple: Labels]: SupportOf[LogNormal[T], T, Float32] =
  *     d => Support.GreaterThan(Tensor(d.shape).fill(0f))
  * }}}
  *
  * Contravariant in `Dist`: a subclass has the support of its parent unless it
  * gives its own instance.
  */
trait SupportOf[-Dist, T <: Tuple, V]:
  def apply(d: Dist): Support[T, V]

object SupportOf:
  import Support.*

  /** For a distribution without an instance: the support is unknown. Not a
    * given, but the default of the `using` parameter of `~`, so it never
    * competes with an instance.
    */
  def unknown[D, T <: Tuple, V]: SupportOf[D, T, V] = _ => Unknown()

  given normal[T <: Tuple, V]: SupportOf[Normal[T, V], T, V] = _ => Real()
  given cauchy[T <: Tuple, V]: SupportOf[Cauchy[T, V], T, V] = _ => Real()
  given studentT[T <: Tuple, V]: SupportOf[StudentT[T, V], T, V] = _ => Real()
  given mvNormal[L, V]: SupportOf[MVNormal[L, V], Tuple1[L], V] = _ => Real()

  given halfNormal[T <: Tuple, V]: SupportOf[HalfNormal[T, V], T, V] =
    d => GreaterThan(d.loc)
  given halfCauchy[T <: Tuple, V]: SupportOf[HalfCauchy[T, V], T, V] =
    d => GreaterThan(d.loc)
  given exponential[T <: Tuple: Labels]
      : SupportOf[Exponential[T, Float32], T, Float32] =
    d => GreaterThan(Tensor.like(d.rate).fill(0f))

  given uniform[T <: Tuple, V]: SupportOf[Uniform[T, V], T, V] =
    d => Interval(d.low, d.high)
  given truncatedNormal[T <: Tuple, V]: SupportOf[TruncatedNormal[T, V], T, V] =
    d => Interval(d.low, d.high)
  given beta[T <: Tuple: Labels]: SupportOf[Beta[T, Float32], T, Float32] =
    d => Interval(Tensor.like(d.alpha).fill(0f), Tensor.like(d.alpha).fill(1f))

  given dirichlet[L: Label, V]: SupportOf[Dirichlet[L, V], Tuple1[L], V] =
    _ => Simplex()

  given bernoulli[T <: Tuple]: SupportOf[Bernoulli[T], T, Bool] =
    _ => Discrete()
  given binomial[T <: Tuple, V]: SupportOf[Binomial[T, V], T, V] =
    _ => Discrete()
  given poisson[T <: Tuple, VS, VR]: SupportOf[Poisson[T, VS, VR], T, VS] =
    _ => Discrete()
  given discreteUniform[T <: Tuple]: SupportOf[DiscreteUniform[T], T, Int32] =
    _ => Discrete()
  given categorical[L]: SupportOf[Categorical[L], EmptyTuple, Int32] =
    _ => Discrete()
  given multinomial[L]: SupportOf[Multinomial[L], Tuple1[L], Int32] =
    _ => Discrete()
