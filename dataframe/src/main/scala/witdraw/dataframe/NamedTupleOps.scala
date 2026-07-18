package witdraw
package dataframe

import scala.NamedTuple.*

// ---------------------------------------------------------------------------
// Col – phantom label reference carrying the name at the type level
// ---------------------------------------------------------------------------

class Col[L <: String & Singleton]()
object Col:
  inline def apply[L <: String & Singleton](using ValueOf[L]): Col[L] =
    new Col[L]()

// ---------------------------------------------------------------------------
// Type-level utilities
// ---------------------------------------------------------------------------

object NamedTupleOps:

  /** Remove a name `L` from a Names tuple. */
  type DropName[Names <: Tuple, L] <: Tuple = Names match
    case L *: t     => t
    case h *: t     => h *: DropName[t, L]
    case EmptyTuple => EmptyTuple

  /** Remove the type at the position of `L` in `Names` from `Types`. */
  type DropType[Types <: Tuple, Names <: Tuple, L] <: Tuple = Names match
    case L *: _     => Types match { case _ *: t => t }
    case _ *: nt    => Types match { case h *: t => h *: DropType[t, nt, L] }
    case EmptyTuple => EmptyTuple

  /** Keep only names NOT in `Drop`. */
  type KeepNames[Names <: Tuple, Drop <: Tuple] <: Tuple = Names match
    case EmptyTuple => EmptyTuple
    case h *: t     =>
      Tuple.Contains[Drop, h] match
        case true  => KeepNames[t, Drop]
        case false => h *: KeepNames[t, Drop]

  /** Extract label types from a tuple of [[Col]] phantoms. */
  type ExtractLabels[Cols <: Tuple] <: Tuple = Cols match
    case Col[l] *: t => l *: ExtractLabels[t]
    case EmptyTuple  => EmptyTuple

  /** True iff every label in `Labels` is present in `Names`. */
  type AllContained[Names <: Tuple, Labels <: Tuple] <: Boolean = Labels match
    case EmptyTuple => true
    case h *: t     =>
      Tuple.Contains[Names, h] match
        case true  => AllContained[Names, t]
        case false => false

  /** Look up the type for name `L` in a Names/Types pair. */
  type LookupType[Names <: Tuple, Types <: Tuple, L] = Names match
    case L *: _  => Types match { case h *: _ => h }
    case _ *: nt => Types match { case _ *: tv => LookupType[nt, tv, L] }

  /** For each name in `Select`, look up its type in Names/Types. */
  type LookupTypes[Names <: Tuple, Types <: Tuple, Select <: Tuple] <: Tuple =
    Select match
      case EmptyTuple => EmptyTuple
      case h *: t => LookupType[Names, Types, h] *: LookupTypes[Names, Types, t]

  // ---------------------------------------------------------------------------
  // DropField – drop one field from a (Names, Values) pair by label
  // ---------------------------------------------------------------------------

  trait DropField[Names <: Tuple, Values <: Tuple, L <: String & Singleton]:
    type OutNames <: Tuple
    type OutValues <: Tuple
    def apply(values: Values): OutValues

  object DropField:
    given baseCase[L <: String & Singleton, V, TN <: Tuple, TV <: Tuple]
        : DropField[L *: TN, V *: TV, L] with
      type OutNames = TN
      type OutValues = TV
      def apply(values: V *: TV): TV = values.tail

    given recursiveCase[
        H <: String & Singleton,
        HV,
        TN <: Tuple,
        TV <: Tuple,
        L <: String & Singleton
    ](using
        tail: DropField[TN, TV, L]
    ): DropField[H *: TN, HV *: TV, L] with
      type OutNames = H *: tail.OutNames
      type OutValues = HV *: tail.OutValues
      def apply(values: HV *: TV): HV *: tail.OutValues =
        values.head *: tail(values.tail)

  // ---------------------------------------------------------------------------
  // LookupField – extract one field value by label
  // ---------------------------------------------------------------------------

  trait LookupField[Names <: Tuple, Values <: Tuple, L <: String & Singleton]:
    type Out
    def apply(values: Values): Out

  object LookupField:
    given baseCase[L <: String & Singleton, V, TN <: Tuple, TV <: Tuple]
        : LookupField[L *: TN, V *: TV, L] with
      type Out = V
      def apply(values: V *: TV): V = values.head

    given recursiveCase[
        H <: String & Singleton,
        HV,
        TN <: Tuple,
        TV <: Tuple,
        L <: String & Singleton
    ](using
        tail: LookupField[TN, TV, L]
    ): LookupField[H *: TN, HV *: TV, L] with
      type Out = tail.Out
      def apply(values: HV *: TV): tail.Out = tail(values.tail)

  // ---------------------------------------------------------------------------
  // UpdateField – replace the value of one field by label, optionally changing its type
  // ---------------------------------------------------------------------------

  /** Replace the type at position of `L` in `Names` with `A`. */
  type UpdatedType[Types <: Tuple, Names <: Tuple, L, A] <: Tuple = Names match
    case L *: nt => Types match { case _ *: tv => A *: tv }
    case h *: nt =>
      Types match { case hv *: tv => hv *: UpdatedType[tv, nt, L, A] }
    case EmptyTuple => EmptyTuple

  trait UpdateField[
      Names <: Tuple,
      Values <: Tuple,
      L <: String & Singleton,
      A
  ]:
    def apply(values: Values, newValue: A): UpdatedType[Values, Names, L, A]

  object UpdateField:
    given baseCase[L <: String & Singleton, V, A, TN <: Tuple, TV <: Tuple]
        : UpdateField[L *: TN, V *: TV, L, A] with
      def apply(values: V *: TV, newValue: A): A *: TV =
        newValue *: values.tail

    given recursiveCase[
        H <: String & Singleton,
        HV,
        TN <: Tuple,
        TV <: Tuple,
        L <: String & Singleton,
        A
    ](using
        tail: UpdateField[TN, TV, L, A]
    ): UpdateField[H *: TN, HV *: TV, L, A] with
      def apply(
          values: HV *: TV,
          newValue: A
      ): UpdatedType[HV *: TV, H *: TN, L, A] =
        (values.head *: tail(values.tail, newValue))
          .asInstanceOf[UpdatedType[HV *: TV, H *: TN, L, A]]

  // ---------------------------------------------------------------------------
  // SelectFields – build an output tuple by selecting fields in order
  // ---------------------------------------------------------------------------

  trait SelectFields[Names <: Tuple, Values <: Tuple, Select <: Tuple]:
    type OutValues <: Tuple
    def apply(values: Values): OutValues

  object SelectFields:
    given emptyCase[Names <: Tuple, Values <: Tuple]
        : SelectFields[Names, Values, EmptyTuple] with
      type OutValues = EmptyTuple
      def apply(values: Values): EmptyTuple = EmptyTuple

    given consCase[
        Names <: Tuple,
        Values <: Tuple,
        H <: String & Singleton,
        T <: Tuple
    ](using
        head: LookupField[Names, Values, H],
        tail: SelectFields[Names, Values, T]
    ): SelectFields[Names, Values, H *: T] with
      type OutValues = head.Out *: tail.OutValues
      def apply(values: Values): head.Out *: tail.OutValues =
        head(values) *: tail(values)

// ---------------------------------------------------------------------------
// Extension methods on NamedTuple
// ---------------------------------------------------------------------------

extension [K <: Tuple, V <: Tuple](nt: NamedTuple[K, V])

  /** Drop a single column by name.
    *
    * Example: {{{row.drop(Col["age"])}}}
    */
  inline def drop[L <: String & Singleton](col: Col[L])(using
      ev: NamedTupleOps.DropField[K, V, L]
  ): NamedTuple[
    NamedTupleOps.DropName[K, L],
    NamedTupleOps.DropType[V, K, L]
  ] =
    ev(nt.toTuple).asInstanceOf[NamedTuple[
      NamedTupleOps.DropName[K, L],
      NamedTupleOps.DropType[V, K, L]
    ]]

  /** Select (and optionally reorder) columns by passing a tuple of [[Col]]
    * phantoms. Types are inferred from the existing schema.
    *
    * Example: {{{row.select((Col["age"], Col["income"]))}}}
    */
  inline def select[Cols <: Tuple](cols: Cols)(using
      NamedTupleOps.AllContained[K, NamedTupleOps.ExtractLabels[Cols]] =:= true
  )(using
      ev: NamedTupleOps.SelectFields[K, V, NamedTupleOps.ExtractLabels[Cols]]
  ): NamedTuple[
    NamedTupleOps.ExtractLabels[Cols],
    NamedTupleOps.LookupTypes[K, V, NamedTupleOps.ExtractLabels[Cols]]
  ] =
    ev(nt.toTuple).asInstanceOf[NamedTuple[
      NamedTupleOps.ExtractLabels[Cols],
      NamedTupleOps.LookupTypes[K, V, NamedTupleOps.ExtractLabels[Cols]]
    ]]

  /** Drop multiple columns by passing a tuple of [[Col]] phantoms.
    *
    * Example: {{{row.dropColumns((Col["age"], Col["income"]))}}}
    */
  inline def dropColumns[Cols <: Tuple](cols: Cols)(using
      NamedTupleOps.AllContained[K, NamedTupleOps.ExtractLabels[Cols]] =:= true
  )(using
      ev: NamedTupleOps.SelectFields[
        K,
        V,
        NamedTupleOps.KeepNames[K, NamedTupleOps.ExtractLabels[Cols]]
      ]
  ): NamedTuple[
    NamedTupleOps.KeepNames[K, NamedTupleOps.ExtractLabels[Cols]],
    NamedTupleOps.LookupTypes[K, V, NamedTupleOps.KeepNames[
      K,
      NamedTupleOps.ExtractLabels[Cols]
    ]]
  ] =
    ev(nt.toTuple).asInstanceOf[NamedTuple[
      NamedTupleOps.KeepNames[K, NamedTupleOps.ExtractLabels[Cols]],
      NamedTupleOps.LookupTypes[
        K,
        V,
        NamedTupleOps.KeepNames[K, NamedTupleOps.ExtractLabels[Cols]]
      ]
    ]]

  /** Replace the value of a single field by col reference, optionally changing
    * its type.
    *
    * Example:
    * {{{row.updated(Col["MarketCap_BCHF"])(row.MarketCap_BCHF * 1000)}}}
    */
  inline def updated[L <: String & Singleton, A](col: Col[L])(newValue: A)(using
      ev: NamedTupleOps.UpdateField[K, V, L, A]
  ): NamedTuple[K, NamedTupleOps.UpdatedType[V, K, L, A]] =
    ev(nt.toTuple, newValue)
      .asInstanceOf[NamedTuple[K, NamedTupleOps.UpdatedType[V, K, L, A]]]
