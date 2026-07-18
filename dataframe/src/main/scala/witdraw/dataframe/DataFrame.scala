package witdraw
package dataframe

import java.io.File
import scala.NamedTuple.*
import scala.compiletime.constValueTuple
import scala.io.Source
import scala.annotation.publicInBinary

// ---------------------------------------------------------------------------
// FromString – converts a raw CSV cell string to a typed value
// ---------------------------------------------------------------------------

trait FromString[A]:
  def parse(s: String): A

object FromString:
  given FromString[String] with
    def parse(s: String): String = s

  given FromString[Int] with
    def parse(s: String): Int = s.trim.toInt

  given FromString[Long] with
    def parse(s: String): Long = s.trim.toLong

  given FromString[Double] with
    def parse(s: String): Double = s.trim.toDouble

  given FromString[Float] with
    def parse(s: String): Float = s.trim.toFloat

  given FromString[Boolean] with
    def parse(s: String): Boolean = s.trim.toLowerCase match
      case "true" | "1" | "yes" => true
      case _                    => false

  given [A: FromString]: FromString[Option[A]] with
    def parse(s: String): Option[A] =
      if s == null || s.trim.isEmpty then None
      else Some(summon[FromString[A]].parse(s))

  given [A: FromString]: FromString[A | Null] with
    def parse(s: String): A | Null =
      if s == null || s.trim.isEmpty then null
      else summon[FromString[A]].parse(s)

// ---------------------------------------------------------------------------
// FromRow – parses a positional List[String] into a typed tuple V
// ---------------------------------------------------------------------------

trait FromRow[V <: Tuple]:
  def parse(cells: List[String]): V

object FromRow:
  given FromRow[EmptyTuple] with
    def parse(cells: List[String]): EmptyTuple = EmptyTuple

  given [H: FromString, T <: Tuple: FromRow]: FromRow[H *: T] with
    def parse(cells: List[String]): H *: T =
      summon[FromString[H]].parse(cells.head) *: summon[FromRow[T]].parse(
        cells.tail
      )

// ---------------------------------------------------------------------------
// ToRow – serialises a typed tuple V back to List[String]
// ---------------------------------------------------------------------------

trait ToRow[V <: Tuple]:
  def serialise(values: V): List[String]

object ToRow:
  given ToRow[EmptyTuple] with
    def serialise(values: EmptyTuple): List[String] = Nil

  given [H, T <: Tuple: ToRow]: ToRow[H *: T] with
    def serialise(values: H *: T): List[String] =
      values.head.toString :: summon[ToRow[T]].serialise(values.tail)

// ---------------------------------------------------------------------------
// DataFrame – pure Scala, backed by IndexedSeq[NamedTuple[K, V]]
// ---------------------------------------------------------------------------

/** A typed, pure-Scala dataframe backed by an IndexedSeq of NamedTuple rows.
  *
  * Names – tuple of column-name string literals, e.g. `("age", "income")` Types
  * – tuple of Scala value types, e.g. `(Int, Double)`
  */
class DataFrame[ColNames <: Tuple, Values <: Tuple](
    val data: IndexedSeq[NamedTuple[ColNames, Values]]
):
  /** Return the first `n` rows as a new DataFrame. */
  def head(n: Int = 5): DataFrame[ColNames, Values] =
    new DataFrame[ColNames, Values](data.take(n))

  /** All rows as typed NamedTuples. Scala collections operate directly on the
    * result.
    */
  def rows: IndexedSeq[NamedTuple[ColNames, Values]] = data

  /** Extract a single column as an IndexedSeq.
    *
    * Example:
    * {{{df.col(_.actor) // IndexedSeq[Int] df.col(_.balance) // IndexedSeq[Double]}}}
    */
  def col[A](f: NamedTuple[ColNames, Values] => A): IndexedSeq[A] =
    data.map(f)

  /** Transform each row, producing a new DataFrame with a different schema.
    *
    * Example:
    * {{{
    * df.mapRows { row =>
    *   (defaultNum = if row.default == "Yes" then 1.0 else 0.0,
    *    balance    = row.balance,
    *    income     = row.income)
    * }
    * }}}
    */
  def mapRows[ColNames2 <: Tuple, Values2 <: Tuple](
      f: NamedTuple[ColNames, Values] => NamedTuple[ColNames2, Values2]
  ): DataFrame[ColNames2, Values2] =
    new DataFrame[ColNames2, Values2](data.map(f))

  /** Filter rows, producing a new DataFrame with the same schema. */
  def filter(
      f: NamedTuple[ColNames, Values] => Boolean
  ): DataFrame[ColNames, Values] =
    new DataFrame[ColNames, Values](data.filter(f))

  /** Drop a single column by [[Col]] reference.
    *
    * Example: {{{df.drop(Col["age"])}}}
    */
  inline def drop[L <: String & Singleton](col: Col[L])(using
      ev: NamedTupleOps.DropField[ColNames, Values, L]
  ): DataFrame[
    NamedTupleOps.DropName[ColNames, L],
    NamedTupleOps.DropType[Values, ColNames, L]
  ] =
    new DataFrame(
      data.map(row =>
        ev(row.toTuple).asInstanceOf[NamedTuple[
          NamedTupleOps.DropName[ColNames, L],
          NamedTupleOps.DropType[Values, ColNames, L]
        ]]
      )
    )

  /** Select (and optionally reorder) columns by passing a tuple of [[Col]]
    * phantoms.
    *
    * Example: {{{df.select((Col["age"], Col["income"]))}}}
    */
  inline def select[C <: Tuple](cols: C)(using
      ev: NamedTupleOps.SelectFields[
        ColNames,
        Values,
        NamedTupleOps.ExtractLabels[C]
      ]
  ): DataFrame[
    NamedTupleOps.ExtractLabels[C],
    NamedTupleOps.LookupTypes[ColNames, Values, NamedTupleOps.ExtractLabels[C]]
  ] =
    new DataFrame(
      data.map(row =>
        ev(row.toTuple).asInstanceOf[NamedTuple[
          NamedTupleOps.ExtractLabels[C],
          NamedTupleOps.LookupTypes[
            ColNames,
            Values,
            NamedTupleOps.ExtractLabels[C]
          ]
        ]]
      )
    )

  /** Drop multiple columns by passing a tuple of [[Col]] phantoms.
    *
    * Example: {{{df.dropColumns((Col["age"], Col["income"]))}}}
    */
  inline def dropColumns[C <: Tuple](cols: C)(using
      ev: NamedTupleOps.SelectFields[
        ColNames,
        Values,
        NamedTupleOps.KeepNames[ColNames, NamedTupleOps.ExtractLabels[C]]
      ]
  ): DataFrame[
    NamedTupleOps.KeepNames[ColNames, NamedTupleOps.ExtractLabels[C]],
    NamedTupleOps.LookupTypes[ColNames, Values, NamedTupleOps.KeepNames[
      ColNames,
      NamedTupleOps.ExtractLabels[C]
    ]]
  ] =
    new DataFrame(
      data.map(row =>
        ev(row.toTuple).asInstanceOf[NamedTuple[
          NamedTupleOps.KeepNames[ColNames, NamedTupleOps.ExtractLabels[C]],
          NamedTupleOps.LookupTypes[
            ColNames,
            Values,
            NamedTupleOps.KeepNames[ColNames, NamedTupleOps.ExtractLabels[C]]
          ]
        ]]
      )
    )

  inline def show(n: Int = 5): String =
    val keys = constValueTuple[ColNames].toList.map(_.toString)
    val sample = data.take(n)
    val header = keys.mkString(", ")
    val body = sample.map(_.toList.mkString(", ")).mkString("\n")
    s"DataFrame[${data.size} rows]\n$header\n$body"

  override def toString: String = s"DataFrame[${data.size} rows]"

object DataFrame:

  import scala.compiletime.{erasedValue, summonInline, error}

  /** Applier for [[DataFrame.updateCol]] — carries `L` so `A` can be inferred
    * from the function.
    */
  final class UpdateColApplier[
      L <: String & Singleton,
      CN <: Tuple,
      V <: Tuple
  ](
      private val data: IndexedSeq[NamedTuple[CN, V]]
  ):
    def apply[A](f: NamedTuple[CN, V] => A)(using
        ev: NamedTupleOps.UpdateField[CN, V, L, A]
    ): DataFrame[CN, NamedTupleOps.UpdatedType[V, CN, L, A]] =
      new DataFrame(
        data.map(row =>
          ev(row.toTuple, f(row)).asInstanceOf[NamedTuple[
            CN,
            NamedTupleOps.UpdatedType[V, CN, L, A]
          ]]
        )
      )

  /** Load a CSV file from disk, schema driven by a named-tuple type parameter.
    *
    * Example:
    * {{{DataFrame.fromCSV[(name: String, age: Int, income: Double)](new File("data.csv"))}}}
    */
  inline def fromCSV[NT <: AnyNamedTuple](file: File)(using
      FromRow[DropNames[NT]]
  ): DataFrame[Names[NT], DropNames[NT]] =
    val text = Source.fromFile(file).mkString
    fromCsvText[Names[NT], DropNames[NT]](text)

  /** Load a CSV from a URL using the `requests` library. */
  inline def fromUrl[NT <: AnyNamedTuple](url: String)(using
      FromRow[DropNames[NT]]
  ): DataFrame[Names[NT], DropNames[NT]] =
    val resp = requests.get(url)
    require(
      resp.statusCode == 200,
      s"Failed to fetch $url: HTTP ${resp.statusCode}"
    )
    fromCsvText[Names[NT], DropNames[NT]](resp.text())

  /** Write a DataFrame to a CSV file. */
  inline def writeCSV[Names <: Tuple, Types <: Tuple](
      df: DataFrame[Names, Types],
      file: File
  )(using
      ToRow[Types]
  ): Unit =
    val keys = constValueTuple[Names].toList.map(_.toString)
    val writer = new java.io.PrintWriter(file)
    try
      writer.println(keys.mkString(","))
      df.data.foreach { row =>
        writer.println(
          summon[ToRow[Types]].serialise(row.asInstanceOf[Types]).mkString(",")
        )
      }
    finally writer.close()

  private inline def fromCsvText[Names <: Tuple, Types <: Tuple](text: String)(
      using fr: FromRow[Types]
  ): DataFrame[Names, Types] =
    val (rawHeader, rawRows) = CsvParser.parse(text)
    val expectedKeys = constValueTuple[Names].toList.map(_.toString)
    // Drop a leading unnamed index column if present (common in R/Python exports)
    val (csvHeader, csvRows) =
      if rawHeader.headOption.exists(h => h.isEmpty || h == "") then
        (rawHeader.tail, rawRows.map(_.tail))
      else (rawHeader, rawRows)
    // Validate all expected columns are present (order-independent)
    val missing = expectedKeys.filterNot(csvHeader.contains)
    require(
      missing.isEmpty,
      s"CSV is missing columns: $missing\n  Available: $csvHeader"
    )
    // Build a reorder index: for each expected key, the position in the CSV header
    val colIndex: Array[Int] =
      expectedKeys.map(k => csvHeader.indexOf(k)).toArray
    val rows = csvRows.map { cells =>
      val reordered = colIndex.map(i => cells(i)).toList
      fr.parse(reordered).asInstanceOf[NamedTuple[Names, Types]]
    }
    new DataFrame[Names, Types](rows)
