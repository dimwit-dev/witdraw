package witdraw
package dataframe

import java.io.File
import scala.NamedTuple.*
import scala.compiletime.constValueTuple
import scala.io.Source

// ---------------------------------------------------------------------------
// CsvParser – minimal RFC 4180-compatible CSV splitter
// ---------------------------------------------------------------------------

private[witdraw] object CsvParser:

  /** Split a single CSV line into fields, respecting double-quoted fields. */
  def splitLine(line: String): List[String] =
    val fields = List.newBuilder[String]
    val buf = new StringBuilder
    var inQuote = false
    var i = 0
    while i < line.length do
      val c = line.charAt(i)
      if inQuote then
        if c == '"' then
          if i + 1 < line.length && line.charAt(i + 1) == '"' then
            buf += '"'; i += 1 // escaped quote
          else inQuote = false
        else buf += c
      else if c == '"' then inQuote = true
      else if c == ',' then
        fields += buf.result(); buf.clear()
      else buf += c
      i += 1
    fields += buf.result()
    fields.result()

  /** Parse CSV text into (header, rows). */
  def parse(text: String): (List[String], IndexedSeq[List[String]]) =
    val lines = text.linesIterator.filter(_.nonEmpty).toIndexedSeq
    require(lines.nonEmpty, "CSV is empty")
    val header = splitLine(lines.head)
    val rows = lines.tail.map(splitLine)
    (header, rows)
