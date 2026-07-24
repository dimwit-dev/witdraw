package witdraw.examples.cleaning

import witdraw.dataframe.DataFrame
import witdraw.dataframe.updated
import witdraw.dataframe.Col

object Cleaning:

  @main
  def clean(): Unit =

    dimwit.initialize()

    val dataDir = new java.io.File(System.getProperty("user.dir"))
    type Schema = (
        Symbol: String,
        Sector: String,
        Fiscal_period: String,
        Fiscal_period_end: String,
        MarketCap_BCHF: Double,
        SharePrice_CHF: Double,
        Volume_Shares: Long,
        EPS_CHF: Double | Null,
        DivYield_Prct: Double,
        Revenue_BCHF: String,
        RevenueGrowth_Prct: String,
        GrossProfit_BCHF: String,
        OperatingIncome_BCHF: String,
        EBITDA_BCHF: String,
        EnterpriseValue_BCHF: String
    )

    import scala.NamedTuple.*
    type StocksDF = DataFrame[Names[Schema], DropNames[Schema]]
    val df: StocksDF = DataFrame.fromCSV[Schema](
      new java.io.File(dataDir, "data/large_caps_CH_2026-01-31.csv")
    )

    // positive dividend yield
    val df_positive_yield = DataFrame(
      df.rows.filter(row => row.DivYield_Prct > 0 && row.EPS_CHF != null)
    )

    println(df_positive_yield.col(_.Sector).toSet.mkString("\n "))

    println(df_positive_yield.drop(Col["Sector"]).show(10))
    val a = df_positive_yield.col(_.Sector)

    val df3 =
      df.select((Col["Symbol"], Col["Sector"], Col["MarketCap_BCHF"]))

    // Same result using mapRows with chained .updated on NamedTuple (both type params required)
    val df5 = df.mapRows(row =>
      row
        .updated(Col["MarketCap_BCHF"])(row.MarketCap_BCHF * 1000)
        .updated(Col["SharePrice_CHF"])(row.SharePrice_CHF * 1000)
    )

    println(df5.show(3))
    val result =
      DataFrame(
        df.rows
          .groupBy(_.Sector)
          .map((sector, rows) =>
            (
              Sector = sector,
              MeanMarketCap = rows.map(_.MarketCap_BCHF).sum / rows.size
            )
          )
          .toIndexedSeq
      )
    val z =
      df.rows.map(row =>
        row.updated(Col["MarketCap_BCHF"])(row.MarketCap_BCHF * 1000)
      )
    result.show(5)
