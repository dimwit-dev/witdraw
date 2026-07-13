name := "witdraw"

version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.8.1"
ThisBuild / resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"

organization := "ch.contrafactus"

import scala.concurrent.duration.DurationInt
import lmcoursier.definitions.CachePolicy

csrConfiguration := csrConfiguration.value
  .withTtl(Some(0.seconds))
  .withCachePolicies(Vector(CachePolicy.FetchMissing))

resolvers += Resolver.defaultLocal

lazy val root = (project in file("."))
  .aggregate(core, plotting, dataframe, examples)

lazy val core = (project in file("core"))
  .settings(
    libraryDependencies ++= Seq(
      "ch.contrafactus" %% "dimwit-core" % "0.1.0-SNAPSHOT" changing (),
      "ch.contrafactus" %% "deepwit-core" % "0.1.0-SNAPSHOT" changing ()
    ),
    fork := true
  )

lazy val plotting = (project in file("plotting"))
  .dependsOn(core)
  .settings(
    name := "witdraw-plotting",
    libraryDependencies ++= Seq(
      "io.github.quafadas" %% "dedav4s" % "0.10.5"
    ),
    fork := true,
    run / baseDirectory := (ThisBuild / baseDirectory).value / "plotting"
  )

lazy val dataframe = (project in file("dataframe"))
  .dependsOn(core)
  .settings(
    name := "witdraw-dataframe",
    libraryDependencies ++= Seq(
      "io.github.quafadas" %% "dedav4s" % "0.10.5"
    ),
    fork := true,
    run / baseDirectory := (ThisBuild / baseDirectory).value / "dataframe"
  )

lazy val examples = (project in file("examples"))
  .dependsOn(core, plotting, dataframe)
  .settings(
    name := "witdraw-examples",
    libraryDependencies ++= Seq(
      "org.scala-lang" %% "toolkit" % "0.9.1",
      "com.lihaoyi" %% "requests" % "0.9.3"
    ),
    fork := true,
    run / baseDirectory := (ThisBuild / baseDirectory).value / "examples"
  )

lazy val docs =
  (project in file(".witdraw-docs")) // Hidden folder for sbt metadata
    .dependsOn(root)
    .enablePlugins(MdocPlugin)
    .settings(
      name := "witdraw-docs",
      mdocIn := (ThisBuild / baseDirectory).value / "docs/mdoc",
      mdocOut := (ThisBuild / baseDirectory).value / "docs",
      mdocExtraArguments := Seq("--no-link-hygiene"),
      mdocVariables := Map(
        "VERSION" -> version.value
      ),
      fork := true,
      envVars := (ThisBuild / envVars).value
    )
