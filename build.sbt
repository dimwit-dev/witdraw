name := "witdraw"

version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.9.0"
ThisBuild / resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"
ThisBuild / organization := "ch.contrafactus"

import scala.concurrent.duration.DurationInt
import lmcoursier.definitions.CachePolicy

csrConfiguration := csrConfiguration.value
  .withTtl(Some(0.seconds))
  .withCachePolicies(Vector(CachePolicy.FetchMissing))

resolvers += Resolver.defaultLocal

lazy val root = (project in file("."))
  .aggregate(core, examples)

lazy val core = (project in file("core"))
  .settings(
    name := "witdraw-core",
    libraryDependencies ++= Seq(
      "ch.contrafactus" %% "dimwit-core" % "0.2.0" changing (),
      "ch.contrafactus" %% "deepwit-core" % "0.2-SNAPSHOT" changing ()
    ),
    fork := true
  )

lazy val examples = (project in file("examples"))
  .dependsOn(core)
  .settings(
    name := "witdraw-examples",
    libraryDependencies ++= Seq(
      "org.scala-lang" %% "toolkit" % "0.9.1",
      "ch.contrafactus" %% "witling-core" % "0.1-SNAPSHOT" changing (),
      "ch.contrafactus" %% "plotwit-core" % "0.2-SNAPSHOT" changing ()
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
