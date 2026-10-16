ThisBuild / version := "0.1-SNAPSHOT"
ThisBuild / scalaVersion := "3.9.0"
ThisBuild / organization := "ch.contrafactus"
ThisBuild / versionScheme := Some("early-semver")
ThisBuild / resolvers += "Central Portal Snapshots" at "https://central.sonatype.com/repository/maven-snapshots/"

// Publishing to Sonatype Central. The `ch.contrafactus` namespace is verified once for the whole
// organisation, so witdraw needs no verification of its own.
ThisBuild / sonatypeCredentialHost := "central.sonatype.com"
ThisBuild / publishTo := {
  if (isSnapshot.value)
    Some("central-snapshots" at "https://central.sonatype.com/repository/maven-snapshots/")
  else
    sonatypePublishToBundle.value
}
ThisBuild / homepage := Some(url("https://github.com/dimwit-dev/witdraw"))
ThisBuild / licenses := List("Apache-2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0"))
ThisBuild / scmInfo := Some(
  ScmInfo(
    url("https://github.com/dimwit-dev/witdraw"),
    "scm:git@github.com:dimwit-dev/witdraw.git"
  )
)
ThisBuild / developers := List(
  Developer(
    id = "dimwit-dev",
    name = "WitDraw Contributors",
    email = "",
    url = url("https://github.com/dimwit-dev")
  )
)

// Publishes `core` only. `examples` depends on witling and plotwit, which are not published.
addCommandAlias("sonaUploadCore", "; project core; sonatypeCentralUpload; project root")

import scala.concurrent.duration.DurationInt
import lmcoursier.definitions.CachePolicy

csrConfiguration := csrConfiguration.value
  .withTtl(Some(0.seconds))
  .withCachePolicies(Vector(CachePolicy.FetchMissing))

resolvers += Resolver.defaultLocal

lazy val root = (project in file("."))
  .aggregate(core, examples)
  .settings(
    name := "witdraw-root",
    publish / skip := true,
    publishLocal / skip := true,
    publishArtifact := false
  )

lazy val core = (project in file("core"))
  .settings(
    name := "witdraw-core",
    description := "Typed probabilistic programming for Scala 3, built on DimWit",
    libraryDependencies ++= Seq(
      "ch.contrafactus" %% "dimwit-core" % "0.2.0"
    ),
    Compile / packageSrc / publishArtifact := true,
    Compile / packageDoc / publishArtifact := true,
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
    // Don't publish examples
    publish / skip := true,
    publishLocal / skip := true,
    publishArtifact := false,
    run / baseDirectory := (ThisBuild / baseDirectory).value / "examples"
  )

lazy val docs =
  (project in file(".witdraw-docs")) // Hidden folder for sbt metadata
    .dependsOn(root)
    .enablePlugins(MdocPlugin)
    .settings(
      name := "witdraw-docs",
      publish / skip := true,
      mdocIn := (ThisBuild / baseDirectory).value / "docs/mdoc",
      mdocOut := (ThisBuild / baseDirectory).value / "docs",
      mdocExtraArguments := Seq("--no-link-hygiene"),
      mdocVariables := Map(
        "VERSION" -> version.value
      ),
      fork := true,
      envVars := (ThisBuild / envVars).value
    )
