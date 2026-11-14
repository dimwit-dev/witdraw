#!/usr/bin/env bash
# Start a Scala 3 REPL with a project's classpath (default: examples).
#
# `sbt console` fails with Scala 3.9, because its REPL needs JLine 4 while
# sbt 1.x puts its own JLine 3 on the console classpath. This starts the
# REPL in its own JVM instead.
#
# Usage: ./repl.sh [project]    (activate the Python env first)
set -euo pipefail
cd "$(dirname "$0")"

project="${1:-examples}"
scala_version=$(grep -oP 'scalaVersion := "\K[^"]+' build.sbt)

cp=$(sbt -batch -error "$project/Runtime/fullClasspath" "export $project/Runtime/fullClasspath" | tail -1)
repl_cp=$(cs fetch --classpath \
  "org.scala-lang:scala3-repl_3:$scala_version" \
  "org.scala-lang:scala-library:$scala_version")

# Tell ScalaPy where to find libpython of the active Python env
py_libdir=$(python -c "import sysconfig as s; print(s.get_config_var('LIBDIR'))")
py_version=$(python -c "import sysconfig as s; print(s.get_config_var('LDVERSION'))")

# Same working directory as `sbt run` (examples use os.pwd / "data")
[ -d "$project" ] && cd "$project"

exec java \
  -Djna.library.path="$py_libdir" \
  -Dscalapy.python.library="python$py_version" \
  -Dscalapy.python.programname="$(command -v python)" \
  -cp "$repl_cp" dotty.tools.repl.Main \
  -classpath "$cp:$repl_cp"
