name := "Spark-FFM-paralsgd"

version := "0.0.1"

scalaVersion := "2.11.8"

libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "2.0.0" % "provided"
libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.0.0" % "provided"
libraryDependencies += "it.unimi.dsi" % "fastutil" % "7.0.2"
resolvers += Resolver.sonatypeRepo("public")
