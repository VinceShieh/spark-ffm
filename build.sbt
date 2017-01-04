name := "Spark-FFM"

version := "0.0.1"

scalaVersion := "2.11.8"

libraryDependencies += "org.apache.spark" % "spark-core_2.11" % "2.0.0"
libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.0.0"

resolvers += Resolver.sonatypeRepo("public")
