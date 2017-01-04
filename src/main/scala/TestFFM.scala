import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification._
import org.apache.spark.rdd.RDD


object TestFFM extends App {

  override def main(args: Array[String]): Unit = {

    val sc = new SparkContext(new SparkConf().setAppName("TESTFFM").setMaster("local[4]"))

    val data = sc.textFile("data/a9a_ffm").map(_.split("\\s")).map(x => {
      val y = if(x(0).toInt > 0 ) 1.0 else -1.0
      val nodeArray: Array[FFMNode] = x.drop(1).map(_.split(":")).map(x => {
        val node = new FFMNode; node.f = x(0).toInt; node.j = x(1).toInt; node.v = x(2).toDouble; node
      })

      (y, nodeArray)
    }).repartition(4)

    val splits = data.randomSplit(Array(0.7, 0.3))
    val (training, testing) = (splits(0), splits(1))

    val m = training.flatMap(x=>x._2).map(_.f).collect.reduceLeft(_ max _) + 1
    val n = training.flatMap(x=>x._2).map(_.j).collect.reduceLeft(_ max _) + 1

    val param: FFMParameter = new FFMParameter().defaultParameter
    param.eta = 0.1
    param.lambda = 0.00002
    param.n_iters = 10
    param.k = 2
    param.normalization = true
    param.random = false

    val ffm: FFMModel = FFMWithAdag.train(training, m, n, param)
    val scores: RDD[(Double, Double)] = testing.map(x => {
      val p = ffm.predict(x._2)
      val ret = if (p >= 0.5) 1.0 else -1.0
      (ret, x._1)
    })
    val accuracy = scores.filter(x => x._1 == x._2).count().toDouble / scores.count()
    println(s"accuracy = $accuracy")
  }
}

