import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.rdd.RDD


object TestFFM extends App {

  override def main(args: Array[String]): Unit = {

    val sc = new SparkContext(new SparkConf().setAppName("TESTFFM").setMaster("local[4]"))

    if (args.length != 7) {
      println("testFFM <train_file> <k> <n_iters> <eta> <lambda> " + "<normal> <random>")
    }

    val data= sc.textFile(args(0)).map(_.split("\\s")).map(x => {
      val y = if(x(0).toInt > 0 ) 1.0 else -1.0
      val nodeArray: Array[(Int, Int, Double)] = x.drop(1).map(_.split(":")).map(x => {
        (x(0).toInt, x(1).toInt, x(2).toDouble)
      })
      (y, nodeArray)
    }).repartition(4)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (training: RDD[(Double, Array[(Int, Int, Double)])], testing) = (splits(0), splits(1))

    val m = training.flatMap(x=>x._2).map(_._1).collect.reduceLeft(_ max _) //+ 1
    val n = training.flatMap(x=>x._2).map(_._2).collect.reduceLeft(_ max _) //+ 1

    val ffm: FFMModel = FFMWithAdag.train(training, m, n, k = args(1).toInt, n_iters = args(2).toInt,
      eta = args(3).toDouble, lambda = args(4).toDouble, normalization = args(5).toBoolean,
      random = args(6).toBoolean, "adagrad")

    val scores: RDD[(Double, Double)] = testing.map(x => {
      val p = ffm.predict(x._2)
      val ret = if (p >= 0.5) 1.0 else -1.0
      (ret, x._1)
    })
    val accuracy = scores.filter(x => x._1 == x._2).count().toDouble / scores.count()
    println(s"accuracy = $accuracy")
  }
}

