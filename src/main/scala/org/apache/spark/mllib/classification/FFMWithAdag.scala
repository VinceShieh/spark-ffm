package org.apache.spark.mllib.classification

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.optimization._


import scala.util.Random
/**
  * Created by vincent on 17-1-4.
  */
class FFMWithAdag(param: FFMParameter, m: Int, n: Int) extends Serializable {

  private val k = param.k
  /**
    * Run the algorithm with the configured parameters on an input RDD
    * of FFMNode entries.
    */
  def run(input: RDD[(Double, Array[FFMNode])]): FFMModel = {
    val gradient = new FFMGradient(m, n, k)
    val optimizer = new GradientDescentFFM(gradient, null, param)

    val initWeights = generateInitWeights()
    val weights = optimizer.optimize(input, initWeights, param)
    createModel(weights)
  }

  private def generateInitWeights(): Vector = {
    val W = new Array[Double](n * m * param.k *2)

    val coef = 0.5 / Math.sqrt(param.k)
    val random = new Random()

    var position = 0
    for(j <-0 to n-1){
      for(f <- 0 to m-1){
        for(d <- 0 to param.k - 1){
          W(position) = coef * random.nextDouble()
          position += 1
        }
        for(d <- param.k to 2 * param.k - 1){
          W(position) = 1.0
          position += 1
        }
      }
    }
    Vectors.dense(W)
  }

  /**
    * Create a FMModle from an encoded vector.
    */
  private def createModel(weights: Vector): FFMModel = {

    val values = weights.toArray

    new FFMModel(n, m, param, values)
  }
}

object FFMWithAdag {
  /**
    *
    * @param data
    * @param m
    * @param n
    * @param param
    * @return
    */
  def train(data: RDD[(Double, Array[FFMNode])], m: Int, n: Int, param: FFMParameter): FFMModel = {
    new FFMWithAdag(param, m, n)
      .run(data)
  }
}