package org.apache.spark.mllib.classification

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.optimization._
import org.apache.spark.network.protocol.Encoders.Strings

import scala.util.Random
/**
  * Created by vincent on 17-1-4.
  */
class FFMWithAdag(m: Int, n: Int, k_num: Int, n_iters: Int, eta: Double, lambda: Double,
                  normalization: Boolean, random: Boolean, solver: String) extends Serializable {
  private val k = k_num
  private val sgd = setOptimizer(solver)

  private def generateInitWeights(): Vector = {
    val W = if(sgd){
      new Array[Double](n * m * k)
    } else {
      new Array[Double](n * m * k *2)
    }
    val coef = 1.0 / Math.sqrt(k)
    val random = new Random()
    var position = 0
    if(sgd) {
      for (j <- 0 to n - 1; f <- 0 to m - 1; d <- 0 to k - 1) {
        W(position) = coef * random.nextDouble()
        position += 1
      }
    } else {
      for (j <- 0 to n - 1; f <- 0 to m - 1; d <- 0 to 2 * k - 1) {
        W(position) = if (d < k) coef * random.nextDouble() else 1.0
        position += 1
      }
    }
    Vectors.dense(W)
  }

  /**
    * Create a FFMModle from an encoded vector.
    */
  private def createModel(weights: Vector): FFMModel = {
    val values = weights.toArray
    new FFMModel(n, m, k, n_iters, eta, lambda, normalization, random, values, sgd)
  }

  /**
    * Run the algorithm with the configured parameters on an input RDD
    * of FFMNode entries.
    */
  def run(input: RDD[(Double, Array[(Int, Int, Double)])]): FFMModel = {
    val gradient = new FFMGradient(m, n, k, sgd)
    val optimizer = new GradientDescentFFM(gradient, null, k, n_iters, eta, lambda, normalization, random)

    val initWeights = generateInitWeights()
    val weights = optimizer.optimize(input, initWeights,n_iters, eta, lambda, sgd)
    createModel(weights)
  }

  def setOptimizer(op: String): Boolean = {
    if("sgd" == op) true else false
  }

}

object FFMWithAdag {
  /**
    *
    * @param data
    * @param m
    * @param n
    * @param k
    * @param n_iters
    * @param eta
    * @param lambda
    * @param normalization
    * @param random
    * @param solver
    * @return
    */
  def train(data: RDD[(Double, Array[(Int, Int, Double)])], m: Int, n: Int,
            k: Int, n_iters: Int, eta: Double, lambda: Double, normalization: Boolean, random: Boolean,
            solver: String = "sgd"): FFMModel = {
    new FFMWithAdag(m, n, k, n_iters, eta, lambda, normalization, random, solver)
      .run(data)
  }
}