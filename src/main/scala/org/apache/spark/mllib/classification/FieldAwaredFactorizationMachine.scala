package org.apache.spark.mllib.classification

import java.io._

import breeze.linalg.{DenseVector => BDV}

import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.optimization.Gradient

import scala.util.Random

/**
  * Created by vincent on 16-12-19.
  */
/**
  *
  * @param numFeatures
  * @param numFields
  * @param param
  */
class FFMModel(val numFeatures: Int,
               val numFields: Int,
               val param: FFMParameter,
               val weights: Array[Double]) extends Serializable {

  private var n: Int = numFeatures
  //numFeatures
  private var m: Int = numFields
  //numFields
  private var k: Int = param.k
  //numFactors
  private var normalization: Boolean = param.normalization
  private var initMean: Double = 0
  private var initStd: Double = 0.01


  require(n > 0 && k > 0 && m > 0)

  def radomization(l: Int, rand: Boolean): Array[Int] = {
    val order = Array.fill(l)(0)
    for (i <- 0 to l - 1) {
      order(i) = i
    }
    if (rand) {
      val rand = new Random()
      for (i <- l - 1 to 1) {
        val tmp = order(i - 1)
        val index = rand.nextInt(i)
        order(i - 1) = order(index)
        order(index) = tmp
      }
    }
    return order
  }

  def predict(data: Array[FFMNode], r: Double = 1.0): Double = {
    var t = 0.0
    val align0: Int = k * 2
    val align1: Int = m * k * 2
    for(n1 <- 0 to data.size - 1; n2 <- n1 + 1 to data.size - 1) {
      val j1 = data(n1).j
      val f1 = data(n1).f
      val v1 = data(n1).v
      val j2 = data(n2).j
      val f2 = data(n2).f
      val v2 = data(n2).v
      if(j1 < n && f1 < m && j2 < n && f2 < m) {
        val w1_index: Int = j1 * align1 + f2 * align0
        val w2_index: Int = j2 * align1 + f1 * align0
        val v: Double = 2.0 * v1 * v2 * r
        for (d <- 0 to k - 1) {
          t += weights(w1_index + d) * weights(w2_index + d) * v
        }
      }
    }
    t
  }
}

class FFMGradient(m: Int, n: Int, k:Int) extends Gradient {
  private def predict (data: Array[FFMNode], weights: Array[Double], r: Double = 1.0): Double = {
    var t = 0.0
    val align0: Int = k * 2
    val align1: Int = m * k * 2
    for(n1 <- 0 to data.size - 1; n2 <- n1 + 1 to data.size - 1) {
      val j1 = data(n1).j
      val f1 = data(n1).f
      val v1 = data(n1).v
      val j2 = data(n2).j
      val f2 = data(n2).f
      val v2 = data(n2).v
      if(j1 < n && f1 < m && j2 < n && f2 < m) {
        val w1_index: Int = j1 * align1 + f2 * align0
        val w2_index: Int = j2 * align1 + f1 * align0
        val v: Double = 2.0 * v1 * v2 * r
        for (d <- 0 to k - 1) {
          t += weights(w1_index + d) * weights(w2_index + d) * v
        }
      }
    }
    t
  }
  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    throw new Exception("This part is merged into computeFFM()")
  }

  override def compute(data: Vector, label: Double, weights: Vector, cumGradient: Vector): Double = {
    throw new Exception("This part is merged into computeFFM()")
  }
  def computeFFM(label: Double, data2: Array[FFMNode], weights: Vector,
                 r: Double = 1.0, eta: Double, lambda: Double, do_update: Boolean, iter: Int): (BDV[Double], Double) = {
    val data = data2.toVector
    val weightsArray: Array[Double] = weights.asInstanceOf[DenseVector].values
    val t = predict(data2, weightsArray, r)
    val expnyt = math.exp(-label * t)
    val tr_loss = math.log(1 + expnyt)
    val kappa = -label * expnyt / (1 + expnyt)
    val align0: Int = k * 2
    val align1: Int = m * k * 2

    for(n1 <- 0 to data.size - 1; n2 <- n1 + 1 to data.size - 1) {
      val j1 = data(n1).j
      val f1 = data(n1).f
      val v1 = data(n1).v
      val j2 = data(n2).j
      val f2 = data(n2).f
      val v2 = data(n2).v
      if(j1 < n && f1 < m && j2 < n && f2 < m) {
        val w1_index: Int = j1 * align1 + f2 * align0
        val w2_index: Int = j2 * align1 + f1 * align0
        val v: Double = 2.0 * v1 * v2 * r
        val wg1_index: Int = w1_index + k
        val wg2_index: Int = w2_index + k
        val kappav: Double = kappa * v
        for(d <- 0 to k-1) {
          val g1: Double = lambda * weightsArray(w1_index + d) + kappav * weightsArray(w2_index + d)
          val g2: Double = lambda * weightsArray(w2_index + d) + kappav * weightsArray(w1_index + d)
          val wg1: Double = weightsArray(wg1_index + d) + g1 * g1
          val wg2: Double = weightsArray(wg2_index + d) + g2 * g2
          weightsArray(w1_index + d) -= eta / (math.sqrt(wg1)) * g1
          weightsArray(w2_index + d) -= eta / (math.sqrt(wg2)) * g2

          weightsArray(wg1_index + d) = wg1
          weightsArray(wg2_index + d) = wg2
        }
      }
    }

    (BDV(weightsArray), tr_loss)
  }
}
/**
  * FFMParameter
  */
class FFMParameter extends Serializable {
  var eta: Double = 0.0
  var lambda: Double = 0.0
  var n_iters: Int = 0
  var k: Int = 0
  var normalization: Boolean = false
  var random: Boolean = false

  def defaultParameter: FFMParameter = {
    val parameter: FFMParameter = new FFMParameter
    parameter.eta = 0.1
    parameter.lambda = 0.0
    parameter.n_iters = 15
    parameter.k = 4
    parameter.normalization = true
    parameter.random = true
    return parameter
  }
}

/**
  * FFMNode
  */
class FFMNode extends Serializable {
  var v: Double = 0.0
  // field_num
  var f: Int = 0
  // feature_num
  var j: Int = 0
  // value
}