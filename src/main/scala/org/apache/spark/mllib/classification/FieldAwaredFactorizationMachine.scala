/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.classification

import java.io._

import breeze.linalg.{DenseVector => BDV}

import org.json4s.DefaultFormats
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.optimization.Gradient
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.mllib.util.Loader._
import org.apache.spark.mllib.util.{Loader, Saveable}

import scala.util.Random

import scala.collection.mutable.WrappedArray
/**
  * Created by vincent on 16-12-19.
  */
/**
  *
  * @param numFeatures number of features
  * @param numFields number of fields
  * @param dim A (Boolean,Boolean,Int) 3-Tuple stands for whether the global bias term should be used, whether the
  *            one-way interactions should be used, and the number of factors that are used for pairwise
  *            interactions, respectively.
  * @param n_iters number of iterations
  * @param eta step size to be used for each iteration
    * @param regParam A (Double, Double, Double) 3-Tuple stands for regularization params of bias, one-way interactions and pairwise interactions
  * @param isNorm whether normalize data
  * @param random whether randomize data
  * @param weights weights of FFMModel
  * @param sgd "true": parallelizedSGD, parallelizedAdaGrad would be used otherwise
  */
class FFMModel(val numFeatures: Int,
               val numFields: Int,
               val dim: (Boolean, Boolean, Int),
               val n_iters: Int,
               val eta: Double,
               val regParam: (Double, Double, Double),
               val isNorm: Boolean,
               val random: Boolean,
               val weights: Array[Double],
               val sgd: Boolean = true ) extends Serializable with Saveable {

  private var n: Int = numFeatures
  //numFeatures
  private var m: Int = numFields
  //numFields
  private var k: Int = dim._3
  //numFactors
  private var k0 = dim._1
  private var k1 = dim._2
  private var normalization: Boolean = isNorm
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

  def setOptimizer(op: String): Boolean = {
    if("sgd" == op) true else false
  }

  def predict(data: Array[(Int, Int, Double)], r: Double = 1.0): Double = {

    var t = if (k0) weights(weights.length - 1) else 0.0

    val (align0, align1) = if(sgd) {
      (k, m * k)
    } else {
      (k * 2, m * k * 2)
    }

    // j: feature, f: field, v: value
    val valueSize = data.size //feature length
    val indicesArray = data.map(_._2) //feature index
    val valueArray: Array[(Int, Double)] = data.map(x => (x._1, x._3))
    var i = 0
    var ii = 0
    val pos = if (sgd) n * m * k else n * m * k * 2
    // j: feature, f: field, v: value
    while (i < valueSize) {
      val j1 = data(i)._2 - 1
      val f1 = data(i)._1
      val v1 = data(i)._3

      if(k1) t += weights(pos + j1) * v1

      ii = i + 1
      if (j1 < n && f1 < m) {
        while (ii < valueSize) {
          val j2 = data(ii)._2
          val f2 = data(ii)._1
          val v2 = data(ii)._3
          if (j2 < n && f2 < m) {
            val w1_index: Int = j1 * align1 + f2 * align0
            val w2_index: Int = j2 * align1 + f1 * align0
            val v: Double = v1 * v2 * r
            for (d <- 0 to k - 1) {
              t += weights(w1_index + d) * weights(w2_index + d) * v
            }
          }
          ii += 1
        }
      }
      i += 1
    }
    1 / (1 + math.exp(-t))
  }

  override protected def formatVersion: String = "1.0"

  override def save(sc: SparkContext, path: String): Unit = {
    val data = FFMModel.SaveLoadV1_0.Data(numFeatures, numFields, dim._1, dim._2, dim._3, n_iters, eta, regParam._1, regParam._2, regParam._3, isNorm, random, weights, sgd)
    FFMModel.SaveLoadV1_0.save(sc, path, data)
  }
}

object FFMModel extends Loader[FFMModel] {

  private object SaveLoadV1_0 {

    def thisFormatVersion = "1.0"

    def thisClassName = "com.intel.imllib.ffm.classification.FFMModel$SaveLoadV1_0$"

    /** Model data for model import/export */
    case class Data(numFeatures: Int, numFields: Int, dim0: Boolean, dim1: Boolean, dim2: Int,
                      n_iters: Int, eta: Double, reg1: Double, reg2:Double, reg3:Double, isNorm: Boolean,
                      random: Boolean, weights: Array[Double], sgd: Boolean)

    def save(sc: SparkContext, path: String, data: Data): Unit = {
      val sqlContext = new SQLContext(sc)
      import sqlContext.implicits._
      // Create JSON metadata.
      val metadata = compact(render(
        ("class" -> this.getClass.getName) ~ ("version" -> thisFormatVersion) ~
          ("numFeatures" -> data.numFeatures) ~ ("numFields" -> data.numFields) ~
          ("dim0" -> data.dim0) ~ ("dim1" -> data.dim1) ~ ("dim2" -> data.dim2)
          ~ ("n_iters" -> data.n_iters) ~ ("eta" -> data.eta) ~ ("reg1" -> data.reg1) ~ ("reg2" -> data.reg2) ~ ("reg3" -> data.reg3)
          ~ ("isNorm" -> data.isNorm) ~ ("random" -> data.random) ~ ("sgd" -> data.sgd)))
      sc.parallelize(Seq(metadata), 1).saveAsTextFile(metadataPath(path))

      // Create Parquet data.
      val dataRDD: DataFrame = sc.parallelize(Seq(data), 1).toDF()
      dataRDD.write.parquet(dataPath(path))
    }

    def load(sc: SparkContext, path: String): FFMModel = {
      val sqlContext = new SQLContext(sc)
      // Load Parquet data.
      val dataRDD = sqlContext.parquetFile(dataPath(path))
      // Check schema explicitly since erasure makes it hard to use match-case for checking.
      checkSchema[Data](dataRDD.schema)
      val dataArray = dataRDD.select("numFeatures", "numFields", "dim0", "dim1", "dim2", "n_iters", "eta", "reg1", "reg2", "reg3", "isNorm", "random", "weights", "sgd").take(1)
      assert(dataArray.length == 1, s"Unable to load FMModel data from: ${dataPath(path)}")
      val data = dataArray(0)
      val numFeatures = data.getInt(0)
      val numFields = data.getInt(1)
      val dim0 = data.getBoolean(2)
      val dim1 = data.getBoolean(3)
      val dim2 = data.getInt(4)
      val n_iters = data.getInt(5)
      val eta = data.getDouble(6)
      val reg1 = data.getDouble(7)
      val reg2 = data.getDouble(8)
      val reg3 = data.getDouble(9)
      val isNorm = data.getBoolean(10)
      val random = data.getBoolean(11)
      val weights = data.getAs[WrappedArray[Double]](12).toArray
      val sgd = data.getBoolean(13)
      val dim = (dim0, dim1, dim2)
      val regParam = (reg1, reg2, reg3)
      new FFMModel(numFeatures, numFields, dim, n_iters, eta, regParam, isNorm, random, weights, sgd)
    }
  }

  override def load(sc: SparkContext, path: String): FFMModel = {
    implicit val formats = DefaultFormats

    val (loadedClassName, version, metadata) = loadMetadata(sc, path)
    val classNameV1_0 = SaveLoadV1_0.thisClassName

    (loadedClassName, version) match {
      case (className, "1.0") if className == classNameV1_0 =>
        val numFeatures = (metadata \ "numFeatures").extract[Int]
        val numFields = (metadata \ "numFields").extract[Int]
        val model = SaveLoadV1_0.load(sc, path)
        assert(model.numFeatures == numFeatures,
          s"FFMModel.load expected $numFeatures features," +
            s" but model had ${model.numFeatures} featues")
        assert(model.numFields == numFields,
          s"FFMModel.load expected $numFields fields," +
            s" but model had ${model.numFields} fields")
        model

      case _ => throw new Exception(
        s"FFMModel.load did not recognize model with (className, format version):" +
          s"($loadedClassName, $version).  Supported:\n" +
          s"  ($classNameV1_0, 1.0)")
    }
  }
}


class FFMGradient(m: Int, n: Int, dim: (Boolean, Boolean, Int), sgd: Boolean = true) extends Gradient {

  private val k0 = dim._1
  private val k1 = dim._2
  private val k = dim._3

  private def predict (data: Array[(Int, Int, Double)], weights: Array[Double], r: Double = 1.0): Double = {

    var t = if (k0) weights(weights.length - 1) else 0.0

    val (align0, align1) = if(sgd) {
      (k, m * k)
    } else {
      (k * 2, m * k * 2)
    }
    val valueSize = data.size //feature length
    val indicesArray = data.map(_._2) //feature index
    val valueArray: Array[(Int, Double)] = data.map(x => (x._1, x._3))
    var i = 0
    var ii = 0
    val a = data.size
    val b = indicesArray.length
    val c = valueArray.length
    val tt = 0
    val pos = if (sgd) n * m * k else n * m * k * 2
    // j: feature, f: field, v: value
    while (i < valueSize) {
      val j1 = data(i)._2 - 1
      val f1 = data(i)._1
      val v1 = data(i)._3
      ii = i + 1
      if(k1) t += weights(pos + j1) * v1

      if (j1 < n && f1 < m) {
        while (ii < valueSize) {
        val j2 = data(ii)._2
        val f2 = data(ii)._1
        val v2 = data(ii)._3
        if (j2 < n && f2 < m) {
          val w1_index: Int = j1 * align1 + f2 * align0
          val w2_index: Int = j2 * align1 + f1 * align0
          val v: Double = v1 * v2 * r
          for (d <- 0 to k - 1) {
            t += weights(w1_index + d) * weights(w2_index + d) * v
          }
        }
        ii += 1
      }
    }
      i += 1
    }
    t
  }

  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    throw new Exception("This part is merged into computeFFM()")
  }

  override def compute(data: Vector, label: Double, weights: Vector, cumGradient: Vector): Double = {
    throw new Exception("This part is merged into computeFFM()")
  }
  def computeFFM(label: Double, data2: Array[(Int, Int, Double)], weights: Vector,
                 r: Double = 1.0, eta: Double, regParam: (Double,Double,Double),
                 do_update: Boolean, iter: Int, solver: Boolean = true): (BDV[Double], Double) = {
    val weightsArray: Array[Double] = weights.asInstanceOf[DenseVector].values
    val t = predict(data2, weightsArray, r)
    val expnyt = math.exp(-label * t)
    val tr_loss = math.log(1 + expnyt)
    val kappa = -label * expnyt / (1 + expnyt)
    val (align0, align1) = if(sgd) {
      (k, m * k)
    } else {
      (k * 2, m * k * 2)
    }
    val valueSize = data2.size //feature length
    val indicesArray = data2.map(_._2) //feature index
    val valueArray: Array[(Int, Double)] = data2.map(x => (x._1, x._3))
    var i = 0
    var ii = 0

    val r0 = regParam._1
    val r1 = regParam._2
    val r2 = regParam._3
    val pos = if (sgd) n * m * k else n * m * k * 2
    if(k0) weightsArray(weightsArray.length - 1) -= eta * (kappa + r0 * weightsArray(weightsArray.length - 1))
    // j: feature, f: field, v: value
    while (i < valueSize) {
      val j1 = data2(i)._2 - 1
      val f1 = data2(i)._1
      val v1 = data2(i)._3
      if(k1) weightsArray(pos + j1) -= eta * (v1 * kappa + r1 * weightsArray(pos + j1))
      if (j1 < n && f1 < m) {
        ii = i + 1
        while (ii < valueSize) {
          val j2 = data2(ii)._2
          val f2 = data2(ii)._1
          val v2 = data2(ii)._3
          if (j2 < n && f2 < m) {
            val w1_index: Int = j1 * align1 + f2 * align0
            val w2_index: Int = j2 * align1 + f1 * align0
            val v: Double = v1 * v2 * r
            val wg1_index: Int = w1_index + k
            val wg2_index: Int = w2_index + k
            val kappav: Double = kappa * v
            for (d <- 0 to k - 1) {
              val g1: Double = r2 * weightsArray(w1_index + d) + kappav * weightsArray(w2_index + d)
              val g2: Double = r2 * weightsArray(w2_index + d) + kappav * weightsArray(w1_index + d)
              if (sgd) {
                weightsArray(w1_index + d) -= eta * g1
                weightsArray(w2_index + d) -= eta * g2
              } else {
                val wg1: Double = weightsArray(wg1_index + d) + g1 * g1
                val wg2: Double = weightsArray(wg2_index + d) + g2 * g2
                weightsArray(w1_index + d) -= eta / (math.sqrt(wg1)) * g1
                weightsArray(w2_index + d) -= eta / (math.sqrt(wg2)) * g2
                weightsArray(wg1_index + d) = wg1
                weightsArray(wg2_index + d) = wg2

              }
            }
          }
          ii += 1
        }
      }
      i += 1
    }
    (BDV(weightsArray), tr_loss)
  }
}
