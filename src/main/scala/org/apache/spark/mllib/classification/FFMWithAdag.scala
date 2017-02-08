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

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.optimization._
import org.apache.spark.network.protocol.Encoders.Strings

import scala.util.Random
/**
  * Created by vincent on 17-1-4.
  */
class FFMWithAdag(param: FFMParameter, m: Int, n: Int, solver: String) extends Serializable {
  private val k = param.k
  private val sgd = setOptimizer(solver)

  private def generateInitWeights(): Vector = {
    val W = if(sgd){
      new Array[Double](n * m * param.k)
    } else {
      new Array[Double](n * m * param.k *2)
    }
    val coef = 0.5 / Math.sqrt(param.k)
    val random = new Random()
    var position = 0
    if(sgd) {
      for (j <- 0 to n - 1; f <- 0 to m - 1; d <- 0 to param.k - 1) {
        W(position) = coef * random.nextDouble()
        position += 1
      }
    } else {
      for (j <- 0 to n - 1; f <- 0 to m - 1; d <- 0 to 2 * param.k - 1) {
        W(position) = if (d < param.k) coef * random.nextDouble() else 1.0
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
    new FFMModel(n, m, param, values, sgd)
  }

  /**
    * Run the algorithm with the configured parameters on an input RDD
    * of FFMNode entries.
    */
  def run(input: RDD[(Double, Array[FFMNode])]): FFMModel = {
    val gradient = new FFMGradient(m, n, k, sgd)
    val optimizer = new GradientDescentFFM(gradient, null, param)

    val initWeights = generateInitWeights()
    val weights = optimizer.optimize(input, initWeights, param, sgd)
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
    * @param param
    * @return
    */
  def train(data: RDD[(Double, Array[FFMNode])], m: Int, n: Int, param: FFMParameter, solver: String = "sgd"): FFMModel = {
    new FFMWithAdag(param, m, n, solver)
      .run(data)
  }
}
