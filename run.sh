$SPARK_HOME/bin/spark-submit \
    --class TestFFMpsgd \
    --master spark://sr443:7077 \
    --driver-memory 180g \
    --total-executor-cores 144 \
    --executor-memory 180g \
    --conf spark.kryoserializer.buffer.max=2047m \
    --conf spark.driver.maxResultSize=30g \
    --conf spark.scheduler.mode="FAIR" \
    target/scala-2.11/Spark-FFM-paralsgd-assembly-0.0.1.jar \
    hdfs://sr443/data/tr_std_ffm \
    4 \
    3 \
    0.1 \
    0.00002 \
    false \
    false \
    144
