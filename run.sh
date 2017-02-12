$SPARK_HOME/bin/spark-submit \
    --class TestFFM \
    --master local[*] \
    --driver-memory 4g \
    target/scala-2.11/spark-ffm_2.11-0.0.1.jar \
    data/a9a_ffm \
    8 \
    3 \
    0.1 \
    0.00002 \
    false \
    false
