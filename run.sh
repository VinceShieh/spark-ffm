$SPARK_HOME/bin/spark-submit \
    --class TestFFM \
    --master local[*] \
    --driver-memory 4g \
    target/scala-2.11/spark-ffm_2.11-0.0.1.jar \
    0.1 \
    0.00002 \
    100 \
    2 \
    true \
    false \
    data/testdata.txt \
    va_std
