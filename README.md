# Spark-FFM
A Spark-based implementation of Field-Awared Factorization Machine. See
http://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf

The data should be formatted as

	label field1:feat1:val1 field2:feat2:val2

to fit FFM, that is to extends LIBSVM data format by adding field information to each feature.

Currently, we support paralleledSGD and paralledAdagrad optimization methods, as they are more
efficient in dealing with large dataset.

Besides, user can also choose to have FFMModel with/without global bias and one-way
interactions.

## Contact & Feedback

 If you encounter bugs, feel free to submit an issue or pull request.
