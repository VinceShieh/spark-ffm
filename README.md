# Spark-FFM
A Spark-based implementation of Field-Awared Factorization Machine with parallelled AdaGrad solver.
See http://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf

Need to rework the data format the fit FFM, that is to extends LIBSVM data format by adding field
information to each feature to have formation like:
	label field1:feat1:val1 field2:feat2:val2

## Contact & Feedback

 If you encounter bugs, feel free to submit an issue or pull request.
