#!/bin/sh
#spark-submit matrix_factorization_from_pickle_spark.py 10 rdd2.pickle >>log.txt
#spark-submit matrix_factorization_from_pickle_spark.py 10 rdd2.pickle rdd3.pickle >>log.txt
#spark-submit matrix_factorization_from_pickle_spark.py 10 rdd2.pickle rdd3.pickle rdd4.pickle >>log.txt
spark-submit matrix_factorization_from_pickle_spark.py 10 rdd2.pickle rdd3.pickle rdd4.pickle rdd5.pickle >>log.txt

spark-submit matrix_factorization_from_pickle_spark.py 50 rdd2.pickle >>log.txt
spark-submit matrix_factorization_from_pickle_spark.py 100 rdd2.pickle >>log.txt
spark-submit matrix_factorization_from_pickle_spark.py 1000 rdd2.pickle >>log.txt

spark-submit matrix_factorization_from_pickle_spark.py 50 rdd2.pickle rdd3.pickle >>log.txt
spark-submit matrix_factorization_from_pickle_spark.py 50 rdd2.pickle rdd3.pickle rdd4.pickle >>log.txt
spark-submit matrix_factorization_from_pickle_spark.py 50 rdd2.pickle rdd3.pickle rdd4.pickle rdd5.pickle >>log.txt

spark-submit matrix_factorization_from_pickle_spark.py 100 rdd2.pickle rdd3.pickle >>log.txt
spark-submit matrix_factorization_from_pickle_spark.py 100 rdd2.pickle rdd3.pickle rdd4.pickle >>log.txt
spark-submit matrix_factorization_from_pickle_spark.py 100 rdd2.pickle rdd3.pickle rdd4.pickle rdd5.pickle >>log.txt

spark-submit matrix_factorization_from_pickle_spark.py 1000 rdd2.pickle rdd3.pickle >>log.txt
spark-submit matrix_factorization_from_pickle_spark.py 1000 rdd2.pickle rdd3.pickle rdd4.pickle >>log.txt
spark-submit matrix_factorization_from_pickle_spark.py 1000 rdd2.pickle rdd3.pickle rdd4.pickle rdd5.pickle >>log.txt
