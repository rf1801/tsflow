import tensorflow as tf
import os



t_zero_d = tf.constant(1)
#print(t_zero_d)

t_one_d = tf.constant([[1],[2]])
#print(t_one_d)

t_two_d = tf.constant([[3],[4]])

print(tf.math.multiply(t_one_d,t_two_d))
#print(t_two_d)


idmat=tf.eye(num_rows=2 , num_columns=2 , batch_shape=[3,4])

print("-----------------------")
g = tf.random.Generator.from_seed(11).normal(shape=(2,2))

