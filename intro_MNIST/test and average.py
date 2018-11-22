# Test 2000 times
import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息  
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error   
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
sum=0
last=0
for num in range(0,20):
  x = tf.placeholder("float", [None, 784])
  W = tf.Variable(tf.zeros([784,10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.nn.softmax(tf.matmul(x,W) + b)
  y_ = tf.placeholder("float", [None,10])
  cross_entropy = -tf.reduce_sum(y_*tf.log(y))
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init)
  for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  sum+=sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
  print (sum-last)
  last=sum
print("Average is:")
print(sum/20)