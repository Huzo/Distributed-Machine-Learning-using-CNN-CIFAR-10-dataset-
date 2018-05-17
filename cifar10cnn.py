import tensorflow as tf 
import numpy as np 
import os 
import urllib
import tarfile
import argparse
import sys

#Constants
FLAGS = None
OUTPUT_EVERY=200
EVAL_EVERY=500
BATCH_SIZE = 128
GENERATIONS = 20000
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
CROP_HEIGHT = 24
CROP_WIDTH = 24
NUM_CHANNELS = 3
NUM_TARGETS = 10
LEARNING_RATE = 0.1
LR_DECAY = 0.9
NUM_GENS_TO_WAIT = 250.
IMAGE_VECTOR_LENGTH = IMAGE_HEIGHT * IMAGE_WIDTH * NUM_CHANNELS
RECORD_LENGTH = IMAGE_VECTOR_LENGTH + 1
data_dir = 'cifar10data'
extract_folder = 'cifar-10-batches-bin'

global_step = tf.contrib.framework.get_or_create_global_step()




def download_data():
	#Get the CIFAR-10 dataset 
	data_dir = 'cifar10data'
	if not os.path.exists(data_dir):
		os.makedirs(data_dir)
	cifar10_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

	#Check if the file exists, else download it
	data_file = os.path.join(data_dir, 'cifar-10-binary.tar.gz')
	if os.path.isfile(data_file):
		pass
	else:
		#Download the file
		def progress(block_num, block_size, total_size):
			progress_info = [cifar10_url, float(block_num * block_size) / float(total_size) * 100.0]
			print('\r Downloading {} - {:.2f}%'.format(*progress_info), end="")
		filepath,_ = urllib.request.urlretrieve(cifar10_url,data_file,progress)
		#Extract file
		tarfile.open(filepath,'r:gz').extractall(data_dir)

def read_cifar_files(filename):
	reader = tf.FixedLengthRecordReader(record_bytes=RECORD_LENGTH)
	key, record_string = reader.read(filename)
	record_bytes = tf.decode_raw(record_string, tf.uint8)
	image_label = tf.cast(tf.slice(record_bytes, [0], [1]), tf.int32)

	#Extract Image
	image_extracted = tf.reshape(tf.slice(record_bytes,[1],[IMAGE_VECTOR_LENGTH]),
								[NUM_CHANNELS,IMAGE_HEIGHT,IMAGE_WIDTH])

	#Reshape Image
	image_uint8image = tf.transpose(image_extracted, [1, 2, 0])
	reshaped_image = tf.cast(image_uint8image, tf.float32)
	#Randomly Crop Image
	final_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, CROP_WIDTH, CROP_HEIGHT)

	return(final_image, image_label)

def input_pipeline(batch_size, train_logical=True):
	#This function will populate our image pipeline for the batch processor to use. We first need to set
	#up the file list of images we want to read through prebuilt TensorFlow functions. The input produces 
	#can be passed into the reading function that we created in the preceding step, read_cifar_files(). 
	#We will then set a batch reader on the queue, shuffle_batch().
	if train_logical:
		files = [os.path.join(data_dir, extract_folder, 'data_batch_{}.bin'.format(i)) for i in range(1,6)]
	else:
		files = [os.path.join(data_dir, extract_folder, 'test_batch.bin')]
	
	filename_queue = tf.train.string_input_producer(files)
	image, label = read_cifar_files(filename_queue)

	min_after_dequeue = 5000
	capacity = min_after_dequeue + 3 * batch_size
	example_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                        batch_size=batch_size,
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)
	return(example_batch, label_batch)


def create_cnn(input_images,batch_size):
	#Our model function. The model we will use has two convolutional layers, followed by three fully connected layers.

	def truncated_normal_var(name,shape,dtype):
		return(tf.get_variable(name=name,shape=shape,dtype=dtype,initializer=tf.truncated_normal_initializer(stddev=0.05)))

	def bias_var(name,shape,dtype):
		return(tf.get_variable(name=name,shape=shape,dtype=dtype,initializer=tf.constant_initializer(0.1)))

	
	#First convolutional layer
	with tf.variable_scope('conv1') as scope:
		conv1_kernel = truncated_normal_var(name='conv1_kernel',shape=[5,5,3,64],dtype=tf.float32)
		conv1 = tf.nn.conv2d(input_images,conv1_kernel,[1,1,1,1],padding='SAME')
		conv1_bias = bias_var(name='conv1_bias',shape=[64],dtype=tf.float32)
		conv1_add_bias = tf.nn.bias_add(conv1,conv1_bias)
		relu_conv1 = tf.nn.relu(conv1_add_bias)

	
	pool1 = tf.nn.max_pool(relu_conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool1')

	#Second convolutional layer
	with tf.variable_scope('conv2') as scope:
		conv2_kernel = truncated_normal_var('conv2_kernel',[5,5,64,64],tf.float32)
		conv2 = tf.nn.conv2d(pool1,conv2_kernel,[1,1,1,1],padding='SAME')
		conv2_bias = bias_var('conv2_bias',[64],tf.float32)
		conv2_add_bias = tf.nn.bias_add(conv2,conv2_bias)
		relu_conv2 = tf.nn.relu(conv2_add_bias)

	pool2 = tf.nn.max_pool(relu_conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool2')


	reshaped_output = tf.reshape(pool2, [batch_size, -1])
	reshaped_dim = reshaped_output.get_shape()[1].value

	#First fully connected layer
	with tf.variable_scope('full1') as scope:
		full_weight_1 = truncated_normal_var('full_weight_1',[reshaped_dim,384],tf.float32)
		full_bias_1 = bias_var('full_bias_1',[384],tf.float32)
		full_layer1 = tf.nn.relu(tf.add(tf.matmul(reshaped_output,full_weight_1),full_bias_1))

	#Second fully connected layer
	with tf.variable_scope('full2') as scope:
		full_weight_2 = truncated_normal_var('full_weight_2',[384,192],tf.float32)
		full_bias_2 = bias_var('full_bias_2',[192],tf.float32)
		full_layer2 = tf.nn.relu(tf.add(tf.matmul(full_layer1,full_weight_2),full_bias_2))

	#Final fully connected layer, 10 targets
	with tf.variable_scope('full3') as scope:
		full_weight_3 = truncated_normal_var('full_weight_3',[192,NUM_TARGETS],tf.float32)
		full_bias_3 = bias_var('full_bias_3',[NUM_TARGETS],tf.float32)
		final_output = tf.nn.relu(tf.add(tf.matmul(full_layer2,full_weight_3),full_bias_3))

	return(final_output)


def cifar_loss(logits, targets): #Loss function.
	# Get rid of extra dimensions and cast targets into integers
	targets = tf.squeeze(tf.cast(targets, tf.int32))
	# Calculate cross entropy from logits and targets
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
	# Take the average loss across batch size
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	return(cross_entropy_mean)

def train_step(loss_value, generation_num):
	#Our training step. The learning rate will decrease in an exponential step function. 
	model_learning_rate = tf.train.exponential_decay(LEARNING_RATE, generation_num, NUM_GENS_TO_WAIT,LR_DECAY, staircase=True)
	my_optimizer = tf.train.GradientDescentOptimizer(model_learning_rate)
	train_step = my_optimizer.minimize(loss_value,global_step=global_step)
	return(train_step)

def batch_accuracy(logits,targets):
	#We must also have an accuracy function that calculates the accuracy across a batch of images. 
	#We will input the logits and target vectors, and output an averaged accuracy. We can then use 
	#this for both the train and test batches.

	# Make sure targets are integers and drop extra dimensions
	targets = tf.squeeze(tf.cast(targets, tf.int32))
	batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
	predicted_correctly = tf.equal(batch_predictions, targets)
	accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
	return(accuracy)


def main(_):

	download_data()


	#Create worker,ps cluster and embed it into a server. 
	ps_hosts = FLAGS.ps_hosts.split(",")
	worker_hosts = FLAGS.worker_hosts.split(",")

	cluster = tf.train.ClusterSpec({"ps":ps_hosts,"worker":worker_hosts})
	server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)

	if FLAGS.job_name == "ps":
		server.join()
	elif FLAGS.job_name == "worker":

		with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,
													cluster=cluster)):


			
			#split data
			images_train, labels_train = input_pipeline(BATCH_SIZE, train_logical=True)
			images_test, labels_test = input_pipeline(BATCH_SIZE, train_logical=False)

			with tf.variable_scope('model_definition') as scope:
				#Next, we will initialize our loss and test accuracy functions. Then we will declare the generation variable. This 
				#variable needs to be declares as non-trainable, and passed to our training function that uses it in the learning rate 
				#exponential decay calculation
				model_output = create_cnn(images_train,BATCH_SIZE)
				scope.reuse_variables()
				test_output = create_cnn(images_test,BATCH_SIZE)


			loss = cifar_loss(model_output,labels_train)
			accuracy_test = batch_accuracy(test_output,labels_test)
			accuracy_train = batch_accuracy(model_output,labels_train)
			generation_num = tf.Variable(0,trainable=False)
			train_op = train_step(loss,generation_num)

			hooks = [tf.train.StopAtStepHook(last_step=GENERATIONS)]

			#Create monitored session and train model. 
			with tf.train.MonitoredTrainingSession(master=server.target,is_chief=(FLAGS.task_index==0),checkpoint_dir=FLAGS.log_dir,hooks=hooks) as mon_sess:
				tf.train.start_queue_runners(sess=mon_sess)
				i=0
				print('Starting Training')
				train_loss = []
				test_accuracy = []
				while not mon_sess.should_stop():

					_, loss_value = mon_sess.run([train_op, loss])

					if(i+1)%OUTPUT_EVERY==0:
						train_loss.append(loss_value)
						print('global_step %s, task:%d_step %d, training accuracy %g'
						% (tf.train.global_step(mon_sess, global_step), FLAGS.task_index, i, mon_sess.run([accuracy_train])[0]))

					if(i+1)%EVAL_EVERY==0:
						[temp_accuracy] = mon_sess.run([test_accuracy])
						test_accuracy.append(temp_accuracy)
						acc_output = ' --- Test Accuracy = {:.2f}%.'.format(100.*temp_accuracy)
						print(acc_output)
					i = i + 1
				

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.register("type", "bool", lambda v: v.lower() == "true")

	parser.add_argument("--ps_hosts",
						type=str,
						default="",
						help="Comma-seperated list of hostname:port pairs")
	parser.add_argument("--worker_hosts",
						type=str,
						default="",
						help="Comma-seperated list of hostname:port pairs")
	parser.add_argument("--job_name",
						type=str,
						default="",
						help="One of 'ps', 'worker'")
	parser.add_argument("--task_index",
						type=int,
						default=0,
						help="Index of task within the job")
	parser.add_argument("--data_dir",
						type=str,
						default="/tmp/mnist_data",
						help="Directory for storing input data")
	parser.add_argument("--log_dir",
						type=str,
						default="/tmp/train_logs",
						help="Directory of train logs")
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
