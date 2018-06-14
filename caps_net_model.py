'''
Main file to run capsule network on either mnist or notmnist dataset.

Will run distributed, if possible

References: Requirements.txt to run the model

'''


import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils import load_mnist
from capsNet import CapsNet


'''
For distributed models, based on https://github.com/clusterone/self-driving-demo
'''
def device_and_target():
    # If FLAGS.job_name is not set, we're running single-machine TensorFlow.
    # Don't set a device.
    if FLAGS.job_name is None:
        print("Running single-machine training")
        return (None, "")

    # Otherwise we're running distributed TensorFlow
    print("Running distributed training")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")
    if FLAGS.ps_hosts is None or FLAGS.ps_hosts == "":
        raise ValueError("Must specify an explicit `ps_hosts`")
    if FLAGS.worker_hosts is None or FLAGS.worker_hosts == "":
        raise ValueError("Must specify an explicit `worker_hosts`")

    cluster_spec = tf.train.ClusterSpec({
          "ps": FLAGS.ps_hosts.split(","),
          "worker": FLAGS.worker_hosts.split(","),
      })
    server = tf.train.Server(
          cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()

    worker_device = "/job:worker/task:{}".format(FLAGS.task_index)
    # The device setter will automatically place Variables ops on separate
    # parameter servers (ps). The non-Variable ops will be placed on the workers.
    return (
          tf.train.replica_device_setter(
              worker_device=worker_device,
              cluster=cluster_spec),
          server.target,
      )

# Configure distributed task
try:
    job_name = os.environ['JOB_NAME']
    task_index = os.environ['TASK_INDEX']
    ps_hosts = os.environ['PS_HOSTS']
    worker_hosts = os.environ['WORKER_HOSTS']
except:
    job_name = None
    task_index = 0
    ps_hosts = None
    worker_hosts = None

flags = tf.app.flags

# Flags for configuring the distributed task
#used in the function device_and_target
flags.DEFINE_string("job_name", job_name,
                    "job name: worker or ps")
flags.DEFINE_integer("task_index", task_index,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the chief worker task that performs the variable "
                     "initialization and checkpoint handling")
flags.DEFINE_string("ps_hosts", ps_hosts,
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", worker_hosts,
                    "Comma-separated list of hostname:port pairs")

flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_integer("epochs", 50, "Epochs to run")

flags.DEFINE_string("path_to_data", '/data/kporter/multimnist', "Epochs to run")
#flags.DEFINE_string("path_to_data", 'data/', "Location of the data") #to run locally

flags.DEFINE_boolean("use_mnist", False, "When false, uses notmnist dataset")

FLAGS = flags.FLAGS


''''
Set up variables
'''
config = tf.ConfigProto()
    
config.gpu_options.allow_growth = True
train_sum_freq = 5
val_sum_freq = 500

'''
Set up model
'''
#To make it Distributed
device, target = device_and_target() # getting node environment
with tf.device(device): 
    global_step1 = tf.train.get_or_create_global_step()
    model = CapsNet(batch=FLAGS.batch_size, mnist=FLAGS.use_mnist, data_path=FLAGS.path_to_data,global_step=global_step1)
step1 = tf.assign_add(global_step1,1)

'''
Load the data
'''
trX, trY, num_tr_batch, valX, valY, num_val_batch = load_mnist(FLAGS.batch_size, is_training=True, 
                                                               path=FLAGS.path_to_data, mnist=FLAGS.use_mnist)

#Format Y    
Y = valY[:num_val_batch * FLAGS.batch_size].reshape((-1, 1))

merge = tf.summary.merge_all()

'''
Run the Model

Pass in target to determine the worker
'''
with tf.train.MonitoredTrainingSession(master=target, is_chief=(FLAGS.task_index == 0), checkpoint_dir='/logs/train') as sess:
    train_writer = tf.summary.FileWriter('/logs/train', sess.graph)
#     counter = 0 
    for epoch in range(FLAGS.epochs):
        print("Training for epoch %d/%d:" % (epoch, FLAGS.epochs))
        
        for step in tqdm(range(num_tr_batch), total=num_tr_batch, ncols=70, leave=False, unit='b'):
            start = step * FLAGS.batch_size
            end = start + FLAGS.batch_size
            global_step = epoch * num_tr_batch + step
#             counter += 1
            
            if global_step % train_sum_freq == 0:
                _, loss, train_acc, summary_str, _ = sess.run([ model.train_op, model.total_loss, model.accuracy, merge, step1])
                assert not np.isnan(loss), 'Something wrong! loss is nan...'
                train_writer.add_summary(summary_str,global_step)
            else:
                sess.run(model.train_op)
            
            #determine the current validation accuracy
            if val_sum_freq != 0 and (global_step) % val_sum_freq == 0:
                val_acc = 0
                for i in range(num_val_batch):
                    start = i * FLAGS.batch_size
                    end = start + FLAGS.batch_size
                    acc = sess.run(model.accuracy, {model.X: valX[start:end], model.labels: valY[start:end]})
                    val_acc += acc
                val_acc = val_acc / (FLAGS.batch_size * num_val_batch)
                print (str(val_acc))
