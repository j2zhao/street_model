# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py
import numpy
import tensorflow as tf
import tensorflow.contrib.slim as slim
import utility
import scipy.io
import cnn_model
import time

MODEL_DIRECTORY = "model/model.ckpt"
LOGS_DIRECTORY = "logs/train"

# Params for Train
training_epochs = 20 # 10 for augmented training data, 20 for training data
TRAIN_BATCH_SIZE = 50
display_step = 100
validation_step = 500
# Params for test
TEST_BATCH_SIZE = 1000
TRAINING_DATA_SIZE = utility.Size.S_10000 #100, 500, 5000, 10000, Full

def train():
    batch_size = TRAIN_BATCH_SIZE
    num_labels = 10

    path = utility.generate_data_path(TRAINING_DATA_SIZE, utility.Data.STREET, label=False)
    train_images = utility.load_data(path)
    path = utility.generate_data_path(TRAINING_DATA_SIZE, utility.Data.STREET, label=True)
    train_labels = utility.load_data(path, label = True)
    path = utility.generate_data_path(utility.Size.TEST, utility.Data.STREET, label=False)
    test_images = utility.load_data(path)
    path = utility.generate_data_path(utility.Size.TEST, utility.Data.STREET, label=True)
    test_labels = utility.load_data(path, label = True)
    path = utility.generate_data_path(utility.Size.VALIDATION, utility.Data.STREET, label=False)
    validation_images = utility.load_data(path)
    path = utility.generate_data_path(utility.Size.VALIDATION, utility.Data.STREET, label=True)
    validation_labels = utility.load_data(path, label = True)

    train_size = train_images.shape[0]
    total_batch = int(train_size / batch_size)

    # Boolean for MODE of train or test
    is_training = tf.placeholder(tf.bool, name='MODE')

    # tf input
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y_ = tf.placeholder(tf.float32, [None, 10]) #answer
    y = cnn_model.CNN(x)
    
    # Get loss of model
    with tf.name_scope("LOSS"):
        loss = cnn_model.loss(y,y_, )
    # Define optimizer
    with tf.name_scope("ADAM"):
        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            1e-4,  # Base learning rate.
            batch * batch_size,  # Current index into the dataset.
            train_size,  # Decay step.
            0.95,  # Decay rate.
            staircase=True)
        # Use simple momentum for the optimization.
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=batch)

    # Get accuracy of model
    with tf.name_scope("ACC"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Create a summary to monitor learning_rate tensor
    tf.summary.scalar('learning_rate', learning_rate)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar('acc', accuracy)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()
    # Add ops to save and restore all the variables
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())
    max_acc = 0.
    start_time = time.time()
    # Loop for epoch
    for epoch in range(training_epochs):

        # Random shuffling
        [test_images, test_labels] = utility.shuffle_data(test_images, test_labels)
        [train_images, train_labels] = utility.shuffle_data(train_images, train_labels)

        # Loop over all batches
        for i in range(total_batch):

            # Compute the offset of the current minibatch in the data.
            offset = (i * batch_size) % (train_size)
            batch_xs = train_images[offset:(offset + batch_size), :]
            batch_ys = train_labels[offset:(offset + batch_size), :]

            # Run optimization op (backprop), loss op (to get loss value)
            # and summary nodes
            _, train_accuracy, summary = sess.run([train_step, accuracy, merged_summary_op] , feed_dict={x: batch_xs, y_: batch_ys, is_training: True})

            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)

            # Display logs
            if i % display_step == 0:
                print("Epoch:", '%04d,' % (epoch + 1),
                "batch_index %4d/%4d, training accuracy %.5f" % (i, total_batch, train_accuracy))

            # Get accuracy for validation data
            if i % validation_step == 0:
                # Calculate accuracy
                validation_accuracy = sess.run(accuracy,
                feed_dict={x: validation_images, y_: validation_labels, is_training: False})

                print("Epoch:", '%04d,' % (epoch + 1),
                "batch_index %4d/%4d, validation accuracy %.5f" % (i, total_batch, validation_accuracy))

            # Save the current model if the maximum accuracy is updated
            if validation_accuracy > max_acc:
                max_acc = validation_accuracy
                save_path = saver.save(sess, MODEL_DIRECTORY)
                print("Model updated and saved in file: %s" % save_path)

    print("Optimization Finished!")
    print("--- %s seconds ---" % (time.time() - start_time))

    # Restore variables from disk
    saver.restore(sess, MODEL_DIRECTORY)

    # TESTING
    test_size = test_labels.shape[0]
    batch_size = TEST_BATCH_SIZE
    total_batch = int(test_size / batch_size)

    acc_buffer = []

    for i in range(total_batch):
        offset = (i * batch_size) % (test_size)
        batch_xs = test_images[offset:(offset + batch_size), :]
        batch_ys = test_labels[offset:(offset + batch_size), :]
        y_final = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys, is_training: False})
        correct_prediction = numpy.equal(numpy.argmax(y_final, 1), numpy.argmax(batch_ys, 1))
        acc_buffer.append(numpy.sum(correct_prediction) / batch_size)

    print("test accuracy for the stored model: %g" % numpy.mean(acc_buffer))

if __name__ == '__main__':
    train()
