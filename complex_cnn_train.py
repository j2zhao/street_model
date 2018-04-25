# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py
import numpy
import tensorflow as tf
import tensorflow.contrib.slim as slim
import utility
import scipy.io
import complex_cnn_model_3 as cnn_model
import time

MODEL_DIRECTORY = "model_3/model.ckpt"
LOGS_DIRECTORY = "logs_3/train"

# Params for Train
training_epochs = 20 # 10 for augmented training data, 20 for training data
TRAIN_BATCH_SIZE = 50
display_step = 100
validation_step = 500
# Params for test
TEST_BATCH_SIZE = 1000

TRAINING_DATA_SIZE = utility.Size.S_100 #100, 500, 5000, 10000
def train():
    batch_size = TRAIN_BATCH_SIZE
    num_labels = 10
    
    train_images = {}
    train_labels = {}
    test_images = {}
    test_labels = {}
    validation_images = {}
    validation_labels = {}
    for data in utility.Data:
        path = utility.generate_data_path(TRAINING_DATA_SIZE, data, label=False)
        train_images[data] = utility.load_data(path)
        path = utility.generate_data_path(TRAINING_DATA_SIZE, data, label=True)
        train_labels[data] = utility.load_data(path, label = True)
        path = utility.generate_data_path(utility.Size.TEST, data, label=False)
        test_images[data] = utility.load_data(path)
        path = utility.generate_data_path(utility.Size.TEST, data, label=True)
        test_labels[data] = utility.load_data(path, label = True)
        path = utility.generate_data_path(utility.Size.VALIDATION, data, label=False)
        validation_images[data] = utility.load_data(path)
        path = utility.generate_data_path(utility.Size.VALIDATION, data, label=True)
        validation_labels[data] = utility.load_data(path, label = True)

    train_size = train_images[utility.Data.CUSTOM].shape[0] + train_images[utility.Data.STREET].shape[0] + train_images[utility.Data.MNIST].shape[0]
    total_batch = int(train_size / batch_size)

    # Boolean for MODE of train or test
    #is_training = tf.placeholder(tf.bool, name='MODE')
    data_type = tf.placeholder(tf.string)
    # tf input
    x = tf.placeholder(tf.float32)
    y_ = tf.placeholder(tf.float32, [None, 10]) #answer
    y = cnn_model.CNN(x, data_type)
    
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
    sess.run(tf.global_variables_initializer(), feed_dict={})
    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())
    max_acc = 0.

    start_time = time.time()
    # Loop for epoch
    for epoch in range(training_epochs):
        type_order = utility.generate_random_order(train_size/(3*TRAIN_BATCH_SIZE))
        # Random shuffling
        for data in utility.Data:
            [validation_images[data], validation_labels[data]] = utility.shuffle_data(validation_images[data], validation_labels[data])
            [train_images[data], train_labels[data]] = utility.shuffle_data(train_images[data], train_labels[data])
            
        count = {utility.Data.CUSTOM: 0, utility.Data.MNIST: 0, utility.Data.STREET: 0}
        # Loop over all batches
        for i in range(total_batch):
            set_type = type_order[i]
            #print("HELLLLLLLLO")
            
            #print("GOOODBYYYYYYE")
            offset = (count[set_type] * batch_size) % (train_size)
            count[set_type] = count[set_type] + 1
            # Compute the offset of the current minibatch in the data.
            #offset = (i * batch_size) % (train_size)
            batch_xs = train_images[set_type][offset:(offset + batch_size), :]
            batch_ys = train_labels[set_type][offset:(offset + batch_size), :]

            assert set_type.value in [utility.Data.CUSTOM.value, utility.Data.MNIST.value, utility.Data.STREET.value]
            # Run optimization op (backprop), loss op (to get loss value)
            # and summary nodes
            _, train_accuracy, summary = sess.run([train_step, accuracy, merged_summary_op] , 
                feed_dict={x: batch_xs, y_: batch_ys, data_type: set_type.value})

            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)

            # Display logs
            if i % display_step == 0:
                print("Epoch:", '%04d,' % (epoch + 1),
                "batch_index %4d/%4d, training accuracy %.5f" % (i, total_batch, train_accuracy))

            # Get accuracy for validation data
            # need to average 3 validation data entries
            if i % validation_step == 0:
                # Calculate accuracy
                validation_accuracy_a = sess.run(accuracy,
                    feed_dict={x: validation_images[utility.Data.CUSTOM], y_: validation_labels[utility.Data.CUSTOM], data_type:utility.Data.CUSTOM.value })

                #print("HIIII")
                #print(validation_images[utility.Data.STREET].size)
                validation_accuracy_b = sess.run(accuracy,
                    feed_dict={x: validation_images[utility.Data.STREET], y_: validation_labels[utility.Data.STREET],  data_type:utility.Data.STREET.value})

                validation_accuracy_c = sess.run(accuracy,
                    feed_dict={x: validation_images[utility.Data.MNIST], y_: validation_labels[utility.Data.MNIST],  data_type:utility.Data.MNIST.value})
                
                validation_accuracy = (validation_accuracy_a + validation_accuracy_b + validation_accuracy_c)/3
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
    # TODO: edit testing
    for data in utility.Data:
        labels = test_labels[data]
        images = test_images[data]
        test_size = labels.shape[0]
        batch_size = TEST_BATCH_SIZE
        total_batch = int(test_size / batch_size)

        acc_buffer = []
        for i in range(total_batch):
            offset = (i * batch_size) % (test_size)
            batch_xs = images[offset:(offset + batch_size), :]
            batch_ys = labels[offset:(offset + batch_size), :]
            y_final = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys, data_type: data.value })
            correct_prediction = numpy.equal(numpy.argmax(y_final, 1), numpy.argmax(batch_ys, 1))
            acc_buffer.append(numpy.sum(correct_prediction) / batch_size)

        print("test accuracy for the stored model for %s images: %g" % (data, numpy.mean(acc_buffer)))

if __name__ == '__main__':
    train()
