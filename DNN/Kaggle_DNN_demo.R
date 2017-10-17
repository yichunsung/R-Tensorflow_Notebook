library(tensorflow)
library(magrittr)

# Building DNN model
## Building layer structure
add_layer <- function(inputs, input_size, output_size, activation_function = "None"){
  Weights <- tf$Variable(tf$random_normal(shape(input_size, output_size)))
  biases <- tf$Variable(tf$zeros(shape(1, output_size)))
  Wx_plus_b <-tf$matmul(inputs, Weights)+biases
  if(activation_function == "None"){
    outputs <- Wx_plus_b
  }else{
    outputs <- activation_function(Wx_plus_b)
  }
}

# Building placeholder
xs <- tf$placeholder(tf$float32, shape(NULL, 784L))
ys <- tf$placeholder(tf$float32, shape(NULL, 10L))

# Building Nerual Network Structure
output_layer <- add_layer(xs, 784, 10, activation_function = tf$nn$softmax)

# Building loss error method (cross entropy)
cross_entropy <- tf$reduce_mean(-tf$reduce_sum(ys*tf$log(output_layer), 
                                               reduction_indices = 1L))

# Buliding Gradient Descent and learning rate
learning_rate <- 0.1 # Set learning rate = 0.1
train_step_by_GD <- tf$train$GradientDescentOptimizer(learning_rate)$minimize(cross_entropy)

# data mungung
Mnist_train <- read.csv("C:/Machine-Learing-Notebook/data/Digit_Recognizer/train.csv")
Mnist_label <- Mnist_train[,1]
Mnist_train_image <- Mnist_train[,-1]/256
Matrix_Mnist_train_image <- as.matrix(Mnist_train_image)

MnistLabel <- data.frame(Mnist_label)
for(i in c(0:9)){
  newCol <- ifelse(MnistLabel$Mnist_label == i,
                   1,
                   0)
  MnistLabel <- cbind(MnistLabel, newCol)
}
names(MnistLabel)[2:11] <- c(0:9)
Matrix_MNIST_label <- as.matrix(MnistLabel[,-1])

# Session setting
initiz <- tf$global_variables_initializer()
sess <- tf$Session()
sess$run(initiz)

# Training
for (i in 1:1000){
  #batches <- mnist$train$next_batch(100L)
  #batch_xs <- batches[[1]]
  #batch_ys <- batches[[2]]
  sess$run(train_step_by_GD, feed_dict = dict(xs = Matrix_Mnist_train_image, ys = Matrix_MNIST_label))
  if(i %% 50 == 0){
    sess$run(cross_entropy, feed_dict = dict(xs = Matrix_Mnist_train_image, ys = Matrix_MNIST_label)) %>% print()
  }
}
# Buliding Accuracy function
# def compute_accuracy(v_xs, v_ys):
#   global output_Layer
# y_pre = sess.run(output_Layer, feed_dict={xs: v_xs, keep_prob: 1})
# correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob:1})
# return result

# Test trained model
correct_prediction <- tf$equal(tf$argmax(y, 1L), tf$argmax(ys, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
sess$run(accuracy,
         feed_dict = dict(xs = Matrix_Mnist_train_image, ys = Matrix_MNIST_label))
