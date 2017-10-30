# Convoluation neural network demo
############################
#      Input               #
#       ↓                  #
# Convolution layer 1      #    
#       ↓                  #
# Max pooling layer 1      #
#       ↓                  #
# Convolution layer 2      #
#       ↓                  #
# Max pooling layer 2      #
#       ↓                  #
# Flatten layer            #
#       ↓                  #
# Fully connected layer 1  #
#       ↓                  #
# Fully connected layer 2  #
#       ↓                  #
#     Output               #
############################
# Library Tensorflow and other packages
library(tensorflow)
library(magrittr)

# Define function for filter
# [filter_height, filter_width, in_channels, out_channels]
add_conv_filter <- function(filterShape){ # random filter as Variable
  filterForConvLayer <- tf$truncated_normal(filterShape, stddev = 0.1) %>% tf$Variable() 
  return(filterForConvLayer)
}
# Define function for bias
add_bias <- function(BiasShape){
  bias <- tf$constant(0.1, shape = BiasShape) %>% tf$Variable()
  return(bias)
}
# Define function for convolution layer
add_convolutionLayer <- function(inputData, filter_weight, activation_function = "None"){
  conv2dLayer <- tf$nn$conv2d(input = inputData, # input data
                              # filter should be shape(filter_height, filter_width, in_channels, out_channels)
                              filter = filter_weight, 
                              strides = shape(1L, 2L, 2L, 1L), # strides = [1, x_strides, y_strides, 1]
                              padding = 'SAME'
                              )
  if(activation_function == "None"){
    output_result <- conv2dLayer
  }else{
    output_result <- activation_function(conv2dLayer)
  }
  return(output_result)
}
# Define function for Max pooling layer
add_maxpoolingLayer <- function(inputData){
  MaxPooling <- tf$nn$max_pool(inputData, # input data should be [batch, height, width, channels]
                               ksize = shape(1L, 2L, 2L, 1L), # 2*2 pixels for max pooling
                               strides = shape(1L, 2L, 2L, 1L), # strides =  [1, x_strides, y_strides, 1]
                               padding = 'SAME')
  return(MaxPooling)
}
# Define function for flatten layer
add_flattenLayer <- function(inputData, numberOfFactors){
  flatten_layer <- tf$reshape(inputData, shape(-1, numberOfFactors))
  return(flatten_layer)
}
# Define function for fully connected layer
add_fullyConnectedLayer <- function(inputData, Weight_FCLayer, bias_FCLayer, activation_function = "None"){
  Wx_plus_b <- tf$matmul(inputData, Weight_FCLayer)+bias_FCLayer
  if(activation_function == "None"){
    FC_output_result <- Wx_plus_b
  }else{
    FC_output_result <- activation_function(Wx_plus_b)
  }
  return(FC_output_result)
}

# Define compute_accuracy function
compute_accuracy <- function(model_result, v_xs, v_ys){
  y_pre <- sess$run(model_result, feed_dict = dict(xs = v_xs))
  correct_prediction <- tf$equal(tf$argmax(y_pre, 1L), tf$argmax(v_ys, 1L))
  accuracy <- tf$cast(correct_prediction, tf$float32) %>% tf$reduce_mean(.)
  result <- sess$run(accuracy, feed_dict = dict(xs = v_xs, ys = v_ys))
  return(result)
}

# Model building =======
## Setting placeholder
xs <- tf$placeholder(tf$float32, shape(NULL, 784L)) # input data = 28*28(784 factors) pixels image.
ys <- tf$placeholder(tf$float32, shape(NULL, 10L)) # output = 10 labels (0~9)
x_image <- tf$reshape(xs, shape(-1L, 28L, 28L, 1L)) # [batch, height, width, channels]
## Convolution layer 1 
convolayer1 <- add_convolutionLayer(
  inputData = x_image,
  filter_weight = shape(5L, 5L, 1L, 32L) %>% add_conv_filter(),
  activation_function = tf$nn$relu
)
## Max pooling layer 1
maxPooling_1 <- add_maxpoolingLayer(
  convolayer1
)
## Convolution layer 2 
convolayer2 <- add_convolutionLayer(
  inputData = maxPooling_1, 
  filter_weight = shape(4L, 4L, 32L, 64L) %>% add_conv_filter(),
  activation_function = tf$nn$relu
) 
## Max pooling layer 2 
maxPooling_2 <- add_maxpoolingLayer(
  inputData = convolayer2
)
## Flatten layer
flatLayer_output <- add_flattenLayer(
  inputData = maxPooling_2,
  numberOfFactors = c(2L*2L*64L) %>% as.numeric()
)
## Fully connected layer 1
fcLayer_1 <- add_fullyConnectedLayer(
  inputData = flatLayer_output,
  Weight_FCLayer = shape(2L*2L*64L, 1024L) %>% tf$random_normal(., stddev = 0.1) %>% tf$Variable(), # Set first layer ouput = 1024
  bias_FCLayer = shape(1024L) %>% add_bias(),
  activation_function = tf$nn$relu
)
## Fully connected layer 2
output_result <- add_fullyConnectedLayer(
  inputData = fcLayer_1,
  Weight_FCLayer = shape(1024L, 10L) %>% tf$random_normal(., stddev = 0.1) %>% tf$Variable(), # Set output layer ouput = 10 labels
  bias_FCLayer = shape(10L) %>% add_bias(),
  activation_function = tf$nn$softmax
)

# Model training =======
## Loss function (cross entropy)
cross_entropy <- tf$reduce_mean(-tf$reduce_sum(ys*tf$log(output_result), 
                                               reduction_indices = 1L))
## Gradient Descent and learning rate setting
learning_rate <- 0.001 # Set learning rate = 0.1
train_step_by_GD <- tf$train$AdamOptimizer(learning_rate)$minimize(cross_entropy)

## Session setting
sess <- tf$Session()
init <- tf$global_variables_initializer()
sess$run(init)

# MNIST data loadding 
datasets <- tf$contrib$learn$datasets
mnist <- datasets$mnist$read_data_sets("MNIST-data", one_hot = TRUE)
# Running
for (i in 1:1000){
  batches <- mnist$train$next_batch(100L)
  batch_xs <- batches[[1]]
  batch_ys <- batches[[2]]
  sess$run(train_step_by_GD, feed_dict = dict(xs = batch_xs, ys = batch_ys))
  if(i %% 50 == 0){
    print(compute_accuracy(output_result, mnist$test$images, mnist$test$labels))
   # sess$run(cross_entropy, feed_dict = dict(xs = batch_xs, ys = batch_ys)) %>% print()
  }
}
