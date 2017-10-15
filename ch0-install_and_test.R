## Installing Tensorflow for R
### 1. Quick install for CPU version
install.packages("tensorflow")
library(tensorflow)
install_tensorflow() # Creating r-tensorflow conda environment for TensorFlow installation

### 2. GPU version
#### If you want to install the GPU version, you can check tensorflow.rstudio.com .
#### [Installing the GPU Version](https://tensorflow.rstudio.com)

## Testing
#### After you successfully install your tensorflow package, you can type this code to test your environment.
#### If you can build a perfect result, congratulation, you can enjoy your tensorflow trip for next step.
library(tensorflow)
sess = tf$Session()
hello <- tf$constant('Hello, TensorFlow!')
sess$run(hello) 