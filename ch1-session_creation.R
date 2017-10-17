## Session
### In Tensorflow, we use the session function to create a object to control Tensorflow deep learning model. 
### After you build your model, you can use session to start and close your Tensorflow model.
## Demo
### Let's code a demo to show you how to use `tf$Session()`.
### In this case, I create two matrix and use Tensorflow function `tf$matmul()` to multiply them together.
### `run()` your model and `close()` it.

library(tensorflow)
# Matrix 1 and Matrix 2
matrix_1 <- matrix(3, 1,2)
matrix_2 <- matrix(2, 2,1)

product <- tf$matmul(matrix_1, matrix_2)

sess <- tf$Session()
result <- sess$run(product)
print(result) #[12]
sess$close()

