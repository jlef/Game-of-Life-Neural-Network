source("NeuralNetFunctions.R")
source("gameOfLifeFunctions.R")

############################################################
# Single Layer Neural Net that will progress the game of life for a specific
# cell using its neighbour cells as inputs. 
############################################################

#create possible results and evolve the board
FOR_mset <- expand.grid(c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1),c(0,1));
colnames(FOR_mset) <- c("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8");


FOR_trainset_off <- cbind(FOR_mset[,1:4],0,FOR_mset[,5:8]);
FOR_trainset_on <- cbind(FOR_mset[,1:4],1,FOR_mset[,5:8]);

#setup the off matrices
FOR_result_off <- vector();
for(counter in 1:(dim(FOR_trainset_off)[1])){
  evolve.Board <- matrix(data.matrix(FOR_trainset_off[counter,]), 3, 3)==1;
  #evolve the board
  evolve.Board <- evolve(evolve.Board);
  FOR_result_off <- c(FOR_result_off, evolve.Board[2,2]+0);
}
FOR_trainset_off <- cbind(FOR_result_off, FOR_trainset_off[,c(1:4,6:9)]);

#setup the on matrices
FOR_result_on <- vector();
for(counter in 1:(dim(FOR_trainset_on)[1])){
  evolve.Board <- matrix(data.matrix(FOR_trainset_on[counter,]), 3, 3)==1;
  #evolve the board
  evolve.Board <- evolve(evolve.Board);  
  FOR_result_on <- c(FOR_result_on, evolve.Board[2,2]+0);
}
FOR_trainset_on <- cbind(FOR_result_on, FOR_trainset_on[,c(1:4,6:9)]);

FOR_trainset_off <- cbind(FOR_trainset_off[,1:5],0,FOR_trainset_off[,6:9]);
FOR_trainset_on <- cbind(FOR_trainset_on[,1:5],1,FOR_trainset_on[,6:9]);
colnames(FOR_trainset_off) <- c("y","x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9");
colnames(FOR_trainset_on) <- c("y","x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9");
FOR_trainset <- rbind(FOR_trainset_off, FOR_trainset_on);
FOR_result <- c(FOR_result_off, FOR_result_on);


############################################################
# Train neural networks
############################################################

input_layer_size = 9;
hidden_layer_size = 25;   # 25 hidden units
num_labels = 2;           # 2 labels, from 0 to 1 

lambda <- 2;


#initialise the Theta matrices
initial_Theta1 <- randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 <- randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params <- c(as.vector(initial_Theta1), as.vector(initial_Theta2));

Theta1Data <- matrix(0, 0, 1 + input_layer_size)
Theta2Data <- matrix(0, 0, 1 + hidden_layer_size)

X <- (((data.matrix(FOR_trainset[,c(2:10)]))+1)*2)-3;
y <- FOR_result+1; #1 will be off/0 and 2 will be on/1

res <- optim(initial_nn_params, nnCostFunction, nnGrad, method="L-BFGS-B")
nn_params <- res$par

Theta1Data = matrix(nn_params[1:(hidden_layer_size * (input_layer_size + 1))], ncol=(input_layer_size + 1), nrow=hidden_layer_size);
Theta2Data = matrix(nn_params[(1 + (hidden_layer_size * (input_layer_size + 1))) : length(nn_params)], ncol=(hidden_layer_size + 1), nrow=num_labels);



#########################################################
# Results
#########################################################
require('klaR')

Predicrion <- nnpredict(Theta1Data, Theta2Data, data.matrix(X))
Pred <- (apply(Predicrion, 1, which.max)-1)

errormatrix(na.omit(as.vector(Pred+0)), na.omit(as.vector(FOR_result)), relative = F)
errormatrix(na.omit(as.vector(Pred+0)), na.omit(as.vector(FOR_result)), relative = T)


# Create random board
test <- matrix(0, 10, 10)
set.seed(1984)
test <- apply(test, c(1,2), function(x) sample(c(0,0,1),1))


#Evolve the board 50 times (Game of Life code)
test_final <- test==1;
for(counter in 1:50){
  test_final <- evolve(test_final)
}

#Evolve the board 50 times (Neural Network)
nn_test_final <- test;
for(counter in 1:50){
  nn_test_final <- predictMatrix(nn_test_final);
}

# Compare results
test_final + 0
nn_test_final + 0;

errormatrix(na.omit(as.vector(nn_test_final+0)), na.omit(as.vector(test_final+0)), relative = F)

