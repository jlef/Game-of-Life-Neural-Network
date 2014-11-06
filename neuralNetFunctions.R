############################################################
# Neural Net Functions
############################################################

predictMatrix <- function(M){
  M <- (((data.matrix(M))+1)*2)-3;
  P <- matrix(-1, 12, 12);
  P[2:11, 2:11] <- M;
  
  R <- matrix(0, 0, 9);
  for(i in 2:11){
    for(j in 2:11){
      R <- rbind(R, c(P[j-1,i-1], P[j-1,i], P[j-1,i+1], P[j,i-1], P[j,i], P[j,i+1], P[j+1,i-1], P[j+1,i], P[j+1,i+1]));
    }
  }
  
  Predicrion <- nnpredict(Theta1Data, Theta2Data, data.matrix(R));
  Pred <- (apply(Predicrion, 1, which.max)-1);
  
  matrix(Pred, 10, 10);
}

randInitializeWeights <- function (L_in, L_out){
  #initialize the weights of a layer with L_in incoming connections and L_out outgoing connections
  W <- matrix(0, L_out, 1 + L_in);
  
  epsilon_init <- 0.12;
  randWeights <- matrix(runif(L_out * (1 + L_in), min=0, max=1), L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
  randWeights
}

sigmoid <- function (z){
  g = 1.0 / (1.0 + exp(-z));
}

sigmoidGradient <- function (z){  
  g <- matrix(0, dim(z)[1], dim(z)[2]);
  g <- sigmoid(z) * (1 - sigmoid(z));
}

nnGrad <- function (nn_params){
  Theta1 = matrix(nn_params[1:(hidden_layer_size * (input_layer_size + 1))], ncol=(input_layer_size + 1), nrow=hidden_layer_size);
  Theta2 = matrix(nn_params[(1 + (hidden_layer_size * (input_layer_size + 1))) : length(nn_params)], ncol=(hidden_layer_size + 1), nrow=num_labels);
  m <- dim(X)[1];
  
  # Add ones to the X data matrix
  X1 <- cbind(rep(1, m), X);
  yv <- diag(1,num_labels)[y,];
  
  Theta1_0 <- Theta1;
  Theta1_0[,1] <- 0;
  
  Theta2_0 <- Theta2;
  Theta2_0[,1] <- 0;
  
  a1 <- X1;
  z2 <- Theta1%*%t(a1);
  a2 <- cbind(rep(1, dim(t(z2))[1]), sigmoid(t(z2)));
  z3 <- Theta2%*%t(a2);
  a3 <- sigmoid(z3);
  
  delta3 = t(a3) - yv;
  newTheta <- Theta2_0;
  newTheta <- newTheta[,-1];
  delta2 <- t(delta3 %*% newTheta)*sigmoidGradient(z2);
  Theta2_grad <- (1/m) * t(t(a2)%*%delta3)+(lambda/m) * Theta2_0;
  Theta1_grad <- (1/m) * (delta2 %*% a1)+(lambda/m) * Theta1_0;
  
  grad <- c(as.vector(Theta1_grad), as.vector(Theta2_grad)); 
  grad
  
}

nnCostFunction <- function (nn_params) {
  Theta1 = matrix(nn_params[1:(hidden_layer_size * (input_layer_size + 1))], ncol=(input_layer_size + 1), nrow=hidden_layer_size);
  Theta2 = matrix(nn_params[(1 + (hidden_layer_size * (input_layer_size + 1))) : length(nn_params)], ncol=(hidden_layer_size + 1), nrow=num_labels);
  
  # Initialize variables
  m <- dim(X)[1];
  if(is.null(m)){
    m1 <- 1;
  }
  J <- 0;
  Theta1_grad <- matrix(0, dim(Theta1)[1], dim(Theta1)[2]);
  Theta2_grad <- matrix(0, dim(Theta2)[1], dim(Theta2)[2]);
  
  # Add ones to the X data matrix
  X1 <- cbind(rep(1, m), X);
  yv <- diag(1,num_labels)[y,];
  
  Theta1_0 <- Theta1;
  Theta1_0[,1] <- 0;
  
  Theta2_0 <- Theta2;
  Theta2_0[,1] <- 0;
  if(dim(Theta1)[2] != dim(X1)[2]){
    testCode <- 1;
  }
  ThetaTransX <- Theta1 %*% t(X1);
  z <- (cbind(rep(1, m), t(sigmoid(ThetaTransX))) %*% t(Theta2));
  hofx <- sigmoid(z);
  #Cost function
  J <- sum(sum(-(yv)*log(hofx)-((1-yv)*log(1-hofx))))*(1/m)+(lambda / (2 * m)) * 
    (sum(sum(Theta1_0*Theta1_0))+sum(sum(Theta2_0*Theta2_0)));
  J
  
}

nnpredict <- function (pTheta1, pTheta2, pX){
  m <- dim(pX)[1];
  num_labels <- dim(pTheta2)[1];
  p = matrix(0, dim(X)[1], dim(pX)[2]);
  # Add ones to the X data matrix
  X1 <- cbind(rep(1, m), pX);
  h1 <- sigmoid(X1 %*% t(pTheta1));
  h11 <- cbind(rep(1, m), h1);
  h2 <- sigmoid(h11 %*% t(pTheta2));
}