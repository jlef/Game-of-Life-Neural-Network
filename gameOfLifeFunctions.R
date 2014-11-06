############################################################
# Game of Life functions (http://rosettacode.org/wiki/Conway's_Game_of_Life)
############################################################

# Generates the next interation of the board from the existing one
evolve <- function(board)
{ 
  newboard <- board
  for(i in seq_len(nrow(board)))
  {
    for(j in seq_len(ncol(board)))
    {
      newboard[i,j] <- determine.new.state(board,i,j)         
    }   
  }
  newboard
}

# Returns the number of living neighbours to a location
count.neighbours <- function(x,i,j) 
{   
  sum(x[max(1,i-1):min(nrow(x),i+1),max(1,j-1):min(ncol(x),j+1)]) - x[i,j]
}

# Implements the rulebase
determine.new.state <- function(board, i, j)
{
  N <- count.neighbours(board,i,j)
  (N == 3 || (N ==2 && board[i,j]))
}