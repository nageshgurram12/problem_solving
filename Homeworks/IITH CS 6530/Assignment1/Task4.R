par(mfrow=c(1,2))
# Read User-TV show matrix from data set 
userItemMatrix <- data.matrix(read.csv("dataset/user-shows.txt", sep=" ", header = FALSE))

TVShows <- read.csv("dataset/shows.txt", header = FALSE, stringsAsFactors = FALSE)$V1

top5 <- list()

# -----------User-User Filtering----------

# Actual preference of User20 to the first 100 users
user20Actual <- userItemMatrix[20,1:100]
actual <- table(user20Actual)[2]

#Erase first 100 movies watched by user 20
userItemMatrix[20,1:100] <- rep(0,100)

user20 <- userItemMatrix[20,] #User(20) vector
l2user20 <- 1/sqrt(sum(user20)) # inverse L2 norm of User(20)
l2allUsers <- diag(1/sqrt(rowSums(userItemMatrix))) #inverse L2 norm of all users

#Get similarity between U(20) and other users by taking cosine similarity
user20ToAllUsersSimilarity <- matrix(l2user20,1,1) %*% user20 %*% t(userItemMatrix) %*% l2allUsers

#User20 item preferences based on his similarity with other users and their preference
user20ItemPrefAsPerUU <- user20ToAllUsersSimilarity %*% userItemMatrix

user20ItemPref100UU <- round(user20ItemPrefAsPerUU, 3)[1:100]
user20ItemPref100UUSorted <- sort(user20ItemPref100UU)

truePositiveRate <- rep(0,19)

for(k in 1:19){
  user20TopKUU <- which(user20ItemPref100UU %in% tail(user20ItemPref100UUSorted, k))
  # Save Top 5 shows 
  watched <- table(user20Actual[user20TopKUU])["1"]
  # Save Top 5 shows 
  if(k == 5){
    top5$top5UUNames <- TVShows[user20TopKUU]
    top5$top5UUSimScore <-user20ItemPref100UU[user20TopKUU]
  }
  truePositiveRate[k] <- watched/actual
}
plot(1:19, truePositiveRate, main = "User-User prediction", pch=19, xlab = "top-k predictions", ylab="true positive rate")


# ----------- ITEM-ITEM Filtering---------
l2allItems <- diag(1/sqrt(colSums(userItemMatrix))) #inverse L2 norm for all items

#item-item similarity
itemItemSimilarity <- l2allItems %*% t(userItemMatrix) %*% userItemMatrix %*% l2allItems

#User20 item preferences based on similarity bewteen what he watched actually and their similarity with other items
user20ItemPrefAsPerII <- user20 %*% itemItemSimilarity
user20ItemPref100II <- round(user20ItemPrefAsPerII, 3)[1:100]
user20ItemPref100IISorted <- sort(user20ItemPref100II)
which(user20ItemPref100II %in% tail(user20ItemPref100IISorted, 5))
for(k in 1:19){
  user20TopKII <- which(user20ItemPref100II %in% tail(user20ItemPref100IISorted, k))
  watched <- table(user20Actual[user20TopKII])["1"]
  # Save Top 5 shows 
  if(k == 5){
    top5$top5IINames <- TVShows[user20TopKII]
    top5$top5IISimScore <- user20ItemPref100II[user20TopKII]
  }
  truePositiveRate[k] <- watched/actual
}
plot(1:19, truePositiveRate, main = "Item-Item prediction", pch=19, xlab = "top-k predictions", ylab="true positive rate")

write.table(as.data.frame(top5), file="Top5Shows.csv", quote = F, sep=",", row.names = F)