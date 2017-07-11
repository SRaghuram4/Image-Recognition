## Digit Recognition 

## Load the required packages.
# I have learnt this recently and trying to apply this by creating a function

#create a function to check for installed packages and install them if they are not installed
install <- function(packages){
  new.packages <- packages[!(packages %in% installed.packages()[, "Package"])]
  if (length(new.packages)) 
    install.packages(new.packages, dependencies = TRUE)
  sapply(packages, require, character.only = TRUE)
}

# usage
required.packages <- c("ggplot2", "dplyr", "reshape2", "devtools", "shiny", "shinydashboard",
                       "caret","randomForest","gbm","tm","forecast","knitr","Rcpp","stringr",
                       "lubridate","manipulate","Scale","sqldf","RMongo","foreign","googleVis",
                       "XML","roxygen2","plotly","parallel","car","gridExtra")
install(required.packages)

#Read the data into the software

train <- read.csv("train.csv")
test <- read.csv("test.csv")

## Data Exploration 

dim(train)

str(train$label)

# We can also look at the levels
levels(train$label)
train$label <- as.factor(train$label)

# Look at the digits one by one by creatig a function which can display them.

show.digit <- function(i){
  image(matrix(as.numeric(train[i,-1]), nrow = 28),col = grey.colors(250))
}
show.digit(12)
show.digit(43)
show.digit(456)
show.digit(670)
# And so on

## Feature Engineering

# Intensity is one of the feature which may have effect on prediction capabilities.

train$intensity <- apply(train[,-1], 1, mean) #takes the mean of each row in train

intbylabel <- aggregate (train$intensity, by = list(train$label), FUN = mean)

plot <- ggplot(data=intbylabel, aes(x=Group.1, y = x)) +
  geom_bar(stat="identity")
plot + scale_x_discrete(limits=0:9) + xlab("digit label") + 
  ylab("average intensity")

# As we can see there are some differences in intensity. The digit "1" is the less intense while the
#digit "0" is the most intense. So this new feature seems to have some predictive value if you wanted 
#to know if say your digit is a "1" or no. But the problem of course is that different peple write
#their digits differently. We can get a sense of this by plotting the distribution of the average 
#intensity by label

ggplot(data = train, aes(x=intensity,fill = factor(label))) + geom_density(alpha = 0.4)

#What can we observe from the histograms above? Well most intensity distributions seem roughly
#normally distributed but some have higher variance than others. The digit "1" seems to be the 
#one people write most consistently across the board. Other than that the intensity 
#feature isn't all that helpful

# Again, we may have people who write 4 and 7 differently as far as other digits are concerned. Lets look at them.

p1 <- qplot(subset(train, label ==1)$intensity, binwidth = .75, 
            xlab = "Intensity Histogram for 1")

p2 <- qplot(subset(train, label ==4)$intensity, binwidth = .75,
            xlab = "Intensity Histogram for 4")

p3 <- qplot(subset(train, label ==7)$intensity, binwidth = .75,
            xlab = "Intensity Histogram for 7")

p4 <- qplot(subset(train, label ==9)$intensity, binwidth = .75,
            xlab = "Intensity Histogram for 9")

grid.arrange(p1, p2, p3,p4, ncol = 2)

# As we can see, that number 4 and 7 seem bi-modular and rest of the graphs are normalyy distributed. 
# So Lets dig more into how differently people write 4s and 7s.

train4 <- train[train$label == 4, ]
train7 <- train[train$label == 7, ]

flip <- function(matrix){
  apply(matrix, 2, rev)
}

par(mfrow=c(3,3))
for (i in 16:24){
  digit <- flip(matrix(rev(as.numeric(train4[i,-c(1, 786)])), nrow = 28)) #look at one digit
  image(digit, col = grey.colors(255))
}

image(flip(matrix(rev(as.numeric(train4[22,-c(1,786)])),nrow = 28)))

# There are indeed two different types of how people write 4s.Similarly, 7s.

par(mfrow=c(3,3))
for (i in 16:24){
  digit <- flip(matrix(rev(as.numeric(train7[i,-c(1, 786)])), nrow = 28)) #look at one digit
  image(digit, col = grey.colors(255))
}

## Symmetry 

# We know that "1","3","8 and "0" are the digits which are symmetric along the horizontal axis. So, to know
# how many people really write these digits symmetrically and which one's the most symmetric.

par(mfrow = c(1,1))
pixels <- train[,-c(1, 786)]/255

symmetry <-  function(vect) {
  matrix <- flip(matrix(rev(unlist(vect))))
  flipped <- flip(matrix)
  diff <- flipped - matrix
  return(sum(diff*diff))
}

symmetry((pixels[1,]))
sym <- (apply(X = pixels, MARGIN = 1, FUN = symmetry))
means <- numeric(10)
for (i in 0:9){
  means[i+1] <- mean(sym[train$label == i])
}

means <- (means/intbylabel[,2])**(-1)

mean <- data.frame( label = 0:9,symmetry = means)

plot <- ggplot(data=mean, aes(x= label, y = symmetry)) +
  geom_bar(stat="identity")
plot + scale_x_discrete(limits=0:9) + xlab("digit label") + 
  ylab("symmetry")
# So from the plot, it is inferred that most symmetric digit is "1" and the least one is "7".

## Predictive models 

# A simple random forest 
# Knowing that the ensemble of the decision trees, i.e. the random forests are the most effective models,
# lets execute a random forest predictive model.

set.seed(0)

numTrain <- 30000
numTrees <- 100

rows <- sample(1:nrow(train), numTrain)
labels <- as.factor(train[rows,1])
train <- train[rows,-1]

rf <- randomForest(train, labels, xtest=test, ntree=numTrees)
rf

# First conclusion is that with 30000 training samples and number of trees = 100, we get
# Out of Bag error as 0.0598, and thereby, the accuracy being 100-5.98 = 94.02 %. 

predictions <- data.frame(ImageId=1:nrow(test), Label=levels(labels)[rf$test$predicted])

plot(rf)

# When output of the random forest is plotted, we see that the more the number of trees, lesser will
# be the OOB estimate, and better the accuracy.

## H2O prediction 

# Load the h2o package.
install.packages("h2o")
library(h2o)

# Start a local h2o cluster
local.h2o <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, nthreads=-1)

# Also load the caret package
library(caret)

# Create the partition of 80-20 in the train dataset.
inTrain<- createDataPartition(train$label, p=0.8, list=FALSE)
training<-train[inTrain,]
testing<-train[-inTrain,]

# convert digit labels to factor for classification
training[,1] <- as.factor(training[,1])

# Pass dataframe from inside of the R environment to the H2O instance
trData<-as.h2o(training)
tsData<-as.h2o(testing)

# Next is to train the model. For this experiment, 5 layers of 160 nodes 
# each are used. The rectifier used is Tanh and number of epochs is 20. 
# Just used these figures as from the readings I have done and not my personal choice.

res.dl <- h2o.deeplearning(x = 2:785, y = 1,
                           trData, 
                           activation = "Tanh",
                           hidden=rep(160,5),
                           epochs = 20)

# Use model to predict testing dataset
pred.dl<-h2o.predict(object=res.dl, newdata=tsData[,-1])
pred.dl.df<-as.data.frame(pred.dl)

summary(pred.dl)
test_labels<-testing[,1]

# Calculate number of correct prediction
sum(diag(table(test_labels,pred.dl.df[,1])))

# As we can see that of total 8398 labels of the test data, we got 8106 right. Accuracy can be calculated
# by this formula - 8106/8398 = 96.522 %.
# Which is even better than random forest. 

# shut down virtual H2O cluster
h2o.shutdown(prompt = FALSE)

