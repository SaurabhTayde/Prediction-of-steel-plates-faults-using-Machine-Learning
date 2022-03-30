#setwd('D:/Saurabh T/R/Project/Steel Plates')

########### Load Data

SteelData = read.csv('SteelPlatesData_Updated.csv')

########### Data summary 
head(SteelData)
unique(SteelData$FaultType)

str(SteelData)

table(SteelData$FaultType)
summary(SteelData)


########## Data Partition

set.seed(100)

library(caTools)

index = sample.split(SteelData$FaultType, SplitRatio = 0.7)

SteelTrain = SteelData[index, ]
SteelTest = SteelData[!index,]

str(SteelTrain)
str(SteelTest)


########## Model Building:

library(randomForest)
set.seed(123)
rf = randomForest(FaultType ~ ., data = SteelTrain)
rf                                                 ## OOB error rate = 17.68%

attributes(rf)



######### Prediction - train data:

library(caret)

p1 = predict(rf, SteelTrain)

head(p1)
head(SteelTrain$FaultType)


######### Confusion Matrix:

confusionMatrix(p1, SteelTrain$FaultType)

#Now here, the confusion matrix in 'rf' and above confusion matrix both 
#giving different accuracy. This is because initially model was not familiar
#with all the observations as each bootstrap tree considers 2/3 rd data only.
#But now as model is built with almost all the data in training sample 
#and model is familiar with train data.Also we are predicting for training sample only.
#So accuracy is 100%


p2 = predict(rf, SteelTest)

confusionMatrix(p2, SteelTest$FaultType)  #Accuracy : 84.12 %


#************ Error rate of Random Forest ************#

plot(rf)

#So we here we can notice that after 200 trees, error rate is remaining constant
#So we will tune our model


#************  Tune Random Forest Model  ************#


tuneRF(SteelTrain[,-28], SteelTrain[,28], stepFactor = 1, plot = T, ntreeTry = 300,
       trace = T, improve = 1)

#StepFactor : At each iteration mtry is inflated or deflated by this value.

#In above plot we are just getting one point, so let's adjust stepfactor and 'improve'


tuneRF(SteelTrain[,-28], SteelTrain[,28], stepFactor = 0.5, plot = T, ntreeTry = 300,
       trace = T, improve = 1)

#So as per graph, we are getting good results with  mtry = 10,
#as graph is at lowest point at this position
#Also, after 300 trees error rate is remaining constant
#So lets create another rf model:


rf2 = randomForest(FaultType ~., data = SteelTrain, ntree = 300, mtry =10,
                   importance = T, proximity = T)
rf2

#So we can notice that our accuracy has been improved.

#Lets check our new model on test data

p3 = predict(rf2, SteelTest)

confusionMatrix(p3, SteelTest$FaultType)

#So we can notice that in confusion matrix also our accuracy has been improved.



######## No. of nodes for trees

hist(treesize(rf2),main = "No. of Nodes for the Trees",
     col = "green")

#So from graph we can say that most of the trees have 135 nodes

varImpPlot(rf2,
           sort = T,
           n.var = 10,
           main = "Top 10 - Variable Importance")
importance(rf)
varUsed(rf)


# Extract Single Tree
getTree(rf2, 1, labelVar = TRUE)


# Multi-dimensional Scaling Plot of Proximity Matrix
MDSplot(rf2, SteelTrain$FaultType)

