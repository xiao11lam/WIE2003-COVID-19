#Importing libraries
library(dplyr)
library(DataExplorer)
library(data.table)
library(RCurl)
library(e1071)
library(corrplot)
library(RColorBrewer)
library(psych);
library(FactoMineR);
library(factoextra)
library(corrplot)
library(glmnet)
library(caret)
# ---------------------------Covid-19 Confirmed Cases----------------------------------------------------------

#Hyperparameter
pre_day = 30

# #Read Confirmed cases
confirmed <- read.csv("./datasets/time_series_covid19_confirmed_global.csv")

#Replacing spaces within Country names by underscore to calculate COR properly
confirmed$Country.Region <- gsub(" ", "_", confirmed$Country.Region)

#Removing Lat and Long from data frame
confirmed <- within(confirmed, rm("Lat", "Long", "Province.State"))

#After removing col Lat and Long, we have dates from range 2 to x and categorizing by country
confirmed <- aggregate(confirmed[,2:ncol(confirmed)], by=list(Category=confirmed$Country.Region), FUN=sum)

#Transpose the confirmed data to analyze the linear model
confirmed <- transpose(confirmed)

#Writing the data table in text file with sep=" ", col.names=FALSE, 
#quote=FALSE, row.names=FALSE to form proper structure for linear modelling
write.table(confirmed, "./confirmed.txt", sep=" ", col.names=FALSE, quote=FALSE, row.names=FALSE)

#Reading the structures table from confirmed.txt which was created above 
confirmed <- read.table("./confirmed.txt", header = TRUE, na.strings = " ")

#Association of Southeast Asian Nations            "Myanmar"
confirmed_asean <- subset(confirmed, select = c("Cambodia","Brunei","Indonesia", "Laos", 
                                          "Malaysia", "Philippines", "Singapore", "Thailand", "Vietnam"))

#Explore the overall structure of available data
plot_str(confirmed_asean)
row_num_train = nrow(confirmed_asean)-pre_day
#Divide training set and test set
confirmed_train <- confirmed_asean[1:row_num_train, ]
confirmed_test <- confirmed_asean[(row_num_train+1):nrow(confirmed_asean), ]


#####  Linear Regression Start  #####  

modelc_all = lm(Malaysia ~ ., data=confirmed_train)

prediction_all = predict(modelc_all,newdata=confirmed_test)
pred_lm = as.data.frame(prediction_all)
names(pred_lm)[names(pred_lm) == "prediction_all"] = "pred_lm"

prediction_lm_train = predict(modelc_all,newdata=confirmed_train)
lm_rmse = sqrt(mean((prediction_lm_train - confirmed_train$Malaysia)^2))

#####  Linear Regression End  #####  

##### Ridge Regression Start #####

#regularization

dummies <- dummyVars(Malaysia ~ . , data = confirmed_train)
train_dummies = predict(dummies, newdata = confirmed_train)
pred_dummies = predict(dummies, newdata = confirmed_test)

train_dummies = as.matrix(train_dummies)
pred_dummies = as.matrix(pred_dummies)
y_train = confirmed_train$Malaysia

lambdas <- 10^seq(2, -3, by = -.1)
ridge_reg = glmnet(train_dummies, y_train, nlambda = 25, alpha = 0, family = 'gaussian', lambda = lambdas)

cv_ridge <- cv.glmnet(train_dummies, y_train, alpha = 0, lambda = lambdas)
optimal_lambda <- cv_ridge$lambda.min

pred_ridge <- predict(ridge_reg, s = optimal_lambda, newx = pred_dummies)
pred_ridge = as.data.frame(pred_ridge)
names(pred_ridge)[names(pred_ridge) == "1"] = "pred_ridge"

pred_ridge_train <- predict(ridge_reg, s = optimal_lambda, newx = train_dummies)   
ridge_rmse = sqrt(mean((pred_ridge_train - confirmed_train$Malaysia)^2))

##### Ridge Regression End #####

##### Lasso Regression Start #####

lambdas <- 10^seq(2, -3, by = -.1)
lasso_reg <- cv.glmnet(train_dummies, y_train, alpha = 1, lambda = lambdas, standardize = TRUE, nfolds = 5)

lambda_best <- lasso_reg$lambda.min 
lasso_model <- glmnet(train_dummies, y_train, alpha = 1, lambda = lambda_best, standardize = TRUE)

pred_lasso <- predict(lasso_model, s = lambda_best, newx = pred_dummies)
pred_lasso = as.data.frame(pred_lasso)
names(pred_lasso)[names(pred_lasso) == "1"] = "pred_lasso"

pred_lasso_train <- predict(lasso_model, s = optimal_lambda, newx = train_dummies)                      
lasso_rmse = sqrt(mean((pred_lasso_train - confirmed_train$Malaysia)^2))
##### Ridge Regression End #####

#### Linear Regression with Dimension Reduction - PCA  Start  ####

## read and part the data
confirmed_asean$Type = NULL
confirmed_test$Malaysia = NULL

## Linear Regression with PCA
trainPredictors = confirmed_asean
trainPredictors$Malaysia = NULL
KMO(cor(trainPredictors))

pca_facto = PCA(trainPredictors,graph = F)
pca_facto$eig
fviz_eig(pca_facto,ncp=3,addlabels = T)

# moving forward with only one dimension
pca = prcomp(trainPredictors,scale. = T)
trainPredictors = data.frame(pca$x[,1:1])
trainPredictors2 = cbind(trainPredictors[1:row_num_train,],
                         confirmed_train[1:row_num_train,]$Malaysia)
trainPredictors2 = as.data.frame(trainPredictors2)
# building model
predict_pca_1 = lm(V2~.,trainPredictors2)
summary(predict_pca_1)

# making prediction
trainPredictors3 = trainPredictors[(row_num_train+1):nrow(confirmed_asean), ]
trainPredictors3 = as.data.frame(trainPredictors3)
names(trainPredictors3)[names(trainPredictors3) == "trainPredictors3"] <- "V1"

predict_pca = predict(predict_pca_1,newdata=trainPredictors3)
predict_pca = as.data.frame(predict_pca)

predict_pca_train = predict(predict_pca_1,newdata=trainPredictors2)
pca_rmse = sqrt(mean((predict_pca_train-trainPredictors2$V2)^2))

#### Linear Regression with Dimension Reduction - PCA  End ####

#### FORECAST VISUALIZATION START ####

# consolidating predictions

Malaysia_lm = as.data.frame(confirmed_train$Malaysia)
names(Malaysia_lm)[names(Malaysia_lm) == "confirmed_train$Malaysia"] = "pred_lm"
lm_backward = rbind(Malaysia_lm,pred_lm)

Malaysia_ridge = as.data.frame(confirmed_train$Malaysia)
names(Malaysia_ridge)[names(Malaysia_ridge) == "confirmed_train$Malaysia"] = "pred_ridge"
lm_ridge = rbind(Malaysia_ridge,pred_ridge)

Malaysia_lasso = as.data.frame(confirmed_train$Malaysia)
names(Malaysia_lasso)[names(Malaysia_lasso) == "confirmed_train$Malaysia"] = "pred_lasso"
lm_lasso = rbind(Malaysia_lasso,pred_lasso)

Malaysia_pca = as.data.frame(confirmed_train$Malaysia)
names(Malaysia_pca)[names(Malaysia_pca) == "confirmed_train$Malaysia"] = "predict_pca"
lm_pca = rbind(Malaysia_pca,predict_pca)

days = 1:nrow(confirmed)
days = as.data.frame(days)

library(tidyverse)
final = cbind(days,confirmed_asean$Malaysia,lm_pca,lm_lasso,lm_ridge,lm_backward)

final$predict_pca = ifelse(final$days>=row_num_train+1,final$predict_pca,"")
final$pred_lasso = ifelse(final$days>=row_num_train+1,final$pred_lasso,"")
final$pred_ridge = ifelse(final$days>=row_num_train+1,final$pred_ridge,"")
final$pred_lm = ifelse(final$days>=row_num_train+1,final$pred_lm,"")

final = gather(final,key = "model", value= "value",2:6)
final$value = as.integer(final$value)
final$linetype = ifelse(final$model == 'confirmed_asean$Malaysia',1,2)
final$linetype = as.factor(final$linetype)

## FINAL PLOT ##

library(ggplot2);library(ggrepel)
palette = c("yellow",'#ffa41b','#AEC4EB',
            '#18b0b0','#fe346e',"#1f6650")

ggplot(final,aes(x=days,y=value,color=model,shape=linetype))+
        geom_line(aes(linetype = linetype),size=1.15)+guides(linetype = FALSE)+
        geom_text_repel(aes(label=ifelse(days>nrow(confirmed_asean),as.character(round(value,0)),'')),hjust= -0.5)+
        scale_color_manual(values=palette)+
        scale_x_continuous(breaks=seq(0,130,10))+
        scale_y_continuous(breaks=seq(0,10000,500))+
        labs(title = "Malaysia Coronavirus Case Prediction Result",y= "Coronavirus Case", x = "Days")+
        labs(colour="Models Used")+
        scale_colour_discrete(labels = c("Actual Case", "Model 1 (lm)", "Model 2 (ridge)","Model 3 (lasso)","Model 4 (pca)"))+
        guides(color = guide_legend(override.aes = list(linetype = 1, size=3)))+
        theme(
                legend.position = "right",
                axis.text = element_text(colour = "white"),
                axis.title.x = element_text(colour = "white", size=rel(1.7)),
                axis.title.y = element_text(colour = "white",size=rel(1.7)),
                panel.background = element_rect(fill="black",colour = "black"),
                panel.grid = element_blank(),
                plot.background = element_rect(fill="black",colour = "black"),
                legend.key = element_rect(fill = "black",colour = "black"),
                legend.background = element_blank(),
                legend.text = element_text(colour="white",size = rel(1)),
                legend.title = element_text(colour="white",size = rel(1)),
                panel.grid.minor = element_line(colour="#202020", size=0.3),
                plot.title = element_text(color="white", size= rel(2),hjust = 0))

