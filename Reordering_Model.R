library('data.table')
library('dplyr')
library('modeest')
library('glmnet')
library('pROC')
library('PRROC')
library('caret')
library('ggplot2')
library('randomForest')
library('tree')


#reading the data
orders <- fread("orders.csv")
#removing all orders belonging to the test_users (75000 test users)
test_users <- unique(orders$user_id[which(orders$eval_set =='test')])
orders <- orders[!orders$user_id %in% test_users,]
op_train <- fread("order_products__train.csv")
op_prior <- fread("order_products__prior.csv")

com_prior <- merge.data.table(orders,op_prior,by='order_id')
rm(op_prior)
length(unique(com_prior$order_id)) #2047377
length(unique(orders$order_id)) #2178586
summary(as.factor(orders$eval_set)) #  prior:2047377    train:  131209 

prod <- fread('products.csv')
aisles <- fread('aisles.csv')
dept <- fread('departments.csv')
com_prod_char <- merge.data.table(merge.data.table(prod,aisles,by='aisle_id'),dept,by='department_id')

com_prior_all <- merge.data.table(com_prior,com_prod_char,by='product_id') 
rm(prod,aisles,dept)
rm(com_prior)

#contains the combined data from all datasets
str(com_prior_all)
summary(as.factor(com_prior_all$eval_set)) #whole column is prior
#we can remove it
com_prior_all <- com_prior_all %>% select(-'eval_set') #now this contains only the prior orders


#FEATURE ENGINEERING

#USER BASED FEATURES

#first for each user lets look at the cart size
cart_size_df <- data.frame(com_prior_all %>% group_by(user_id,order_number) %>% summarise(count=n()))
#for each user what is the mean and mode cart_size
user_char <- data.frame(cart_size_df %>% group_by(user_id) %>% summarise(mean = mean(count),mode = mlv(count, method='mfv')[1]))
user_char['cart_size_sd']<- data.frame(cart_size_df %>% group_by(user_id) %>% summarise(sd = sd(count)))['sd']
colnames(user_char)[c(2,3)] <- c('cart_size_mean','cart_size_mode')
#each user what is the mode of days_since_prior_order
user_char['days_since_mode'] <- data.frame(com_prior_all%>% group_by(user_id) %>% summarise(mode = round(mlv(days_since_prior_order, method='mfv', na.rm = TRUE))[[1]]))['mode'] 
user_char['days_since_mean'] <- data.frame(com_prior_all%>% group_by(user_id) %>% summarise(mean = mean(days_since_prior_order,na.rm = TRUE)))['mean'] 
user_char['days_since_sd'] <- data.frame(com_prior_all%>% group_by(user_id) %>% summarise(sd = sd(days_since_prior_order,na.rm = TRUE)))['sd'] 
#what is the users tendency to reorder, seeing the fraction of reorders in each orders by the user
reorder <- data.frame(com_prior_all%>% group_by(user_id,order_number) %>% summarise(sum= sum(reordered),count=n()))
reorder['reorder_tendency'] <- reorder$sum/reorder$count
user_char['reorder_tendency_mean'] <-data.frame(reorder %>% group_by(user_id) %>% summarise(mean =mean(reorder_tendency)))['mean']
user_char['reorder_tendency_sd'] <-data.frame(reorder %>% group_by(user_id) %>% summarise(sd =sd(reorder_tendency)))['sd']

#what is the mode order_dow, hour of day for each user
user_char['mode_dow'] <- data.frame(com_prior_all%>% group_by(user_id)%>% summarise(mode = mlv(order_dow, method='mfv')[1]))['mode']
user_char['dow_sd'] <- data.frame(com_prior_all%>% group_by(user_id)%>% summarise(sd = sd(order_dow)))['sd']
user_char['mode_hod'] <- data.frame(com_prior_all%>% group_by(user_id)%>% summarise(mode = mlv(order_hour_of_day, method='mfv')[1]))['mode']
user_char['hod_sd'] <- data.frame(com_prior_all%>% group_by(user_id)%>% summarise(sd = sd(order_hour_of_day)))['sd']

#what is the mode aile_id/dept_id for each user
user_char['mode_aisle_id'] <- data.frame(com_prior_all%>% group_by(user_id)%>% summarise(mode = mlv(aisle_id, method='mfv')[1]))['mode']
user_char['mode_dept_id'] <- data.frame(com_prior_all%>% group_by(user_id)%>% summarise(mode = mlv(department_id, method='mfv')[1]))['mode']

#number of times the user continuosly ordered new products (strike of new orders)
#streak of new products 
i=52200 #user_id=74638
users <- unique(reorder$user_id)
streak = c()
for (i in 1:131209){
  print(i)
  u <- subset(reorder,user_id == users[i])
  z=0
  c=0
  for (j in 1:nrow(u)){
    if(u$sum[j]==0 & u$sum[j+1]==0 & j!=nrow(u)){
      z= z+1
      c=c+1
    }
    else{
      z=0
    }
  }
  streak <- c(streak,c)
}

user_char['streak_non_reorders'] <- streak


#reorder tendency of the product for the user, seeing the fraction of orders for which the product was reordered by the user
#number of times the product was 'reordered'/ total number of orders for the user
reorder_prod <- data.frame(com_prior_all%>% group_by(user_id,product_id) %>% summarise(sum= sum(reordered)))
dim(reorder_prod) #8,474,661       4
reorder_prod <- merge.data.table(reorder_prod,data.frame(com_prior_all%>% group_by(user_id) %>% summarise(count=n_distinct(order_number))),by ='user_id',all.x = TRUE) 
reorder_prod['reorder_prod_tend'] <- reorder_prod$sum/reorder_prod$count
#reordering ratio of the product i.e. overall number of times reordered/number of times product ordered
prod_char <- data.frame(com_prior_all %>% group_by(product_id) %>% summarise(sum=sum(reordered),count=n()))
prod_char['over_all_reorder'] <- prod_char$sum/prod_char$count
#combining this with user_prod char
reorder_prod <- merge.data.table(reorder_prod,prod_char[,c(1,4)],by ='product_id',all.x = TRUE)
#for each product the user has bought what is the mode add_to_cart_order
user_prod_char <- data.frame(com_prior_all %>% group_by(user_id,product_id) %>% summarise(mode = mlv(add_to_cart_order, method='mfv')[1]))['mode']
user_prod_char_1 <- data.frame(com_prior_all %>% group_by(user_id,product_id) %>% summarise(sd = sd(add_to_cart_order,na.rm = TRUE)))
user_prod_char_1['mode'] <- user_prod_char$mode
colnames(user_prod_char_1)[c(3,4)] <- c('add_to_cart_sd','add_to_cart_mode')
user_prod_char <- merge.data.table(reorder_prod,user_prod_char_1, by=c('user_id','product_id'))

#final table
final <- merge.data.table(user_prod_char,user_char,by='user_id',all.x = TRUE)
colnames(final)[3] <- 'number_product_reordered'
colnames(final)[4] <- 'total_order_user'

#was the product in the current cart in the users penultimate cart
penultimate <- data.frame(com_prior_all %>% group_by(user_id,product_id) %>% summarise(max=max(order_number)))
com_train <- merge.data.table(op_train,orders,by='order_id')
final_order <- data.frame(com_train %>% group_by(user_id,product_id) %>% summarise(max = max(order_number)))
colnames(final_order)[3] <- 'max_1'
final_order <- merge(penultimate,final_order,by=c('user_id','product_id'),all= TRUE)
#here max.x=0 means that the user ordered this product for the first time in this current cart
#max.y=0 means that the user did not order this product in the current cart, there is no user product order history for these
#finding which is the maximum order number for the user in the prior orders, this will be the user's penultimate cart
user_max_order <- data.frame(com_prior_all %>% group_by(user_id) %>% summarise(max =max(order_number)))
colnames(user_max_order)[2] <- 'max_order_num'
final_order <- merge(final_order,user_max_order,by='user_id')
final_order['penultimate'] <- ifelse(final_order$max_order_num==final_order$max,1,0)
final_order['penultimate'] <- ifelse(is.na(final_order$penultimate)==TRUE,0,final_order$penultimate)

#after joining com_prior_all with training data

#Target Variable
final_order['y'] <- ifelse(is.na(final_order$max_1)==FALSE,1,0) #is in user's current cart or not

#getting all the order numbers in which the product was present for the user
user_prod_all_ord <- data.frame(com_prior_all %>% group_by(user_id,product_id) %>% summarize(order_number = paste(sort(unique(order_number)),collapse=", ")))
user_order_days_since <- data.frame(com_prior_all %>% group_by(user_id,order_number) %>% summarize(max=max(days_since_prior_order)))
user_order_days_since$max <- ifelse(is.na(user_order_days_since$max)==TRUE,0,user_order_days_since$max)
user_order_days_since <- user_order_days_since %>% group_by(user_id) %>% mutate(Total =cumsum(max))
com_prior_all <- merge(com_prior_all,data.frame(user_order_days_since[,c(1,2,4)]),by=c('user_id','order_number'),all.x = TRUE)
#getting all the days since prior order for all the orders in which the product was present for the user
user_prod_all_ord['days_since_prod'] <- data.frame(com_prior_all %>% group_by(user_id,product_id) %>% summarize(Total = paste(sort(unique(Total)),collapse=", ")))['Total']


dummy_1 <- list() #list of lists of numeric arrays of days since prior order of the product by the user
dummy <- strsplit(user_prod_all_ord$days_since_prod,', ')
for(i in 1:length(dummy)){
  dummy_1[[i]] <- as.numeric(dummy[[i]])
}

dummy_2 <- lapply(dummy_1, diff) #has the time interval between two orders of that product
#mode for days_since for each product by each user, in case no unimode, take the shortest interval
#because shortest interval in which the user needed that product
len <- lapply(dummy_2, length)

#mode of the interval at which the product was ordered by the user
mode_days_since_prod_diff <- c()

for(i in i:length(dummy_2)){
  if(len[i]>1){
    print(i)
    mode_days_since_prod_diff <- c(mode_days_since_prod_diff,mlv(dummy_2[[i]], method='mfv', na.rm = TRUE)[1])
  }else if(len[i]==1){
    mode_days_since_prod_diff <- c(mode_days_since_prod_diff,unlist(dummy_2[[i]]))
  }
  else{
    mode_days_since_prod_diff <- c(mode_days_since_prod_diff,NA)
  }
}

count_cont_reorder <- function(x){
  return(length(which(x==1)))
}
cont_order <-unlist(lapply(dummy_2,count_cont_reorder ))
user_prod_all_ord <- cbind(user_prod_all_ord,cont_order)

count_days_since_prod_ordered <- function(x){
  return(mlv(x, method='mfv', na.rm = TRUE)[1])
}

days_since_prod_ordered_mode <- unlist(lapply(dummy_2,count_days_since_prod_ordered))
user_prod_all_ord <- cbind(user_prod_all_ord,days_since_prod_ordered_mode)


ref_days_since_prod_ordered <- function(x){
  return(ifelse(length(x)==1,ifelse(as.numeric(x)!=0,x,NaN),NaN))
}
l <- unlist(lapply(dummy_1,ref_days_since_prod_ordered))
user_prod_all_ord <- cbind(user_prod_all_ord,l)
l_1 <- ifelse(is.na(user_prod_all_ord$l)==TRUE,user_prod_all_ord$days_since_prod_ordered_mode,user_prod_all_ord$l)
user_prod_all_ord <- cbind(user_prod_all_ord,l_1)
user_prod_all_ord <- user_prod_all_ord[,-c(7,8)]

#when was the product last ordered by the user
user_order_days_since <- data.frame(com_prior_all %>% group_by(user_id,order_number) %>% summarize(max=max(days_since_prior_order)))
user_order_days_since$max <- ifelse(is.na(user_order_days_since$max)==TRUE,0,user_order_days_since$max)
user_order_days_since <- user_order_days_since %>% group_by(user_id) %>% mutate(Total =cumsum(max))
com_prior_all <- merge(com_prior_all,data.frame(user_order_days_since[,c(1,2,4)]),by=c('user_id','order_number'),all.x = TRUE)
user_prod_all_ord['days_since_prod_latest'] <- data.frame(com_prior_all %>% group_by(user_id,product_id) %>% summarize(Total = paste(sort(unique(Total)),collapse=", ")))['Total']
user_max_days <- data.frame(user_order_days_since %>% group_by(user_id) %>% summarise(max=max(Total)))
user_prod_all_ord <- merge.data.table(user_prod_all_ord,user_max_days,by='user_id')

when_prod_last_ordered <- function(x){
  return(x[length(x)])
}

l <- unlist(lapply(dummy_1,when_prod_last_ordered))
user_prod_all_ord <- cbind(user_prod_all_ord,l)
when_prod_last_ord <- user_prod_all_ord$max - user_prod_all_ord$l
user_prod_all_ord <- cbind(user_prod_all_ord,when_prod_last_ord)
user_prod_all_ord <- user_prod_all_ord %>% select(-c('max','l'))

final_order_1 <- merge(final_order,user_char,by='user_id')
user_prod_char_1 <- user_prod_char[,1:8]
final_order_1 <- merge(final_order_1,user_prod_char_1,by=c('user_id','product_id'),all.x = TRUE)
final_order_1 <- merge(final_order_1,user_prod_all_ord,by=c('user_id','product_id'),all.x = TRUE)
user_prod_all_ord <- rename(user_prod_all_ord,days_since_prod_mode=l_1)



#mean interval at which the product was ordered
mean_days_since <- function(x){
  return(mean(x,na.rm = TRUE))
}
l <- unlist(lapply(dummy_2,mean_days_since ))
user_prod_all_ord <- cbind(user_prod_all_ord,l)
user_prod_all_ord <- rename(user_prod_all_ord,days_since_prod_mean=l)

#sd of the intervals at which the product was ordered
sd_days_since <- function(x){
  return(sd(x,na.rm = TRUE))
}
l <- unlist(lapply(dummy_2,sd_days_since ))
user_prod_all_ord <- cbind(user_prod_all_ord,l)
user_prod_all_ord <- rename(user_prod_all_ord,days_since_prod_sd=l)

final_order_1 <- merge(final_order_1,user_prod_all_ord[,c(1,2,8,9)],by=c('user_id','product_id'),all.x=TRUE)

min_prod_add <- data.frame(com_prior_all %>% group_by(user_id,product_id) %>% summarise(min=min(add_to_cart_order)))
colnames(min_prod_add)[3] <- 'min_prod_add'
final_order_1 <-merge(final_order_1,min_prod_add,by=c('user_id','product_id'),all.x = TRUE)
final_order_1 <- merge(final_order_1,com_prod_char,by='product_id')


#Indicator variables whether the product beloged to the aisle and department that the user orders the most from i.e. mode aisle & dept


final_order_1['aisle_indicator'] = ifelse(final_order_1$mode_aisle_id == final_order_1$aisle_id, 1, 0)
final_order_1['dept_indicator'] = ifelse(final_order_1$mode_dept_id == final_order_1$department_id, 1, 0)

#writing the final to csv to be able to conveniently use for modelling
write.csv(final_order_1, 'final_order_1.csv')

########################################################################################################################################
#MODELLING
#1. Log-Lasso model
#2. Decision Tree
#3. Random Forest
#4. XGB Tree
########################################################################################################################################

df <- fread('final_order_1.csv')
df <- df[,-1]

cols = c("product_id","user_id", "penultimate", "y",  "cart_size_mean",  "cart_size_mode", "days_since_mode" ,"cart_size_sd", "days_since_mean" , "days_since_sd",  "reorder_tendency_mean", "reorder_tendency_sd",  
         "mode_dow",  "dow_sd", "mode_hod", "hod_sd" , "streak_non_reorders", "total_order_user", "reorder_prod_tend",  "over_all_reorder",  "add_to_cart_mode","min_prod_add", "cont_order", "days_since_prod_mode",  "when_prod_last_ord",    "days_since_prod_mean", "days_since_prod_sd",    "department_id",         "aisle_indicator",       "dept_indicator")

colnames(df)

df = df[,c("product_id","user_id", "penultimate", "y",  "cart_size_mean",  "cart_size_mode", "days_since_mode" ,"cart_size_sd", "days_since_mean" , "days_since_sd",  "reorder_tendency_mean", "reorder_tendency_sd",  
                                 "mode_dow",  "dow_sd", "mode_hod", "hod_sd" , "streak_non_reorders", "total_order_user", "reorder_prod_tend",  "over_all_reorder",  "add_to_cart_mode","min_prod_add", "cont_order", "days_since_prod_mode",  "when_prod_last_ord",    "days_since_prod_mean", "days_since_prod_sd",    "department_id",         "aisle_indicator",       "dept_indicator")]


df$days_since_prod_mode <- ifelse(is.na(df$days_since_prod_mode)==TRUE,0,df$days_since_prod_mode)
df$days_since_prod_mean <- ifelse(is.na(df$days_since_prod_mean)==TRUE,df$days_since_prod_mode,df$days_since_prod_mean)
df$days_since_prod_sd <- ifelse(is.na(df$days_since_prod_sd)==TRUE,0,df$days_since_prod_sd)

colnames(df)
sum(is.na(df))

mod_data <- df[,-c(1,2)]
rm(df)
colnames(mod_data)
mod_data$mode_dow <- as.factor(mod_data$mode_dow)
mod_data$mode_hod <- as.factor(mod_data$mode_hod)
mod_data$department_id <- as.factor(as.character(mod_data$department_id))
mod_data$aisle_indicator <- as.factor(mod_data$aisle_indicator) 
mod_data$dept_indicator <- as.factor(mod_data$dept_indicator)
mod_data$y <- as.factor(mod_data$y)

find_NA <- function(x){
  return((sum(is.na(x))/nrow(df))*100)
}
apply(df,2,find_NA)

sum(is.na(mod_data))

#1. LOGISTIC LASSO
x <- model.matrix(y~.,mod_data)[,-1] 
y <- mod_data$y

set.seed(2020)
train <- sample(floor(0.75*(nrow(mod_data))))
train_x = x[train,]
train_y = y[train]
test_x = x[-train,]
test_y = y[-train]
rm(x)
lasso.mod <- glmnet(train_x,train_y,alpha=1,family = "binomial")
log_lasso_1 <-  cv.glmnet(train_x , train_y , family = "binomial", alpha = 1, nfolds = 10)
bestlam <- log_lasso_1$lambda.min

lasso.pred_class <- predict(lasso.mod,newx=test_x,type='class',s=bestlam)
lasso.pred_prob <- predict(lasso.mod,newx=test_x,type='response',s=bestlam)
coef <- coef(lasso.mod,s=bestlam)
coef

lasso_roc <- roc(test_y, lasso.pred_prob,plot = TRUE, print.auc = TRUE) 
coords(lasso_roc, "best", ret = "threshold") 
lasso_class <- as.factor(ifelse(lasso.pred_prob>0.09552832,1,0))
confusionMatrix(lasso_class,test_y,positive='1',mode = "everything") 
plot(pr.curve(lasso.pred_prob[test_y==1], lasso.pred_prob[test_y==0],curve = T))
plot(roc.curve(lasso.pred_prob[test_y==1], lasso.pred_prob[test_y==0],curve = T))

confusionMatrix(as.factor(lasso.pred_class),test_y,positive='1',mode = "everything") 
rm(train_x,test_x,train_y,test_y)


#2. DECISION TREE
tree_model <- tree(y~.,train_data)
#plotting the model
plot(tree_model)
text(tree_model ,pretty =0)

#predicting
y_pred <- predict(tree_model,test_x,type="class")
tree_roc <- roc(test_y, y_pred[,2],plot = TRUE, print.auc = TRUE)
coords(tree_roc, "best", ret = "threshold")
plot(roc.curve(y_pred[,2][test_y==1], y_pred[,2][test_y==0],curve = T))

plot(pr.curve(y_pred[,2][test_y==1], y_pred[,2][test_y==0],curve = T))
confusionMatrix(y_pred,test_y,positive='1',mode = "everything")

#3. RANDOM FOREST
train_data = mod_data[train,]
test_x = mod_data[-train,]
test_y= mod_data$y[-train]
rm(mod_data)
rf1 <- randomForest(y~.,mod_data,ntrees=500,subset = train,importance=TRUE) 
rf1_pred_prob <- predict(rf1,mod_data[-train,-2],type='prob') #predicting on test data
rf1_pred_class <- predict(rf1,mod_data[-train,-2],type='class')

rf_roc <- roc(test_y, rf1_pred_prob[,2],plot = TRUE, print.auc = TRUE) 
coords(rf_roc, "best", ret = "threshold")
rf_class <- as.factor(ifelse(rf1_pred_prob[,2]>0.127,1,0)) 
confusionMatrix(rf_class,test_y,positive='1',mode = "everything") 

confusionMatrix(as.factor(rf1_pred_class),test_y,positive='1',mode = "everything") 

plot(pr.curve(rf1_pred_prob[,2][test_y==1], rf1_pred_prob[,2][test_y==0],curve = T)) #pr_curve for rf
plot(roc.curve(rf1_pred_prob[,2][test_y==1], rf1_pred_prob[,2][test_y==0],curve = T))


#3. RANDOM FOREST with adjustment for imbalanced classes
train_data = mod_data[train,]
test_x = mod_data[-train,]
test_y= mod_data$y[-train]
rm(mod_data)
rf2 <- randomForest(y~.,train_data,ntrees=500,sampsize=c("0"=50000,"1"=20000),strata=train_data$y) #making 1s 40% of the data
rf2_pred_prob <- predict(rf2,test_x[,-2],type='prob') #predicting on test data
rf2_pred_class <- predict(rf2,test_x[,-2],type='class')

rf_roc <- roc(test_y, rf2_pred_prob[,2],plot = TRUE, print.auc = TRUE) 
coords(rf_roc, x="best", input="threshold", best.method="youden")
rf_class <- as.factor(ifelse(rf2_pred_prob[,2]>0.317,1,0)) 
confusionMatrix(rf_class,test_y,positive='1',mode = "everything") 

confusionMatrix(as.factor(rf2_pred_class),test_y,positive='1',mode = "everything") 

plot(pr.curve(rf2_pred_prob[,2][test_y==1], rf2_pred_prob[,2][test_y==0],curve = T)) #pr_curve for rf
plot(roc.curve(rf2_pred_prob[,2][test_y==1], rf2_pred_prob[,2][test_y==0],curve = T))


#4. XGB TREE
f1 <- function (ypred, train) {
  require(ModelMetrics)
  y = getinfo(dtrain, "label")
  dt <- data.table(user_id=train[user_id %in% val_users, user_id], purch=y, pred=ypred)
  f1 <- mean(dt[,.(f1score=f1Score(purch, pred, cutoff=0.2)), by=user_id]$f1score)
  return (list(metric = "f1", value = f1))
}

params <- list(booster="gbtree"
               ,objective="reg:logistic"
               ,eval_metric= f1
               ,eta=0.1
               ,gamma=0
               ,max_depth=5
               ,subsample=1
               ,colsample_bytree=1
               ,base_score=0.2
               ,nthread=8
)
X <- xgboost::xgb.DMatrix(data.matrix(train_data[,-2]), label = as.numeric(as.character(train_data$y)))
xgb_boost <- xgboost::xgb.train(data=X, params=params, nrounds=100, verbose = 2)
xgb_pred_prob <- predict(xgb_boost,data.matrix(test_x[,-2]),type='prob') #predicting on test data
xgb_roc <- roc(test_y, xgb_pred_prob,plot = TRUE, print.auc = TRUE) 
coords(xgb_roc, "best", ret = "threshold")
xgb_class <- as.factor(ifelse(xgb_pred_prob>0.2,1,0)) 
cf =table(truth = as.factor(test_y), predicted = xgb_class)
F1_Score(as.factor(test_y), xgb_class, positive = 1) 
Accuracy(as.factor(test_y), xgb_class) 
Precision(as.factor(test_y), xgb_class, positive = 1) 
Recall(as.factor(test_y), xgb_class, positive = 1) 
plot(roc.curve(xgb_pred_prob[test_y==1], xgb_pred_prob[test_y==0],curve = T))
plot(pr.curve(xgb_pred_prob[test_y==1], xgb_pred_prob[test_y==0],curve = T))
importance <- xgboost::xgb.importance(dimnames(train_data[,-2])[[2]], model = xgb_boost)
xgboost::xgb.plot.importance(importance,measure = 'Importance')



