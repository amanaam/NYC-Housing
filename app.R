#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#
if(!require(shinydashboard)){
  install.packages("shinydashboard")
  library(shinydashboard)
}
library(shinydashboard)
if(!require(caret)){
  install.packages("caret")
  library(caret)
}
library(shiny)
#----Libraries----
library(tidyverse)
if(!require(formattable)){
  install.packages("formattable")
  library(formattable)
}
library(caret)
library(plotly)
library(formattable)
library(stats)
if(!require(stats)){
  install.packages("stats")
  library(stats)
}
if(!require(graphics)){
  install.packages("graphics")
  library(graphics)
}
library(graphics)
library(base)
if(!require(base)){
  install.packages("base")
  library(base)
}
library(ggplot2)

user <- unname(Sys.info()["user"])
if (user == "shiny") {
  
  # Set library locations
  .libPaths(c(
    "C:/Program Files/R/R-4.0.2/library"
  )
  )
  
}
options(scipen = 999)
#---Data Cleaning---------------------------------------------------------------

data = read.csv("sales.csv")
colnames(data) <- gsub(" ", ".", colnames(data), fixed=TRUE)
str(data)
colnames(data)= tolower(names(data))#fixing column names

summary(data)

data[!(apply(data, 1, function(y) any(y == 0))),]#removing entries with 0 as the value

data$sale.price=as.numeric(data$sale.price)
data$gross.square.feet=as.numeric(data$gross.square.feet)
data$land.square.feet=as.numeric(data$land.square.feet)

#Shortlisting the data to ensure its accurate
data = dplyr::filter(data, sale.price>700000, sale.price<10000000, year.built > 1500, residential.units>0,
                     land.square.feet>100, gross.square.feet>100)

#Analysing the neighborhood column to only include neighborhoods with more then 100 sales.
neighborhoods= data%>%
  group_by(neighborhood)%>%
  summarise(count = n())%>%
  arrange(desc(count))
data = merge(data, neighborhoods)
data = dplyr::filter(data, count>100)

#dummy variables were removed as they were causing problems for the regression.
# data$class1 =ifelse(data$tax.class.at.time.of.sale==1,1,0)
# data$class2=ifelse(data$tax.class.at.time.of.sale==2,1,0)
# data$class3=ifelse(data$tax.class.at.time.of.sale==3,1,0)
# data$class4=ifelse(data$tax.class.at.time.of.sale==4,1,0)

#instead the column was made into factor
data$tax.class.at.time.of.sale = as.factor(data$tax.class.at.time.of.sale)

#selecting the columns to be used for analysis
data=dplyr::select(data,neighborhood,
                   block, lot, land.square.feet,
                   gross.square.feet, commercial.units, residential.units,
                   sale.price, year.built, tax.class.at.time.of.sale)
data = na.omit(data)

#----Data Partition----
set.seed(1)
train.index = sample(row.names(data), dim(data)[1]*0.8)
train.df = data[train.index,]
test.index <- setdiff(row.names(data), train.index)
test.df <- data[test.index,]

#----Models----
#Linear Model
linear_model = lm(sale.price~.,data = train.df)#training model
pred_linear <-predict(linear_model,test.df)#predictions

#LASSO Regression
ctrl = trainControl(method="cv",      # simple cross-validation
                    number = 10,      # 10 folds
)
lasso_model = train(sale.price ~ ., data = train.df,
                    method = "lasso",
                    trControl = ctrl
)
pred_lasso <-predict(lasso_model,test.df)

#SVM
ctrl = trainControl(method="cv",      # simple cross-validation
                    number = 10,      # 10 folds
)
svm_model <- train(
  sale.price ~ ., data = train.df, method = "svmLinear",
  trControl = ctrl,
  preProcess = c("center","scale")
)
pred_svm <-predict(svm_model,test.df)


#----Models with Log of Sale Price----
#partitioning 
data$sale.price.log = log(data$sale.price)
train.index.log = sample(row.names(data), dim(data)[1]*0.8)
train.df.log = data[train.index.log,]
test.index.log <- setdiff(row.names(data), train.index.log)
test.df.log <- data[test.index.log,]

#Linear Model
linear_model.log = lm(sale.price.log~.,data = train.df.log)
pred_linear.log <-predict(linear_model.log,test.df.log)

#LASSO Regression
lasso_model.log = train(sale.price.log ~ ., data = train.df.log,
                        method = "lasso",
                        trControl = ctrl
)

pred_lasso.log <-predict(lasso_model.log,test.df.log)

#SVM
svm_model.log <- train(
  sale.price.log ~ ., data = train.df.log, method = "svmLinear",
  trControl = ctrl,
  preProcess = c("center","scale")
)
pred_svm.log <-predict(svm_model.log,test.df.log)

results <- data.frame(pred_linear, pred_lasso, pred_svm,
                      exp(pred_linear.log), exp(pred_lasso.log), exp(pred_svm.log),
                      (pred_linear+pred_lasso+pred_svm+exp(pred_linear.log)+exp(pred_lasso.log)+exp(pred_svm.log))/8,
                      abs(round((test.df$sale.price-pred_linear)*100/test.df$sale.price)),
                      abs(round((test.df$sale.price-pred_lasso)*100/test.df$sale.price)),
                      abs(round((test.df$sale.price-pred_svm)*100/test.df$sale.price)),
                      abs(round((test.df.log$sale.price-exp(pred_linear.log))*100/test.df.log$sale.price)),
                      abs(round((test.df.log$sale.price-exp(pred_lasso.log))*100/test.df.log$sale.price)),
                      abs(round((test.df.log$sale.price-exp(pred_svm.log))*100/test.df.log$sale.price)),
                      abs(round((test.df$sale.price-(pred_linear+pred_lasso+pred_svm+pred_lasso.log+pred_linear.log+pred_svm.log)/6)*100/test.df$sale.price)),
                      test.df.log$sale.price)

colnames(results) <- c('LinearModel', 'LassoModel', 'SVMmodel','LogLinearModel', 'LogLassoModel', 'LogSVMmodel', "AveragePrice",
                       'LinearModelError', 'LassoModelError', 'SVMmodelError','LogLinearModelError', 'LogLassoModelError', 'LogSVMmodelError', "AveragePriceError", "sale.price")



results = results%>%
  dplyr::filter(LinearModel<1000000, LinearModel>700000, LassoModel<1000000, LassoModel>700000, SVMmodel<1000000, SVMmodel>700000,
         LogLinearModel<1000000, LogLinearModel>700000, LogLassoModel<1000000, LogLassoModel>700000, LogSVMmodel<1000000, LogSVMmodel>700000)

# To turn off the scientific notation
#Warning: Boosting Might take extremely long time to train
ui <- dashboardPage(skin = "black",
  dashboardHeader(title = "NYC Housing Price Predictor",
                  titleWidth = 400),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Executive Summary", tabName = "Summary"),
      menuItem("Problem Description", tabName = "problem"),
      menuItem("Data Description", tabName = "data"),
      menuItem("Models and Evaluations", tabName = "models"),
      menuItem("Conclusions and Recommendations", tabName = "Recomendations")
    )
  ),
  dashboardBody(
    tabItems(
      # First tab content
      tabItem(tabName = "Summary",
              h3("Summary"),
              p(
                "The problem we looked at was creating a model to predict housing prices in New York. The Real Estate market is a massive market especially in New York, as millions of people buy and sell properties every year.  There are 80,000 holdings sold in New York City per year which demonstrates just how large this market is.  The uses and purposes of buying these properties vary greatly, as some buy properties as an investment or they are buying their house for the next 25 years. Both buyers and sellers will be looking to get the best price possible to maximize their investment.  The main goal of this project is to create a tool that both buyers and sellers can use to ensure that they are getting a fair price. This tool could help reduce the gap in valuation between buyers and sellers, which would help make negotiations easier. The main objective of this project is to prevent the uncertainty surrounding pricing for both buyers and sellers. With this tool, a seller can receive a fair amount for their property, well the buyer can get good value. Furthermore, this model could help forecast the price of properties in New York to help our clients have an idea about when the best time to sell and buy would be."
              ),
              p(
                "The data we used for this project was annual New York City property sales data published by the New York City government from 2017-2019.  The data includes information such as address, neighborhood, building class, square footage, number of residential/ commercial units, and tax class. The data also includes the date it was sold and the amount it was sold for.  The data does not include the parties involved and any information about the shape the property is in or any interior design.  Before creating our models we cleaned the data, by removing rows with zero values and NAs as well as removed columns that were not relevant to our analysis."
              ),
              p(
                "After subsetting the data to focus on a price range of  $700k-$10mil, we implemented four models to solve this problem: Linear Regression, Lasso, the average price of models, and SVM.  After comparing the average percentage error for each model, we determined that the SVM model was most accurate with an average percentage error of 10.65%.  The client should be aware that the model works best on houses between $700k-$10 million since the model was created based on this range which limits the types of property that the model can be used on.  Furthermore, the data was only from 2017-2019, so the model does not incorporate how the pandemic has affected the housing market.  This time range could affect both the model's accuracy of forecasting and the accuracy of predicting a single property.  Furthermore, the model only takes into account the physical aspects of the property and does have any variables about the property's design and architecture.  It is important for the client to know that this model is flexible as it was trained on both residential and commercial properties.  We ran into a problem while creating the model because the model was not accurate on prices outside of the 700k-10mil price range, which is why we subsetted the data to fit this price range.  Clients could use this model to help quickly determine what is a fair price for a property as long as it fell into the model's specialized price range.  Going forward, we would want to incorporate the latest data as we think that this data would significantly change our model due to the pandemic.  Furthermore, for residential properties, it would be important to obtain data about the house's design and architecture as that could also play a role in its price."
              )
              ),
      tabItem(tabName = "problem",
              h3("Problem Description"),
              p(
                "The Real Estate market is one of the biggest markets in the world as millions of people buy some kind of real estate each year. Many buy houses to live there and make it a home, others buy them as investments. For our project, the client would be any individual who is related to the New York City real estate market including buyers, sellers, and real estate agents. Other stakeholders could include investors in the real estate market, as they would be interested to know estimated prices for future options on the market. Another potential client is the US government, which keeps tabs on the private and public sides of the market. The main goal of this business idea is to reduce the gap between the price the buyer is willing to pay and the one that the seller is willing to sell for. This project will allow people to calculate a fair price of a property by comparing it to the ones that have been sold before and would minimize price evaluation errors. This will be based on the features that are present about each real estate transaction in our dataset. This predictive evaluation will allow those with real estate understanding an estimate that could be used on all sides of real estate. The business benefits of our proposition are showcased in the costs it would reduce when improper real estate valuation occurs. In addition, it will increase customer retention when sellers give fair estimates that result in a return on the customer's investment. It would take the patterns of the market based on this dataset to make a highly quantitative valuation. "
                ),
              p(
                "One of the major objectives of this project is to reduce the uncertainty that revolves around calculating the true price of assets. As mentioned earlier, real estate is one of the major markets in the world with more than 80 thousand holdings sold each year in New York City alone. The total market cap of the Real Estate market in New York City is around $1.378 trillion as of 2019."
                ),
              p(
                "It would be a successful project if our algorithm is able to give close to true prices around 90% of the time while testing using prior data and while beta testing as well. Despite its major promises in the field of real estate, our project has weaknesses. The price it will calculate will be based mainly on the location and the dimensions of the asset, whereas in real life, factors like architecture and interior design of buildings and homes also play a major part in determining their price."
                ),
              p(
                "We are using linear regression, lasso, and SVM models to predict real estate prices in New York. This was a supervised task because we would be training the model on data that has the price in it. We plan to have the model be trained to predict the prices of future real estate transactions. This analysis will be predictive because we are trying to predict the price of the property. The main outcome variable would be price and the main x variables will be year total land square feet, gross square feet, year built, building category, and year sold."
              )
      ),
      tabItem(tabName = "data",
              h3("Data Description"),
              p(
                "The data we will be using is the annualized sales update of properties sold in New York City, published by the NYC government. For each transaction, the data shows information about the property, such as address, building class, tax class, size, and number of residential/commercial units. It also shows the details of the sale date and amount. This data has broad information about the property itself, but it has limited information on the sale - the parties involved, or the price fluctuations of the property versus the actual sale price. A business entity is able to use this data for free since it is in the NYC public record. Updates sales information is published yearly, per borough or per type of neighborhood."
                ),
              p(
                "The dataset has purchases for 2017, 2018, and 2019. This includes information from all 5 boroughs of New York City, both residential and commercial purchases, and estate that had been passed down. There were 256,362 records in the dataset, with 25 variables that represent features of the purchase."
                ),
              p(
                "The NYC government provided a Glossary and Building Class Code Description sheet that explained each variable and showed us the classification descriptions for each Building Code. All of the features were information gathered by the NYC Housing Department"
              ),
              p(
                "To learn more about the data please visit the following link:"
              ),
              HTML('<a href="https://www1.nyc.gov/assets/finance/downloads/pdf/07pdf/glossary_rsf071607.pdf">Visit Website</a>')
      ),
      
      tabItem(tabName = "models",
    # Boxes need to be put in a row (or column)
    fluidRow(
      valueBoxOutput("value1"),
      # valueBoxOutput("value2"),
      # valueBoxOutput("value3"),
    ),
    fluidRow(
      box(plotlyOutput("distPlot")),
      
      box(
        title = "Select Type of Model",
        selectInput("variable", "Type of Model:",
                    c("Linear Regression" = "LinearModelError",
                      "LASSO" = "LassoModelError",
                      "Support Vector Machine" = "SVMmodelError",
                      "Linear Regression using Log of Sale Price" = "LogLinearModelError",
                      "LASSO using Log of Sale Price" = "LogLassoModelError",
                      "Support Vector Machine using Log of Sale Price" = "LogSVMmodelError",
                      "Average of Models" = "AveragePriceError"))
      )
    )
  ),tabItem(tabName = "Recomendations",
            h3("Conclusions and Recommendations"),
            p(
              "Our models are very helpful for predicting property prices in New York City; however, there are both advantages and limitations. Some of the advantages of our models are that they include both residential and commercial properties, so the client has access to information for both of these markets, which gives them the flexibility to enter and exit either market. Another advantage is that the models are streamlined and easy to use because they only include the factors that are most relevant and affected the price the most. Some of the limitations of our models are that they are only based on properties that are priced between $700k - 10million, which could limit the diversity of the buyers and sellers that are represented by the models as this range would typically represent the upper middle class and upper class. Another limitation is that the models only include properties from the years 2017 to 2019, so this could affect the accuracy of future predictions, especially with the effects of the COVID-19 pandemic. Lastly, the model doesn't account for factors like architecture and interior design that play a major role in determining the price."
            ),
            p(
              "Based on our models, we have a variety of operational recommendations. For instance, clients could use these models to more confidently and successfully expand into the NYC real estate market as they can serve as a guide to determine different market segments, what properties are actually worth investing in (taking advantage of properties that are being sold for less), and which market segments to focus on. In particular, the models are especially useful for the commercial real estate market as the models are most accurate for properties between $700k - 10million. These models can also assist new buyers and sellers by helping them attain fair prices for their properties, helping them be bought and sold much faster. In addition, the government could use the model to determine reasonable prices for properties in order to prevent tax evasion."
            ))
    )
  )
)


# Define server logic required to draw a histogram
server <- function(input, output){
  # x = reactive ({strsplit(gsub('([[:upper:]])', ' \\1', input$variable), " ")})
  # x=reactive({x()[[1]][-length(x())][-1]})
  # vals = reactive({paste(x(), collapse='')}) 
  output$value1 <- renderValueBox({
    valueBox(
      percent(mean(results[[input$variable]])/100)
      ,paste("Average Error")
      ,icon = icon("stats",lib='glyphicon')
      ,color = "purple")
  })
  # output$value2 <- renderValueBox({
  #   valueBox(
  #     RMSE(results[[vals]], results$sale.price)
  #     ,paste("Root Mean Square Error")
  #     ,icon = icon("stats",lib='glyphicon')
  #     ,color = "red")  
  # })
  # output$value3 <- renderValueBox({
  #   valueBox(
  #     mean(percent(results[[input$variable]]))
  #     ,paste("R-Squared Error")
  #     ,icon = icon("stats",lib='glyphicon')
  #     ,color = "green")  
  # })

    output$distPlot <- renderPlotly({
        a = ggplot(data = results)+
                     geom_point(aes(sale.price,!!as.symbol(input$variable)))+
                     labs(
                          x= "Sale Prce", y = "Percentage Error")+
                     theme_classic()
        ggplotly(a)
    })
}

# Run the application 
shinyApp(ui = ui, server = server)
