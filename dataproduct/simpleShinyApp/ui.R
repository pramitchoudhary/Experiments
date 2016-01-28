library(shiny)

shinyUI(pageWithSidebar(
  
  # Application title
  headerPanel("Prediction"),
  
  sidebarPanel(
    numericInput('glucose', 'Glucose mg/dl', 90, min = 50, max = 200, step = 5),
    submitButton('Submit')
  ),
  
  mainPanel(
    h3('Result of the Prediction'),
    h4('Input'),
    verbatimTextOutput("inputValue"),
    h4('Prediction Output'),
    verbatimTextOutput("prediction")
  )
))
