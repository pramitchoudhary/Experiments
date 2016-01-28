library(shiny)

diabetesRisk <- function(glucose) glucose / 200
# Define server logic required to plot various variables against mpg
shinyServer(function(input, output) {
    output$inputValue <- renderPrint({input$glucose})
    output$prediction <- renderPrint(diabetesRisk(glucose = {input$glucose}))
})