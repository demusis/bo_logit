# https://www.youtube.com/watch?v=CVL5vj1N1U8&list=WL&index=1

library(dplyr)
library(readxl)
library(tidyverse)
library(pROC)
library(rsample)  # para divisão de dados de treino e teste
library(caret)    # para validação cruzada

# setwd("~/Documentos/")

# Carrega e ajusta as variáveis
dados <- read_excel("BASE DE DADOS_RAIS 2022 TODO ESTADO.xlsx")

# Função para criar colunas binárias para cada valor único em uma coluna de texto
criar_colunas_binarias <- function(df, coluna) {
  # Verifique se existem valores NA (ausentes) na coluna
  df[[coluna]][is.na(df[[coluna]])] <- 'Ausente'
  
  # Converta a coluna para um fator
  df[[coluna]] <- as.factor(df[[coluna]])
  
  # Crie uma nova coluna para cada valor único na coluna
  df_binarios <- model.matrix(~df[[coluna]] - 1, df)
  
  # Converta a matriz de modelo em um dataframe
  df_binarios <- as.data.frame(df_binarios)
  
  # Converta 0s e 1s para inteiros (opcional, dependendo do seu caso de uso)
  df_binarios <- mutate_all(df_binarios, list(~as.integer(.)))
  
  # A matriz de modelo usa a coluna como prefixo para os nomes das colunas.
  colnames(df_binarios) <- sub("^df\\[\\[coluna\\]\\]", "", colnames(df_binarios))
  
  # Adicione essas novas colunas binárias de volta ao dataframe original
  df <- cbind(df, df_binarios)
  
  return(df)
}

# Cria coluna idade
dados$idade <- dados$`Ano do Fato` - as.numeric(dados$`Ano de Nascimento`) 

# Cria colunas binárias
dados$lesao_corporal <- as.numeric(dados$`GRUPO NATUREZA` == "Lesão corporal")
dados <- criar_colunas_binarias(dados, "Faixa Horário")
dados <- criar_colunas_binarias(dados, "Cor")
dados <- criar_colunas_binarias(dados, "MOTIVACAO")

# Seleciona variáveis
dados <- dados[, 17:50]
dados <- dados[, c(-14, -15, -18, -33)]
dados <- na.omit(dados) # Exclui linhas com NAs

# Ajustando o modelo de regressão logística para todos os dados
modelo_todos <- glm(lesao_corporal ~ ., data = dados, family = binomial("logit"))
summary(modelo_todos)
anova(modelo_todos, test="Chisq")
coef(modelo_todos) %>% exp() # odds ratio

# Pseudo R2
R2 <- 1 - modelo_todos$deviance/modelo_todos$null.deviance 
R2

# Gráfico para idade
ggplot(dados, aes(x=idade, y=lesao_corporal)) + 
  geom_point() +
  geom_smooth(method = "glm", method.args = list(family = "binomial"), se = TRUE)


## Validação cruzada
# Dividindo os dados em conjuntos de treinamento e teste
set.seed(42)  # Para reprodução de resultados
data_split <- initial_split(dados, prop = 0.7)  # 70% para treino, 30% para teste
train_data <- training(data_split)
test_data <- testing(data_split)

# Ajustando o modelo de regressão logística nos dados de treinamento
modelo_cv <- glm(lesao_corporal ~ ., data = train_data, family = binomial)
summary(modelo_cv)

# Extrair os coeficientes do modelo (log-odds)
log_odds <- coef(modelo_cv)

# Converter log-odds para odds
odds <- exp(log_odds)

# Converter os odds em um data frame
odds_df <- data.frame(Variable = names(odds), Odds = odds)

# Imprimir a tabela de odds
print(odds_df)

# Previsão nos dados de teste
preds <- predict(modelo_cv, newdata = test_data, type = "response")

# Convertendo probabilidades em previsões de classe
preds_class <- ifelse(preds > 0.5, 1, 0)

# Medindo a acurácia
vv <- preds_class == test_data$lesao_corporal
acuracia <- sum(vv) / length(vv)
print(paste("Acurácia: ", acuracia))

# Calculando a AUC (área sob a curva ROC)
roc_obj <- roc(test_data$lesao_corporal, preds)
auc(roc_obj)

# Plotando a curva ROC
plot(roc_obj, print.thres="best", print.thres.best.method="closest.topleft")

# Adicionando a área sob a curva ao gráfico
text(0.7, 0.3, paste("AUC =", round(auc(roc_obj), 2)))

# Converter a variável 'lesao_corporal' para um fator com níveis "level0" e "level1"
train_data$lesao_corporal <- as.factor(ifelse(train_data$lesao_corporal == 1, "level1", "level0"))

# Realizando validação cruzada k-fold
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, 
                     summaryFunction = twoClassSummary)
modelo_cv <- train(lesao_corporal ~ ., data = train_data, method = "glm", 
                   family = binomial, trControl = ctrl, metric = "ROC")
print(modelo_cv)

