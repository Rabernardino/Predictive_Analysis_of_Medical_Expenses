# Medical Expenses Prediction

This repository contains the project developed to predict medical expenses using linear regression techniques. The primary objective was to create a reliable predictive model, validated by statistical analyses.

## Project Description

The dataset used in this project consists of the following variables:

- **id**: Unique identification for each record.
- **medical_expenses**: User's medical expenses.
- **age**: User's age.
- **user_group**: Category to which the user belongs.
- **income**: User's income.
- **plan_type**: Type of health plan the user has.

### Model Development

Throughout the project, three linear regression models were developed, with progressive adjustments based on statistical analyses:

1. **Model 1**: Initial version of the linear regression model.
2. **Model 2**: Adjusted model after residual analysis.
3. **Model 3**: Final model with complete corrections.

### Statistical Tests Conducted

Statistical tests were applied to ensure the adequacy of the models:

- **Normality Test (Shapiro-Francia)**: Checked for the normality of the model's residuals.
- **Heteroscedasticity Test (Breusch-Pagan)**: Assessed the presence of heteroscedasticity in the residuals.

### Final Result

After corrections and adjustments, **Model 3** was selected as the final model, being the most suitable for predicting medical expenses.

## Repository Structure

- **data/**: Contains the dataset used.
- **notebooks/**: Jupyter notebooks detailing the analysis and modeling process.
- **models/**: Versions of the generated models.
- **tests/**: Scripts for running statistical tests.
- **results/**: Results of the analyses and the final model.

## Conclusion

This project illustrates the complete process of developing a linear regression model for predicting medical expenses, from exploratory analysis to statistical validation of the models.

---

# Análise Preditiva de Despesas Médicas

Este repositório contém o projeto desenvolvido para prever os custos com despesas médicas utilizando técnicas de regressão linear. O objetivo principal foi criar um modelo preditivo confiável, validado por análises estatísticas.

## Descrição do Projeto

A base de dados utilizada neste projeto é composta pelas seguintes variáveis:

- **id**: Identificação única de cada registro.
- **despesas_medicas**: Valor das despesas médicas do usuário.
- **idade**: Idade do usuário.
- **grupo_do_usuario**: Categoria à qual o usuário pertence.
- **renda**: Renda do usuário.
- **tipo_de_plano**: Tipo de plano de saúde do usuário.

### Desenvolvimento dos Modelos

Ao longo do projeto, foram desenvolvidos três modelos de regressão linear, com ajustes progressivos baseados em análises estatísticas:

1. **Modelo 1**: Primeira versão do modelo de regressão linear.
2. **Modelo 2**: Modelo ajustado após análise dos resíduos.
3. **Modelo 3**: Modelo final com correções completas.

### Testes Estatísticos Realizados

Foram aplicados testes estatísticos para garantir a adequação dos modelos:

- **Teste de Normalidade (Shapiro-Francia)**: Verificação da normalidade dos resíduos do modelo.
- **Teste de Heterocedasticidade (Breusch-Pagan)**: Avaliação da presença de heterocedasticidade nos resíduos.

### Resultado Final

Após as correções e ajustes necessários, o **Modelo 3** foi selecionado como o modelo final, sendo o mais adequado para prever as despesas médicas com base nas variáveis disponíveis.

## Estrutura do Repositório

- **data/**: Contém a base de dados utilizada.
- **notebooks/**: Notebooks Jupyter detalhando o processo de análise e modelagem.
- **models/**: Versões dos modelos gerados.
- **tests/**: Scripts para execução dos testes estatísticos.
- **results/**: Resultados das análises e do modelo final.

## Conclusão

Este projeto ilustra o processo completo de desenvolvimento de um modelo de regressão linear para previsão de despesas médicas, desde a análise exploratória até a validação estatística dos modelos.
