Título: Predição de Sobrevivência no Titanic – Projeto Individual (PI1)

Descrição Geral:
Este projeto tem como objetivo construir um modelo de Machine Learning capaz de prever quais passageiros sobreviveram ao naufrágio do Titanic, utilizando o dataset clássico disponibilizado pelo Kaggle.
O trabalho envolve exploração de dados, limpeza, engenharia de atributos, treinamento, avaliação e criação de um pipeline estruturado.

Seções do Projeto

Análise Exploratória (EDA)
A primeira etapa consiste em analisar os dados para identificar padrões, tendências e possíveis problemas.
Entre as visualizações e descobertas incluem-se:
• Distribuição das idades
• Sobrevivência por sexo e por classe social
• Relação entre tarifa, idade e sobrevivência
• Mapa de correlação entre variáveis
• Identificação de dados ausentes

Pré-Processamento dos Dados
O conjunto de dados passa por um tratamento completo:
• Remoção das colunas que não contribuem para o modelo (Cabin, Name, Ticket, PassengerId)
• Preenchimento de valores nulos
• Conversão de variáveis categóricas em numéricas
• Separação dos dados em treino e teste

Engenharia de Atributos
Novas variáveis são criadas para enriquecer o modelo:
• FamilySize (tamanho da família)
• IsAlone (viaja sozinho ou não)
• Extração dos títulos (Mr, Miss, Mrs, Master, Rare) a partir da coluna Name
• Agrupamento de faixas de idade
• Normalização e padronização quando necessário

Modelos Utilizados
Dois algoritmos principais foram treinados:
• Regressão Logística
• Random Forest
Ambos integrados em um pipeline automatizado que inclui pré-processamento, codificação categórica e padronização numérica.

Avaliação dos Modelos
Os modelos passam por uma análise completa de desempenho:
• Matriz de confusão
• Acurácia, precisão, recall e F1-score
• Curva ROC e cálculo da área AUC
• Curva Precision-Recall
• Importância das features (Random Forest)

Estrutura do Projeto (reescrita para ficar perfeita no GitHub)

titanic-pi1
│
├── data
│ Contém o arquivo train.csv utilizado no projeto.
│
├── notebooks
│ 01_exploracao_etl.ipynb – Notebook com toda a análise exploratória.
│
├── src
│ data_preprocessing.py – Funções de limpeza e preparação dos dados
│ features.py – Criação de novas variáveis e transformações
│ models.py – Configuração e criação dos modelos
│ train.py – Execução do treinamento dos modelos
│ evaluate.py – Funções de avaliação e métricas
│ visualization.py – Gráficos e visualizações
│
├── models
│ logistic_regression.joblib – Modelo treinado
│ random_forest.joblib – Modelo treinado
│
├── main.py
│ Script principal que executa o pipeline completo.
│
├── requirements.txt
│ Lista de dependências necessárias.
│
└── README.md
Arquivo de documentação do projeto.

Como Executar o Projeto

Instale todas as dependências:
pip install -r requirements.txt

Execute o pipeline completo:
python main.py

Para usar o notebook:
Abra o arquivo 01_exploracao_etl.ipynb utilizando Jupyter Notebook ou Jupyter Lab.

Tecnologias Aplicadas
• Python
• Pandas
• NumPy
• Scikit-Learn
• Matplotlib
• Seaborn
• Joblib

Resultado Final
O projeto gera dois modelos de machine learning treinados e avaliados, permitindo entender quais fatores mais contribuíram para a sobrevivência no Titanic e fornecendo uma base sólida para estudos, portfólio ou evolução para uma API no futuro.