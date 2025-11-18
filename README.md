🚢 Projeto Titanic - Predição de Sobrevivência
Este projeto implementa uma solução de Machine Learning para prever a sobrevivência de passageiros do Titanic baseado em características como classe, gênero, idade e tarifa paga.

📊 Sobre o Dataset
O dataset contém informações de 891 passageiros do Titanic com as seguintes características:

Survived: Sobreviveu (0 = Não, 1 = Sim)

Pclass: Classe do ticket (1ª, 2ª, 3ª classe)

Sex: Gênero

Age: Idade

SibSp: Número de irmãos/cônjuges a bordo

Parch: Número de pais/filhos a bordo

Fare: Tarifa paga

Embarked: Porto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton)

Clone o repositório:

bash

git clone https://github.com/Thalles-JG-Silva/titanic-pi1.git

cd titanic-pii

Crie um ambiente virtual (opcional mas recomendado):

bash

python -m venv venv

source venv/bin/activate  # Linux/Mac

venv\Scripts\activate    # Windows

Instale as dependências:

bash

pip install -r requirements.txt

🚀 Como Executar

Execução Completa do Pipeline

bash

python main.py

Execução por Etapas

Análise Exploratória:

bash

jupyter notebook notebooks/01_exploracao_etl.ipynb

Treinamento dos Modelos:

bash

python src/train.py

🔧 Módulos Principais

📁 src/data_preprocessing.py

Carregamento e limpeza de dados

Tratamento de valores missing

Codificação de variáveis categóricas

📁 src/features.py
Engenharia de features:

FamilySize: Tamanho da família

IsAlone: Passageiro viaja sozinho

Title: Título extraído do nome

📁 src/models.py
Pipeline de pré-processamento

Modelos implementados:

Regressão Logística

Random Forest

📁 src/evaluate.py
Métricas de avaliação

Matriz de confusão

Curvas ROC e Precision-Recall

🤖 Modelos Implementados
Modelo	Acurácia	Précision	Recall	F1-Score
Regressão Logística	-	-	-	-
Random Forest	-	-	-	-
Nota: As métricas serão preenchidas após execução do treinamento

📈 Principais Insights
Mulheres e crianças tiveram maior taxa de sobrevivência

Passageiros da 1ª classe sobreviveram mais que os da 3ª classe

Famílias menores tiveram melhor chance de sobrevivência

Idade é um fator importante na predição

🛠️ Tecnologias Utilizadas
Python 3.8+

Pandas & NumPy: Manipulação de dados

Scikit-learn: Machine Learning

Matplotlib & Seaborn: Visualização

Jupyter: Análise exploratória

📋 Pré-requisitos
Python 3.8 ou superior

pip (gerenciador de pacotes Python)

4GB de RAM recomendados

👥 Desenvolvimento
Autor: Thalles Silva
Repositório: GitHub

📄 Licença
Este projeto é para fins educacionais. O dataset do Titanic é de domínio público.

