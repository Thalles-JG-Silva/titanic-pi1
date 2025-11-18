# 🚢 Projeto Titanic — Predição de Sobrevivência

Este projeto implementa um pipeline completo de **Machine Learning** para prever a sobrevivência de passageiros do Titanic a partir de atributos como classe, gênero, idade, tamanho da família e tarifa.
O objetivo é demonstrar um fluxo completo de análise, tratamento de dados, modelagem e avaliação.

---

## 📊 Sobre o Dataset

O dataset contém informações de **891 passageiros** e inclui as seguintes variáveis:

* **Survived** — Sobreviveu (0 = Não, 1 = Sim)
* **Pclass** — Classe do ticket (1ª, 2ª, 3ª classe)
* **Sex** — Gênero
* **Age** — Idade
* **SibSp** — Nº de irmãos/cônjuges a bordo
* **Parch** — Nº de pais/filhos a bordo
* **Fare** — Tarifa paga
* **Embarked** — Porto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton)

---

## 🛠️ Instalação e Configuração

### 1. Clone o repositório

```bash
git clone https://github.com/Thalles-JG-Silva/titanic-pi1.git
cd titanic-pi1
```

### 2. (Opcional, mas recomendado) Crie um ambiente virtual

```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

---

## 🚀 Como Executar

### ✔ Execução Completa do Pipeline

```bash
python main.py
```

### ✔ Execução por Etapas

**1. Análise Exploratória**

```bash
jupyter notebook notebooks/01_exploracao_etl.ipynb
```

**2. Treinamento dos Modelos**

```bash
python src/train.py
```

---

## 🔧 Estrutura dos Módulos

### 📁 `src/data_preprocessing.py`

* Carregamento dos dados
* Limpeza e tratamento de missing values
* Codificação de variáveis categóricas

### 📁 `src/features.py`

Engenharia de atributos:

* `FamilySize` — tamanho da família
* `IsAlone` — indica se o passageiro estava sozinho
* `Title` — título extraído do nome

### 📁 `src/models.py`

* Construção do pipeline
* Modelos implementados:

  * Regressão Logística
  * Random Forest

### 📁 `src/evaluate.py`

* Cálculo de métricas
* Matriz de confusão
* Curvas ROC e Precision–Recall

---

## 🤖 Modelos Implementados

| Modelo              | Acurácia | Precisão | Recall | F1-Score |
| ------------------- | -------- | -------- | ------ | -------- |
| Regressão Logística | —        | —        | —      | —        |
| Random Forest       | —        | —        | —      | —        |

**As métricas serão inseridas após a execução do treinamento.**

---

## 📈 Principais Insights Obtidos

* Mulheres e crianças tiveram maior taxa de sobrevivência.
* Passageiros da 1ª classe sobreviveram mais que os da 3ª classe.
* Famílias menores apresentaram melhores chances de sobrevivência.
* Idade foi um dos fatores mais relevantes para a predição.

---

## 🛠️ Tecnologias Utilizadas

* **Python 3.8+**
* **Pandas & NumPy** — manipulação de dados
* **Scikit-learn** — modelagem e avaliação
* **Matplotlib & Seaborn** — visualização
* **Jupyter Notebook** — análise exploratória

---

## 📋 Pré-requisitos

* Python 3.8 ou superior
* pip instalado
* Recomendado: 4 GB de RAM

---

## 👥 Desenvolvimento

**Autor:** Thalles Silva
**Repositório:** [GitHub](https://github.com/Thalles-JG-Silva/titanic-pi1)

---

## 📄 Licença

Este projeto é destinado a fins educacionais.
O dataset Titanic é de domínio público.