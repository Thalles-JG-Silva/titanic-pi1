# 🚢 Projeto Titanic — Predição de Sobrevivência

Este projeto implementa um pipeline completo de **Machine Learning** para prever a sobrevivência de passageiros do Titanic utilizando atributos como classe, idade, gênero, tarifa e tamanho da família.
O objetivo é demonstrar um fluxo profissional de análise, preparação de dados, modelagem e avaliação.

---

## 📊 Sobre o Dataset

O conjunto de dados contém informações de **891 passageiros**, com variáveis relacionadas a características demográficas e sociais que influenciaram a sobrevivência durante o desastre.

Variáveis principais:

* **Survived** — Sobreviveu (0 = Não, 1 = Sim)
* **Pclass** — Classe do ticket (1ª, 2ª, 3ª classe)
* **Sex** — Gênero
* **Age** — Idade
* **SibSp** — Nº de irmãos/cônjuges
* **Parch** — Nº de pais/filhos
* **Fare** — Tarifa paga
* **Embarked** — Porto de embarque

---

## 🛠️ Instalação e Configuração

### 1. Clone o repositório

```bash
git clone https://github.com/Thalles-JG-Silva/titanic-pi1.git
cd titanic-pi1
```

### 2. (Opcional) Crie um ambiente virtual

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

### ✔ Pipeline completo

```bash
python main.py
```

### ✔ Execução por partes

**Análise Exploratória:**

```bash
jupyter notebook notebooks/01_exploracao_etl.ipynb
```

**Treinamento dos modelos:**

```bash
python src/train.py
```

---

## 🔧 Estrutura dos Módulos

### 📁 `data_preprocessing.py`

* Carregamento e limpeza de dados
* Tratamento de dados ausentes
* Codificação de variáveis

### 📁 `features.py`

Engenharia de atributos:

* `FamilySize`
* `IsAlone`
* `Title`

### 📁 `models.py`

Modelos implementados:

* Regressão Logística
* Random Forest

### 📁 `evaluate.py`

* Métricas
* Classification report
* Curvas e matrizes

---

## 🤖 Resultados dos Modelos

Após a execução do pipeline (`python main.py`), os seguintes resultados foram obtidos no conjunto de teste (179 amostras):

### **📌 Regressão Logística**

* **Acurácia:** `0.6759`
* **Resumo:**

  * Bom desempenho em prever não-sobreviventes
  * Dificuldade em capturar sobreviventes (classe 1) devido ao desbalanceamento
  * Melhor equilíbrio geral entre precisão e recall

| Métrica       | Sobreviveu=0 | Sobreviveu=1 |
| ------------- | ------------ | ------------ |
| **Precision** | 0.69         | 0.62         |
| **Recall**    | 0.85         | 0.41         |
| **F1-score**  | 0.76         | 0.49         |

---

### **🌲 Random Forest**

* **Acurácia:** `0.5977`
* **Resumo:**

  * Resultado inferior à Regressão Logística neste dataset
  * Manteve equilíbrio similar entre precision/recall nas duas classes
  * Indica que árvores precisariam de mais tuning ou mais features

| Métrica       | Sobreviveu=0 | Sobreviveu=1 |
| ------------- | ------------ | ------------ |
| **Precision** | 0.67         | 0.48         |
| **Recall**    | 0.67         | 0.48         |
| **F1-score**  | 0.67         | 0.48         |

---

### 🔍 Importância das Features (Random Forest)

A ordem de importância (parcial e obtida do log) evidencia:

* `Pclass`
* `Age`
* `Fare`
* `SibSp`
* `Parch`

> *Obs: Ao final da execução, a listagem completa de importâncias é salva no terminal.*

---

## 📈 Principais Insights

* Passageiros da **1ª classe** tinham maior chance de sobreviver.
* **Mulheres e crianças** sobreviveram mais.
* Passageiros viajando **sozinhos** tiveram menor taxa de sobrevivência.
* Idade, classe e tarifa aparecem como fatores relevantes na predição.
* Regressão Logística superou o Random Forest nesse cenário específico.

---

## 🛠️ Tecnologias Utilizadas

* **Python 3.8+**
* **Pandas & NumPy** — tratamento de dados
* **Scikit-learn** — modelos e pipelines
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
**Repositório:** [https://github.com/Thalles-JG-Silva/titanic-pi1](https://github.com/Thalles-JG-Silva/titanic-pi1)

---

## 📄 Licença

Projeto para fins educacionais utilizando dados de domínio público (Titanic Dataset).
