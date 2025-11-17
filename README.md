# 🤖 Chatbot Inteligente com Técnicas de PLN  
### *Projeto do TCC — Direcionamento Automático de Leads*

Este repositório contém o código-fonte, documentação e arquitetura do meu Trabalho de Conclusão de Curso, cujo objetivo foi desenvolver um **chatbot inteligente**, baseado em **Processamento de Linguagem Natural (PLN)**, capaz de **redirecionar leads automaticamente para os setores apropriados**, aumentando eficiência, velocidade de resposta e qualidade no atendimento.

---

## 📌 Objetivo do Projeto

Criar um sistema capaz de:

- Interpretar mensagens de usuários usando PLN  
- Classificar intenções e categorias de atendimento  
- Direcionar leads automaticamente  
- Manter rastreabilidade das conversas  
- Operar com baixo custo e alta escalabilidade  
- Ser simples de integrar a sistemas existentes

---

## 🧠 Arquitetura da Solução

A inteligência da aplicação foi planejada e prototipada inicialmente no **LangFlow**, utilizando:

- Embeddings  
- Classificação por Similaridade  
- Prompt Nodes  
- Memory Buffer  
- RAG simplificado  
- Lógica de decisão por nós encadeados  

Abaixo está o **fluxograma oficial** extraído do protótipo no LangFlow:

![Fluxograma LangFlow](docs/fluxo_langflow.png)

---

## 🗂️ Estrutura do Projeto
📦 chatboot_tcc/
├── src/
│ ├── main.py
│ └── utils/
│ └── functions.py
├── models/
│ └── intents.json
├── data/
│ └── samples/
├── docs/
│ ├── fluxo_langflow.png
│ └── arquitetura.md
├── requirements.txt
└── README.md

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.10**
- **FastAPI** (API do chatbot)
- **LangFlow** (protótipo de fluxo)
- **OpenAI / HuggingFace Embeddings**
- **RAG simplificado**
- **Classificação de intenção**
- **WSL + VSCode** para ambiente profissional
- **Git + GitHub** para versionamento

---

## ▶️ Como executar o projeto

### 1. Criar e ativar o ambiente virtual
```bash
python3 -m venv .venv
source .venv/bin/activate

✨ Autora

Paula Fregatto
Especialista em IA, Vendas em TI e Projetos de Inovação.
Conecte-se comigo no LinkedIn:
🔗 https://www.linkedin.com/in/paulaeflima/