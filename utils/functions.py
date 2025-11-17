import os
from pathlib import Path

from openai import OpenAI, AzureOpenAI
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader


# Caminhos básicos do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[1]        # pasta chatboot_tcc
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma_langflow"

PDF_PATH = DATA_DIR / "NeuroSpark_AI_Solutions.pdf"       # renomeie se o seu PDF tiver outro nome
COLLECTION_NAME = "langflow"


def get_openai_clients():
    """
    Cria os clientes da API:
    - OpenAI padrão (para embeddings)
    - Azure OpenAI (para o agente / respostas)
    Os valores são lidos de variáveis de ambiente.
    """
    openai_client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"]
    )

    azure_client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )

    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    return openai_client, azure_client, deployment_name


# ---------- ETAPA 1: CARREGAR E QUEBRAR O PDF ----------

def load_pdf_text(pdf_path: Path) -> str:
    """Lê o PDF e concatena o texto de todas as páginas."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF não encontrado em: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages_text = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)

    full_text = "\n".join(pages_text)
    return full_text


def split_text(text: str, chunk_size: int = 800, overlap: int = 200):
    """
    Quebra o texto em pedaços (chunks) para gerar embeddings.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap

    # remove vazios
    return [c for c in chunks if c]


# ---------- ETAPA 2: CRIAR / CARREGAR O VECTOR STORE (CHROMA) ----------

def get_or_create_collection(openai_client: OpenAI):
    """
    Cria (ou carrega) a coleção do Chroma DB.
    Se a coleção estiver vazia, gera os embeddings a partir do PDF.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(allow_reset=True)
    )

    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        print("🔄 Coleção vazia. Gerando embeddings a partir do PDF...")
        text = load_pdf_text(PDF_PATH)
        chunks = split_text(text)

        # gera embeddings em lote
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=chunks,
        )

        embeddings = [item.embedding for item in response.data]
        ids = [f"doc-{i}" for i in range(len(chunks))]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
        )

        print(f"✅ Coleção criada com {len(chunks)} chunks.")
    else:
        print(f"✅ Coleção '{COLLECTION_NAME}' carregada com {collection.count()} itens.")

    return collection


# ---------- ETAPA 3: BUSCA NO VECTOR STORE ----------

def retrieve_context(query: str, collection, openai_client: OpenAI, k: int = 4) -> str:
    """
    Gera o embedding da pergunta e busca os documentos mais relevantes no Chroma.
    """
    query_emb = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[query],
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=k,
    )

    docs = results["documents"][0] if results["documents"] else []
    context = "\n\n---\n\n".join(docs)
    return context


# ---------- ETAPA 4: CHAMADA AO AGENTE (AZURE OPENAI) ----------

def build_system_prompt() -> str:
    """
    Prompt do agente, equivalente ao 'Agent Instructions' do LangFlow.
    Ajuste o texto para ficar igual ao que você configurou lá.
    """
    return (
        "Você é o Neo, o assistente virtual da NeuroSpark. "
        "Ajude o usuário a entender as soluções de IA da empresa, "
        "explicando de forma clara, objetiva e em português do Brasil. "
        "Use apenas o contexto fornecido. "
        "Se não tiver informação suficiente, diga que não sabe em vez de inventar."
    )


def ask_neo(question: str) -> str:
    """
    Função principal:
    - carrega/cria o vector store
    - busca contexto
    - chama o modelo GPT-4o via Azure OpenAI
    """
    openai_client, azure_client, deployment_name = get_openai_clients()
    collection = get_or_create_collection(openai_client)

    context = retrieve_context(question, collection, openai_client, k=4)

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {
            "role": "user",
            "content": (
                f"Pergunta do usuário:\n{question}\n\n"
                f"Contexto para responder (conteúdo do PDF):\n{context}"
            ),
        },
    ]

    response = azure_client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()
