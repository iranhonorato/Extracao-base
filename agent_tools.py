from dotenv import dotenv_values
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import tool
from enum import Enum
from pydantic import Field, BaseModel
from typing import List

# Carregando variáveis de ambiente 
config = dotenv_values(".env")

pinecone_api_key = config["PINECONE_API_KEY"]
pinecone_env = config["PINECONE_ENVIRONMENT"]
openai_api_key = config["OPENAI_API_KEY"]




# Instanciando nosso vector database
embeddings_model = OpenAIEmbeddings(
    api_key=openai_api_key, 
    model="text-embedding-3-small"
)

vector_store_livreto = PineconeVectorStore(
    index_name="livreto-base-evidencia",
    embedding=embeddings_model,
    pinecone_api_key=pinecone_api_key,
    namespace="livreto-metricis"
)

vector_store_ods = PineconeVectorStore(
    index_name="ods-onu",
    embedding=embeddings_model,
    pinecone_api_key=pinecone_api_key,
    namespace="catalogo-ods"
)   


retriever_livreto = vector_store_livreto.as_retriever(
    search_kwargs={
        "namespace": "livreto-metricis", 
        "k": 3
        }
    ) 


retriever_ods = vector_store_ods.as_retriever(
    search_kwargs={
        "namespace": "catalogo-ods",
        "k": 2
        }
    )



# Definindo as tools de consulta no nosso vector database  
tool_livreto = create_retriever_tool(
    retriever_livreto,
    "busca_gestao_publica",
    "Útil para identificar qual etapa do ciclo de políticas públicas se relaciona com um texto ou projeto."
)

tool_ods = create_retriever_tool(
    retriever_ods,
    "busca_ods_onu",
    "Útil para identificar quais Objetivos de Desenvolvimento Sustentável (ODS) e metas da ONU se relacionam com um texto ou projeto."
)


# Definindo uma tool para um output estruturado
class EtapaCicloPP(Enum):
    DEFINICAO = "Definição e Dimensão"
    MOBILIZACAO = "Mobilização"
    MAPEAMENTO = "Mapeamento dos Determinantes"
    SOLUCAO = "Solução"
    JUSTIFICATIVA = "Justificativa"
    APRIMORAMENTO = "Aprimoramento"
    CERTIFICACAO = "Certificação"

class Classificacao(Enum):
    ACADEMICA = "Acadêmica"
    TECNICA = "Técnica"

class AreaAvaliada(Enum):
    EDUCACAO = "Educação"
    SAUDE = "Saúde"
    MEIO_AMBIENTE = "Meio Ambiente"
    GENERO = "Gênero"
    RACA = "Raça"
    POBREZA = "Pobreza"
    DESENVOLVIMENTO_SOCIAL = "Desenvolvimento Social"


class Metodologia(Enum):
    QUALITATIVA = "Qualitativa"
    QUANTITATIVA = "Quantitativa"
    MISTA = "Mista"


class Trabalho(BaseModel):
    classificacao: Classificacao = Field(..., description="Acadêmico ou técnico")
    metodologia: Metodologia = Field(..., description="Abordagem do trabalho")
    area_avaliada: AreaAvaliada = Field(..., description="Área avaliada no trabalho")
    ods: List[int] = Field(..., description="Número das ODS relacionadas ao trabalho (texto)")
    etapa: EtapaCicloPP = Field(..., description="Etapa do ciclo de políticas públicas")
    titulo:str = Field(..., description="Titulo do trabalho")


@tool
def formatar_resposta_final(output:Trabalho):
    """
    Usa esta ferramenta por último para entregar a análise final estruturada. 
    Esta ferramenta valida todos os Enums e campos obrigatórios.
    """
    return output

# lista de tools
tools = [tool_livreto, tool_ods, formatar_resposta_final]

print("Tools criadas com sucesso")