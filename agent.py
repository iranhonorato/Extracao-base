from dotenv import dotenv_values
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from enum import Enum
from pydantic import Field, BaseModel
from typing import List

# Carregando variáveis de ambiente 
config = dotenv_values(".env")
openai_api_key = config["OPENAI_API_KEY"]



# Função do agente
def agente_classificador(query:str, tools:list) -> dict:
    # 1. O system_text: É a "Constituição" do Agente. Aqui você define o Codebook.
    #  Você dá à IA uma "personalidade", nesse caso a personlidade de um pesquisador sênior. 
    system_text = """
        Você é um pesquisador sênior especializado em revisão sistemática de políticas públicas e 
        nos Objetivos de Desenvolvimento Sustentável (ODS) da ONU. 
        Sua tarefa é codificar estudos extraindo dados sistematicamente conforme os critérios abaixo.

        ### CRITÉRIOS DE CODIFICAÇÃO (CODEBOOK)

        1. CLASSIFICAÇÃO:
        - "Academica": Foco em rigor metodológico, fundamentação teórica e revisão de literatura.
        - "Tecnica": Foco em execução, procedimentos operacionais e resultados práticos/gestão.

        2. METODOLOGIA:
        - "Qualitativa", "Quantitativa" ou "Mista".

        3. ÁREA AVALIADA:
        - Escolha uma: "Educação", "Saúde", "Meio Ambiente", "Gênero", "Raça", "Pobreza" ou "Desenvolvimento Social".

        4. ETAPAS DO CICLO DE POLÍTICAS PÚBLICAS (Base: Insper/IAS 2019):
        - "Definição e Dimensão": Especificação, mensuração e consequências do problema.
        - "Mobilização": Sensibilização, percepção de atores-chave e eficácia da mobilização.
        - "Mapeamento dos Determinantes": Identificação e priorização de causas modificáveis.
        - "Solução": Estratégia, modelo de mudança, validade das hipóteses e metas.
        - "Justificativa": Descrição de impacto, valoração de custos/benefícios e custo-efetividade.
        - "Aprimoramento": Monitoramento, eficiência, eficácia alocativa e validação de implementação.
        - "Certificação": Estimativas finais de impacto, custo final, adequação e resolutividade.

        5. ODS (Objetivos de Desenvolvimento Sustentável):
        - Identifique as ODS (1 a 17) mais relacionadas à temática central.
        - Exemplo de formato: "1; 5; 10" (Refere-se à: ODS 1, ODS 5 e ODS 10, respectivamente). 

        ### TAREFA
        Sempre que receber um texto, analise-o profundamente contra os critérios acima e 
        responda **APENAS** com um objeto JSON válido. Não inclua textos explicativos fora do JSON.

        ### FORMATO DE SAÍDA (OBRIGATÓRIO)
        {{
            "titulo": "Título original do trabalho",
            "classificacao": "Academica ou Tecnica",
            "metodologia": "Qualitativa, Quantitativa ou Mista",
            "area_avaliada": "Área correspondente",
            "etapa": "Nome da etapa conforme o Codebook",
            "ods": "Lista de ODS (Ex: 3; 4)",
            "resumo_justificativa": "Breve frase justificando a etapa do ciclo escolhida e as ODS relacionadas"
        }}
    """
    # Note o uso de chaves duplas {{ }} no formato de saída; 
    # isso é necessário porque o Python entende chaves únicas como variáveis, 
    # então "escapamos" elas para que o Agente entenda que é um texto de estrutura JSON.


    # 2. Definimos o modelo (GPT-4o é o melhor para raciocínio técnico)
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_api_key)


    # 3. Criamos o "Manual de Instruções" do Agente (Prompt)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text), # Injeta as regras de pesquisador sênior.
        ("human", "{input}"), # É onde o texto do seu trabalho entrará.
        MessagesPlaceholder(variable_name="agent_scratchpad"), # Esta é a linha mais "mágica". 
        # O scratchpad (bloco de notas) é um espaço dinâmico onde 
        # o LangChain armazena o histórico de quais ferramentas o agente já usou e 
        # o que ele descobriu nelas antes de te dar a resposta final.
    ])



    # 4. Construímos de fato o Agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    # Esta linha "cola" as três partes: 
    # - o cérebro (llm) 
    # - as mãos (tools que definimos antes, como o retriever e a formatação) 
    # - as instruções (prompt). Ela cria o raciocínio lógico que sabe quando chamar uma função externa.

    # 5. Criamos o Executor: O Agente em si é apenas uma lógica; o Executor é quem realmente roda o loop "Pensar -> Agir -> Observar".
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, # com verbose=True para você ver ele 'pensando'
        return_intermediate_steps=False # Queremos apenas o resultado final
    )

    # 6. Execução: O dicionário {"input": query} preenche aquele campo {input} que definimos no prompt lá atrás.
    resposta = agent_executor.invoke({"input": query}) 

    return resposta["output"]