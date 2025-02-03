#wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
#dpkg -i google-chrome-stable_current_amd64.deb 
# if you get error run in above line, run below 2 lines
#apt-get install -f
#dpkg -i google-chrome-stable_current_amd64.deb
#pip install --upgrade pymupdf
#Download and install 2.46.0 for Windows 10 (64-bit): stable_windows_10_cmake_Release_x64_graphviz-install-2.46.0-win64.exe
#python -m pip install --config-settings="--global-option=build_ext" --config-settings="--global-option=-IC:\Program Files\Graphviz\include" --config-settings="--global-option=-LC:\Program Files\Graphviz\lib" pygraphviz
#pip install fastapi==0.112.2
#pip install fpdf markdown2 
# update lightrag.py, comment out the line embedding_func=self.embedding_func in self.chunk_entity_relation_graph = self.graph_storage_cls
#metapub\pubmedfetcher.py added a print stmt        print(f'pmids_for_query: {query}. max_pmid_to_get: {retmax}.').
#updated neo4j_impl.py to remove embedding_func and add print stmt
# prompt.py - added llm's rag resp instruct to also send the citation. llm.py - added prinr stmt to see what is being sent to llm by lightrag.
# copy updated pubmedfetcher.py and pubmedarticle.py and to metapub folder
import gradio as gr
#import streamlit as st
import PyPDF2, uuid, tiktoken, time, datetime, functools, markdown, operator, requests, os, json, shutil
#fitz, 
from datetime import date, timedelta
from bs4 import BeautifulSoup
from typing import Annotated, Sequence, TypedDict, Literal
from typing_extensions import TypedDict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from fpdf import FPDF
from functools import reduce

# below needed to get articles from NCIB for PubMed.
from metapub import PubMedFetcher

# below needed to take sanpshot of a webpage
from selenium import webdriver
#from selenium.webdriver.chrome.options import Options

# below needed to resize thumbnail
from PIL import Image, ImageFilter

# below needed to use nano-vectorDB and graphDB neo4j
from lightrag import LightRAG, QueryParam
from lightrag.llm import gemini_complete, gpt_4o_mini_complete, gpt_4o_complete, bedrock_complete

# below needed to create multi-agents
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.managed.is_last_step import RemainingSteps

# below needed to draw agent interaction graph
from langchain_core.runnables.graph import MermaidDrawMethod 

# below needed to work with LLMs
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    HumanMessage,
    AIMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq

#export OPENAI_API_KEY=xxxxxxxxxx
#export LANGCHAIN_API_KEY=xxxxxxxxxx
#serper_api_key = os.environ["SERPAPI_API_KEY"]
serper_api_key = "aa"

max_out_tokens=4096
max_llm_tokens=8192
recursion_limit = 200
llm_char_limit1 = 30000
llm_char_limit2 = llm_char_limit1 // 2
llm_char_limit3 = 0 - llm_char_limit2

llm_options = ["claude-haiku", "gemini-flash", "gpt-4o-mini", "gpt-4o"]
llm_func_options = [bedrock_complete, gemini_complete, gpt_4o_mini_complete, gpt_4o_complete]

pubmed_base_url = f"https://pubmed.ncbi.nlm.nih.gov"
RAG_DIR = "/home/ubuntu/multi-agent-ai-langgraph/rag"
filePath = "/home/ubuntu/multi-agent-ai-langgraph/docs/"
pngPath = "/home/ubuntu/multi-agent-ai-langgraph/img"
currDate = datetime.datetime.now().strftime("%Y-%m-%d")
currYr = datetime.datetime.now().strftime("%Y")
#    for NEXT gradio version 5
#filePathPre = f"/gradio_api/file" 
filePathPre = f"/file"

max_in_tokens = max_llm_tokens - max_out_tokens
#anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')
os.environ["LANGCHAIN_PROJECT"] = "LangGraph Research Agents"

# 1) Tool for search
@tool("google_search", return_direct=False)
def google_search(query: str) -> str:
    """Searches Google using Serper"""
#    time.sleep(2.0)
    print("Running Google Serper: " + query)
    results = None
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }
    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        text = response.text
#        print("CONTENT:", text)
        if len(text) > llm_char_limit1:
            results = text[:llm_char_limit2] + text[llm_char_limit3:]
        else:
            results = text
    except BaseException as e:
        error = repr(e)
        print("Error: " + error)
        return f"Failed to execute. Error: {error}"
#    print("******************************* \n" + results)
    return results if results else "No results found."

@tool("tavily_tool", return_direct=False)
def tavily_tool(query: str) -> str:
    """Searches the internet using Tavily"""
#    time.sleep(2.0)
    print("Running Tavily: " + query)
    results = None
    try:
        search = TavilySearchResults(max_results=2)
        context = search.invoke(query)
        results = ""
        for i in context:
            results += f"{i['content']}.\n"    
    except BaseException as e:
        error = repr(e)
        print("Error: " + error)
        return f"Failed to execute. Error: {error}"
#    print("******************************* \n" + results)
    print("----------")
    print("DONE.")
    print("----------")
    return results if results else "No results found."

@tool("process_content", return_direct=False)
def process_content(objective: str, url: str) -> str:
    """Processes content from a webpage."""
#    time.sleep(2.0)
    print("Internet search via url: " + url)
    try:
        results = None
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text()
#        print("CONTENT:", text)
            if len(text) > llm_char_limit1:
                results = text[:llm_char_limit2] + text[llm_char_limit3:]
            else:
                results = text
#                output = summary(objective, text)
#                results = output
#                results = text
#            else:
#                results = text
        else:
            results = f"HTTP request failed with status code {response.status_code}"
    except BaseException as e:
        error = repr(e)
        print("Error: " + error)
        results = f"Failed to execute. Error: {error}"
#    print("******************************* \n" + results)
    return results if results else "No results found."

# Warning: This executes code locally, which can be unsafe when not sandboxed

repl = PythonREPL()

@tool("python_repl", return_direct=False)
def python_repl(code: Annotated[str, "The python code to execute to generate your files."]) -> str:
    
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    print("Running the code.")
    print("\n"+code)
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    print("----------")
    print("DONE.")
    print("----------")
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )

@tool("create_md_content", return_direct=False)
def create_md_content(subArea: str, keyword: str, Duration: str, entities: list[str], pngPath: str, hmFile: str) -> str:
    """create_md_content."""
    """Supply subArea as: Therapeutic SubArea"""
    """Supply keyword as: User supplied keyword"""
    """ entities as: ["Biomarkers","Drugs","Genes"]...."""
    """Supply Duration as: 1 year, 10 years, 5 months, 1 week etc..."""
    """Supply full filePath for pngPath as: /home/ubuntu/multi-agent-ai-langgraph/img"""
    """Supply FileName for hmFile as: Breast Cancer-heatmap.png etc... If you want to see the output of a value, you should print it out with `print(...)`. This is visible to the user."""

    print("----------")
    print("running create_md_content.")
    print("----------")
#    currMonDate = datetime.datetime.now().strftime("%m/%d")
#    yr_since=int(currYr)
#    today = date.today()
    if (subArea and keyword and Duration and entities and pngPath and hmFile):
        pass
    else:
        return "Success."
#    print(entities)
    num_of_articles=2
    past_date = ""
    entity_list = ' AND '.join(str(x) for x in entities)
    entities = entity_list
    print(f'create_md_content received: subArea={subArea}, Duration={Duration}, keyword={keyword}, entities={entities}')
    if "year" in Duration:
        yr_past = Duration.replace(" years","").replace(" year","").replace("last ","")
        date_past = date.today() - timedelta(days= (int(yr_past) * 365))

    if "month" in Duration:
        mon_past = Duration.replace(" months","").replace(" month","").replace("last ","")
        date_past = date.today() - timedelta(days= (int(mon_past) * 30))

    if "week" in Duration:
        week_past = Duration.replace(" weeks","").replace(" week","").replace("last ","")
        date_past = date.today() - timedelta(days= (int(week_past) * 7))

    past_date = date_past.strftime("%Y/%m/%d")        
#    print(past_date)

    df = get_pubmed_articles(subArea, keyword, num_of_articles, entities, past_date)
#print(df)
    content = ""
    content = f"#This document is produced by the multi-agentic research agents \n\nCreated on: {currDate}\n\n# Overview\nThis document summarizes upto {num_of_articles} relevant research papers in {subArea} on the topic of {keyword}, focusing on {entities}.\n\n"
    if len(df) == 0:
        content += f'Research result came back empty. Re-submit your query with less specific keyword or fewer entities or a wider search Duration.\n'
        return content

    csvFile = f'{filePath}{subArea}-{keyword}-found-papers.csv'

    if os.path.exists(csvFile):
        print(f"deleting old article csvFile: {csvFile}.")
        os.remove(csvFile)
    else: 
        print("old article csvFile doesnt exist. nothing to delete.")
    print(f"creating new article csvFile: {csvFile}.")
    df.to_csv(csvFile)
    paperCnt = 0
#    print(f'{filePathPre}. {hmFile}')
    for row in df.itertuples():
#        print(f"working on paper: {row.Link}) 
        time.sleep(0.1) # delay avoids taxing the NCBI's PubMed site.
        paperCnt = paperCnt + 1
        pngFile = create_url_snapshot(pubmed_base_url, subArea, row.pmid, paperCnt, pngPath)
        content += f"## Paper {paperCnt}:\n![Paper{paperCnt} Thumbnail]({filePathPre}={pngFile})\n\n## Section 1: \n- **Title:** {row.Title}  \n- **Author Name:** {row.Author}  \n- **Journal Name:** {row.Journal}  \n- **Publication Volume:** {row.Volume}  \n- **Published Date:** {row.PublishedDate}  \n- **Pub Med Id:** {row.pmid}  \n- **Source URL:** [Link to Paper]({row.Link})  \n\n## Section 2:  \nAbstract:\n{row.Abstract}  \n\n"
#    print(f'{content}')
    content += f"![Heatmap Thumbnail]({filePathPre}={pngPath}/{hmFile})\n\n## Summary:  \nThis document provides the latest research obtained from all the major online medical publications, based on user input.\n\n"
#    print(content)
    return content

@tool("create_all_files", return_direct=False)       
def create_all_files(content: str, entities: list[str], tripleList: list[str], subArea: str, csvFile: str, hmFile: str, mdFile: str) -> str:
    """create_all_files."""
    """supply content as string"""
    """supply entities as ['Biomarkers','Drugs','Genes','Pathogens','Proteins','Tissue Cell Types']"""
    """supply tripleList as ["T2DM->has->Biomarkers", "T2DM->is associated with->Acute Pancreatitis", "T1DM->has Differentially Expressed->Genes"]"""
    """supply subArea as string"""    
    """Supply full filepaths for all files, for csvFile like: {filePath}/{subArea}-csvfile.csv and for hmFile like: {pngPath}/{subArea}-heatmap.png etc... If you want to see the output of a value, you should print it out with `print(...)`. This is visible to the user."""

    print("----------")
    print("running create_all_files.")
    print("----------")
    print(f"content: {content}")
    if (content and subArea and csvFile and hmFile and mdFile):
        pass
    else:
        return "Success."
        
    for entity in tripleList:
        print(f"triple: {entity}")
        if "->" in entity:
            pass
        else:
            return "Success."

    create_md(content, mdFile)
    try:
        create_er_hm(tripleList, entities, subArea, csvFile, hmFile)
    except:
        print(f"There was some issue creating Enitiy-Relation csv, or heatmap. Continuing....")
    print("----------")
    print("all_files completed.")
    print("----------")
    return "Success."
 
def get_pubmed_articles(subArea, keyword, num_of_articles, entities, yr_since):
    keywords = f'"{subArea}" AND "{keyword}" AND {entities} [TIAB]'

    fetch = PubMedFetcher()
    pmids = fetch.pmids_for_query(keywords, since=yr_since, retmax=num_of_articles)
    print(f"found {len(pmids)} relevant articles.")
    articles = {}
# Retrieve information for each article
    for pmid in pmids:
        articles[pmid] = fetch.article_by_pmid(pmid)

# Extract relevant information
    titles = {}
    for pmid in pmids:
        titles[pmid] = fetch.article_by_pmid(pmid).title
    Title = pd.DataFrame(list(titles.items()), columns=['pmid', 'Title'])

    authors = {}
    for pmid in pmids:
        authors[pmid] = fetch.article_by_pmid(pmid).authors
    Author = pd.DataFrame(list(authors.items()), columns=['pmid', 'Author'])

    pubdates = {}
    for pmid in pmids:
        pubdates[pmid] = fetch.article_by_pmid(pmid).year
    PublishedDate = pd.DataFrame(list(pubdates.items()), columns=['pmid', 'PublishedDate'])

    journals = {}
    for pmid in pmids:
        journals[pmid] = fetch.article_by_pmid(pmid).journal
    Journal = pd.DataFrame(list(journals.items()), columns=['pmid', 'Journal'])

    volumes = {}
    for pmid in pmids:
        volumes[pmid] = fetch.article_by_pmid(pmid).volume
    Volume = pd.DataFrame(list(volumes.items()), columns=['pmid', 'Volume'])

    links = {}
    for pmid in pmids:
        links[pmid] = f"{pubmed_base_url}/{pmid}"
    Link = pd.DataFrame(list(links.items()), columns=['pmid', 'Link'])

    abstracts = {}
    for pmid in pmids:
        abstracts[pmid] = fetch.article_by_pmid(pmid).abstract
    Abstract = pd.DataFrame(list(abstracts.items()), columns=['pmid', 'Abstract'])

# Merge all DataFrames into a single one
    data_frames = [Title, PublishedDate, Author, Journal, Volume, Link, Abstract]

    df_merged = reduce(lambda  left, right: pd.merge(left, right, on=['pmid'], how='outer'), data_frames)

    df_sorted = df_merged.sort_values(by='PublishedDate', ascending=False)

    return df_sorted

def create_url_snapshot(url, subArea, pubMedId, paperCnt, pngPath):
    tmpPngFile = f'{pngPath}/{subArea}_{pubMedId}-Paper{paperCnt}-tmp_thumbnail.png'
    pngFile = f'{pngPath}/{subArea}_{pubMedId}-Paper{paperCnt}-thumbnail.png'
    if os.path.exists(tmpPngFile):
        print(f"deleting old tmp snapshot file: {tmpPngFile}.")
        os.remove(tmpPngFile)
    else:
        print(f"tmp snapshot file {tmpPngFile} not exists. nothing to delete")

    if os.path.exists(pngFile):
        print(f"skipped creating snapshot file: {pngFile}. It already exists.")
    else:
        print(f"creating new tmp snapshot file: {tmpPngFile}.")
        user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.50 Safari/537.36'
        options = webdriver.ChromeOptions()
        options.add_argument(f'user-agent={user_agent}')
        options.add_argument('--no-sandbox')
        options.add_argument('--incognito')
        options.add_argument('--headless')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument("--disable-blink-features=AutomationControlled")
        driver = webdriver.Chrome(options=options)
        driver.get(f"{url}/{pubMedId}")
        driver.save_screenshot(tmpPngFile)
        driver.quit()
        img = Image.open(tmpPngFile)
#img.thumbnail((150 ,150))
        resize = img.resize((200, 200))
        print(f"creating re-sized snapshot file: {pngFile}.")
        resize.save(pngFile)
        if os.path.exists(tmpPngFile):
            print(f"deleting old tmp snapshot file: {tmpPngFile}.")
            os.remove(tmpPngFile)
        else:
            print(f"tmp snapshot file {tmpPngFile} not exists. nothing to delete")

    return pngFile
 
#Create er, kg, hm files
def create_er_hm(tripleList, entities, subArea, csvFile, hmFile):
    create_er_file(tripleList, entities, subArea, csvFile)
    create_heatmap(csvFile, subArea, hmFile)

def create_er_file(tripleList, entities, subArea, csvFile):
    print("----------")
    print("running create_er_file.")
    print("----------")
    print(entities)
    if os.path.exists(csvFile):
        print(f"deleting old ER csvFile: {csvFile}.")
        os.remove(csvFile)
    else:
        print(f"old ER csvFile doesnt exist. nothing to delete.")
    print(f"creating new ER csvFile: {csvFile}.")
    with open(csvFile, 'w', encoding="utf-8") as f:        
        f.write(f'Entity1,Relationship,Entity2\n')
        for entity in entities:
            f.write(f'{subArea},relates-to,{entity}\n')
        for txt in tripleList:
            triple = txt.replace(",", "").replace("->", ",")
            f.write(f'{triple}\n')
        f.close()

# Create hm file
def create_heatmap(csvFile, subArea, hmFile):
    print("----------")
    print("running create_heatmap.")
    print("----------")
    print(f"reading ER csvFile: {csvFile}.")
    df = pd.read_csv(csvFile, usecols=['Entity1', 'Relationship', 'Entity2'])
# Pivot the data to create a matrix
    pivot_df = df.pivot_table(index='Entity1', columns='Entity2', values='Relationship', aggfunc='first')
    plt.title(f'{subArea} Heatmap')
    fig, ax = plt.subplots(figsize=(15,10))
    plt.subplots_adjust(left=0.2, bottom=0.3)
    sns.heatmap(pivot_df.notnull(), cmap='YlGnBu', cbar=False, ax=ax)
#    plt.show()  
    if os.path.exists(hmFile):
        print(f"deleting old heatmap file: {hmFile}.")
        os.remove(hmFile)
    else:
        print("old heatmap file doesnt exits. nothing to delete.")
    print(f"creating new heatmap file: {hmFile}.")
    plt.savefig(hmFile)
    plt.close()
      
def create_md(content, mdFile):
    print("----------")
    print("running create_md.")
    print("----------")
    try:
        os.remove(mdFile)
    except:
        print("nothing to delete")
#    print(content)
    with open(mdFile, 'w', encoding="utf-8") as f:
        try:
            f.write(content)
        except:
            print("error creating md file")
    
    f.close()    

def insert_rag(RAG_DIR, mdFile, llm_model):
    print("----------")
    print("running insert_rag.")
    print("----------")

    if not os.path.exists(RAG_DIR):
        os.mkdir(RAG_DIR)
    llm_model_func = llm_func_options[llm_model]
    rag = LightRAG(
    working_dir=RAG_DIR,
    llm_model_func=llm_model_func,  
# Optionally, use a stronger model
#   llm_model_func=gpt_4o_complete, 
#    kg="Neo4JStorage",
    graph_storage="Neo4JStorage",
    log_level="INFO",
    )
    text = ""
    with open(mdFile, "r", encoding="utf-8") as f:
        text = f.read()
    f.close()
# Insert in nano-vectordb as embedding, neo4j graphDB as KGs.        
    rag.insert(text)
    print("----------")
    print("DONE.")
    print("----------")
    return rag
    
def summary(objective, content):
    text_splitter = RecursiveCharacterTextSplitter(
        separators =["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt = map_prompt_template,
#        verbose=True
        verbose=False
    )

    output = summary_chain.run(input_documents=docs, objective=objective)
    return output

def create_agent(model, tools, system_message: str):
    """Create an agent."""
    print("----------")
    print("running create_agent.")
    print("----------")
    print(f"tools: {tools}")
#Initialize model
    if model == 'gpt-4o' or model == 'gpt-4o-mini':
#        llm = ChatOpenAI(temperature=0, streaming=True, model=model, max_tokens=max_out_tokens)
        llm = AzureChatOpenAI(api_version="2024-12-01-preview", streaming=True, model=model)
#o1-mini-2024-09-12
    if model == 'gemini-flash':
        llm = ChatVertexAI(temperature=0, streaming=True, model='gemini-2.0-flash-exp', max_tokens=max_out_tokens)
#    if model == 'gemini-flash':
#        llm = ChatGoogleGenerativeAI(temperature=0, 
#        streaming=True, 
#        model='gemini-1.5-flash', maxConcurrency=2, max_tokens=max_out_tokens)
    if model == 'claude-haiku':
        llm = ChatAnthropic(temperature=0, streaming=True, model='claude-3-5-haiku-latest', max_tokens=max_out_tokens)
#llm = ChatAnthropic(temperature=0, streaming=True, model='claude-3-5-sonnet-latest', max_tokens=max_out_tokens)
#llm = ChatGroq(temperature=0, model="llama-3.1-70b-versatile")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards completing all the tasks system user has provided."
                " If you or any of the other assistants have completed all the tasks user had asked for, then prefix your response with FINAL ANSWER."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
#            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    print("----------")
    print("DONE.")
    print("----------")
    return prompt | llm.bind_tools(tools)
#    return prompt | llm.bind_tools(tools) | JsonOutputFunctionsParser()


# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
    remaining_steps: RemainingSteps

# Helper function to create a node for a given agent
def agent_node(state, agent, name):
    print("----------")
    print("Running Agent: " + name)
    print("----------")
#    print(f"agent: {agent}")
#    print(f"state: {state}")
    result = agent.invoke(state)
#    print(result)
    print("----------")
    print("DONE.")
    print("----------")
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }

# Either agent can decide to end

def router(state) -> Literal["call_tool", "__end__", "continue"]:
    print("----------")
    print("Running Router Agent.")
    print("----------")
    messages = state["messages"]
    print(f"messages: {messages}")
    last_message = messages[-1]
#    print(f"last_message: {last_message}")
    last_message_content = last_message.content
#    print(f"last_message_content: {last_message_content}")
    last_agent_name = last_message.name
    print(f"last_agent_name: {last_agent_name}")
    print(f"last_message_tool_calls: {last_message.tool_calls}")
    print(f"remaining_steps: {state["remaining_steps"]}")
    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"
    if state["remaining_steps"] <= 2:
        print("----------")
        print("Forced complete, before hitting recurrsion limit")
        print("----------")
        return "__end__"
    if ("FINAL ANSWER" in last_message_content or "COMPLETE" in last_message_content) and last_agent_name == "Coder_Agent" :
        # Any agent decided the work is done
        print("----------")
        print("All Tasks Complete")
        print("----------")
        return "__end__"
    if isinstance(last_message_content, list):
        last_message_text = last_message.content[0]["text"]
#       print(f"last_message_text: {last_message_text}")
        if ("FINAL ANSWER" in last_message_text or "COMPLETE" in last_message_text) and last_agent_name == "Coder_Agent" :
            # Any agent decided the work is done
            print("----------")
            print("All Tasks Complete")
            print("----------")
            return "__end__"
    return "continue"

def create_graph(model):
    print("----------")
    print("running create_graph.")
    print("----------")
# Research agent and node
    research_agent = create_agent(
    model,
#    [tavily_tool, google_search ],
    [create_md_content],
    
    system_message=f"""
    You are a world class MEDICAL researcher. 
    You are not a code writer. 
    You are not a code executor.     
    You are a ONLY a web searcher agent with insights to summarize and create a world class research paper.
    Based on the system user's list of tasks and topics, you will use the web search tool "create_md_content" and will collect tool's markdown output as content. This markdown content has all the information for the topic system user asked.
    Call the tool "create_md_content" "ONLY-ONCE".
    To call the tool "create_md_content", provide below info to the tool:
    1/ subArea
    2/ keyword
    3/ Duration
    4/ entities
    5/ pngPath
    Then, from the markdown content you received from the tool, extract all the relevant entities and their relationships.
    Create list of triples of "entity->relationship->entity".
    You do not make things up, you create list of triples of "entity->relationship->entity" based on facts in the markdown content you have. 
    Provide: 
        1/ the markdown content you have from "create_md_content" tool's output
        2/ the list of triples of "entity->relationship->entity" you created
        3/ Therapeutic SubArea
        4/ csv File path
        5/ hm File path
        6/ md File path
    to the next agent: Coder_Agent, so that Coder_Agent can complete the further tasks of storing the files on system user's local server.
    When you are done collecting data, you signal completion of your current task, so that Coder_Agent can start working on next steps.""",
)
    research_node = functools.partial(agent_node, agent=research_agent, name="Researcher_Agent")

# Coder_Agent
    Coder_Agent = create_agent(
    model,
    [python_repl, create_all_files],
    system_message=f"""You are a ONLY world class expert python code writer and code executor. 
You are not a web searcher.
Make sure that you have received below items: 
    1/ markdown_content
    2/ list of triples of "entity->relationship->entity" 
from Researcher_Agent agent. 
If any of these 2 items are missing, don't do any of your file creation task below. Immidiately respond back with continue signal, so that control will be passed back to the Researcher_Agent Agent for further actions.
If you have received 2 items: 
    1/ markdown_content 
    2/ list of triples of "entity->relationship->entity" 
from Researcher_Agent agent. 

Follow the below Steps:

StepA: Only keep in your memory the name of the files to be created and the below 2 items: 
    1/ all triples of "entity->relationship->entity"
    2/ markdown_content
you received from Researcher_Agent agent. 
Clear out your memory of any unnecessory info.

StepB: Pass 1/ markdown_content as "content", 2/ the list of triples as "tripleList", 3/ Therapeutic "subArea", and below filePaths:
4/ csvFile
5/ hmFile
6/ mdFile
to the tool create_all_files. 
The tool create_all_files will create the files on the system user's local server.

IMPORTANT. StepC: Never use python_repl tool to delete any file.
Only use python_repl tool run your python code needed to do your task. 

StepD: When executing your code using python_repl tool, if you see any error due to any missing code_library, then go ahead and download and install that missing code_library on system user's local server. 
When installing the missing code_library on the user's local server first determine: "what is the operating system on the system user's local server?". Then use the appropriate command which is suitable for system user's local server's operating system. Use those suitable commands to download and install the code_libraries on the system user's local server.
Then, go ahead and re-run your code. 
If you find that your code has any syntax error while executing it, then, go ahead and re-write your code and execute it again on the system user's local server using the python_repl tool. 
For creating the files, call the tool create_all_files. 
Double check your task list and only when you are sure that you have attempted to complete all your tasks at least once.

Call the tool create_all_files "ONLY-ONCE" and then signal back your task as: COMPLETE."""
,
)
    coder_node = functools.partial(agent_node, agent=Coder_Agent, name="Coder_Agent")

#tools = [tavily_tool, google_search, python_repl]
    tools = [create_md_content, python_repl, create_all_files]

    tool_node = ToolNode(tools)

    workflow = StateGraph(AgentState)

    workflow.add_node("Researcher_Agent", research_node)
    workflow.add_node("Coder_Agent", coder_node)
    workflow.add_node("call_tool", tool_node)

    workflow.add_conditional_edges(
    "Researcher_Agent",
    router,
    {"continue": "Coder_Agent", "call_tool": "call_tool", "__end__": END},
    )
    workflow.add_conditional_edges(
    "Coder_Agent",
    router,
    {"continue": "Researcher_Agent", "call_tool": "call_tool", "__end__": END},
    )

    workflow.add_conditional_edges(
    "call_tool",
    # Each agent node updates the 'sender' field
    # the tool calling node does not, meaning
    # this edge will route back to the original agent
    # who invoked the tool
    lambda x: x["sender"],
    {
        "Researcher_Agent": "Researcher_Agent",
        "Coder_Agent": "Coder_Agent",
    },
    )
    workflow.set_entry_point("Researcher_Agent")
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
# create agent run graph file.
    graph.get_graph().draw_mermaid_png(output_file_path=f"Multi-Agent-Workflow.png", draw_method=MermaidDrawMethod.API)
    print("----------")
    print("DONE.")
    print("----------")

    return graph

#try:
#    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
#except:
    # This requires some extra dependencies and is optional
#    pass

def count_tokens(text: str, llm_model: str) -> int:
    """Counts the number of tokens in a text string for a given OpenAI model."""
    print(f"----------")
    print(f"running count_tokens")
    print(f"----------")
    if llm_model == 'gpt-4o' or llm_model == 'gpt-4o-mini':
        encoding = tiktoken.encoding_for_model(llm_model)
    if llm_model == 'gemini-flash':
        encoding = tiktoken.encoding_for_model('gpt-4o-mini')
    if llm_model == 'claude-haiku':
        encoding = tiktoken.encoding_for_model('gpt-4o-mini')
    print(f"----------")
    print("DONE")
    print(f"----------")
    return len(encoding.encode(text))

#Run the graph
def run_show_markdown(Area, SubArea, Keyword, Duration, Key, entities, model):

    currTime1 = datetime.datetime.now()
    llm_model = llm_options[model]
    print(f"user selected LLM model: {llm_model}")
    entity_list = ' AND '.join(str(x) for x in entities)
    entities = entity_list
    input_message = f"""Using the tool create_md_content, collect markdown content within last {Duration}, on the subject {Keyword} in the Therapeutic Area of {Area} and SubTherapeutic Area of {SubArea} and having information about the entities: {entities}. Supply the pngPath to the tool create_md_content: {pngPath}. 
    Supply {SubArea}-heatmap.png as hmFile to the tool create_md_content.

    Then, using the tool create_all_files, create the below files:
{filePath}{SubArea}-entity-relations.csv know as csvFile
{pngPath}/{SubArea}-heatmap.png known as hmFile
{filePath}{SubArea}-all_research_data.md known as mdFile
"""
    print(f'user message to llm: {input_message}')
    try:
        token_count = 0
        token_count = count_tokens(input_message, llm_model)
#        print(entities)
        print(f"----------")
        print(f"There are {token_count} tokens in llm input")
        print(f"----------")
        if token_count < max_in_tokens:
            thread_id = str(uuid.uuid4())
            print(f"----------")
            print(f"thread_id = {thread_id}")
            print(f"----------")
            graph = create_graph(llm_model)
            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
        # The config is the **second positional argument** to stream() or invoke()!

#            msgs = graph.invoke(
#            {"messages": [("user", input_message)]}, config, stream_mode="values")
#            for msg in msgs['messages']:
#                if "messages" in event:
#                    out_resp = msg["messages"][-1]

            events = graph.stream(
            {"messages": [("user", input_message)]}, config, stream_mode="values")
            for event in events:
                if "messages" in event:
                    out_resp = event["messages"][-1]
        else:
            out_resp = f"content='Skipped calling LLM'"

        print(f"----------")
        print(f"Agents completed. Inserting data in VectorDB and GraphDB")
        print(f"----------")

        rag = insert_rag(RAG_DIR, f'{filePath}{SubArea}-all_research_data.md', model
        )

# mode = naive, local, global OR hybrid search
#        query=f"summarize the content about {SubArea}"
#        answer = rag.query(query, param=QueryParam(mode=mode))
#        print('---------------')
#        print(f'using query mode: {mode}')
#        print(answer)
#        print('---------------')
    except BaseException as e:
        error = repr(e)
        print("Error: " + error)
        out_resp = f"Sorry, unable to answer your query."
    print(f"----------")
    print(f"llm_response: {out_resp}")
    print(f"----------")
    file = f"{filePath}{SubArea}-all_research_data.md"
    with open(file, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    with open(file, "r", encoding="utf-8") as f:
        markdown_content = f.read()
    html = markdown.markdown(markdown_content)
    currTime2 = datetime.datetime.now()
    time_diff = currTime2 - currTime1
    print(f"----------")
    print(f"Time taken(in sec): {time_diff.total_seconds()}")
    print(f"----------")
    return html

def display_markdown(filename):
    with open(filename, "r", encoding="utf-8") as f:
        markdown_content = f.read()
        return markdown_content 

def init_RAG(llm_model):

    print(f"----------")
    print(f"init RAG")
    print(f"----------")
    llm_model_func = llm_func_options[llm_model]
    rag = LightRAG(
    working_dir=RAG_DIR,
    graph_storage="Neo4JStorage",
    llm_model_func=llm_model_func,  
    log_level="INFO",
    )
    print(f"----------")
    print(f"DONE")
    print(f"----------")
    return rag, "RAG is initialized!"

def ask_RAG(rag, query, history):
    currTime1 = datetime.datetime.now()
    print(f"----------")
    print("running ask RAG")
    print(f"----------")

    mode = 'hybrid'
    print(f"mode: {mode}, user query: {query}")
    if rag:
        try:
            answer = rag.query(query, param=QueryParam(mode=mode))
        except:
            answer = "your query didn't bring any useful information."
    else:
        answer = "initilize the Chatbot first and then ask you question"
    print(f"answer: {answer}")
    new_history = history + [(query, answer)]
    print(f"----------")
    print(f"DONE")
    print(f"----------")
    currTime2 = datetime.datetime.now()
    time_diff = currTime2 - currTime1
    print(f"----------")
    print(f"Time taken(in sec): {time_diff.total_seconds()}")
    print(f"----------")
    return rag, gr.update(value=""), new_history


# Create the Gradio interface
gr.close_all()

output = []
ms_choices = ['Biomarkers','Drugs','Genes','Pathogens','Proteins','Tissue Cell Types']
options_1 = ['Oncology', 'Hematology', 'Dermatology', 'Immunology', 'Neuroscience', 'Psychiatry', 'Gastroenterology', 'Cardiology', 'Ophthalmology', 'Endocrinology']
options_2 = {
    'Oncology': ['Breast Cancer','Lung Cancer','Prostate Cancer','Colorectal Cancer','Leukemia','Lymphoma','Melanoma','Glioblastoma','Multiple Myeloma','Ovarian Cancer'],
    'Hematology': ['Leukemia','Lymphoma','Multiple Myeloma','Anemia','Coagulation','Hemophilia','Sickle Cell','Thrombocytopenia','Myelodysplastic Syndrome','Myeloproliferative Disorder'],
    'Dermatology': ['Acne','Eczema','Psoriasis','Melanoma','Atopic Dermatitis','Rosacea','Alopecia','Vitiligo','Skin Infection','Keloids'],
    'Immunology': ['Rheumatoid Arthritis','IBD','Multiple Sclerosis','Lupus','Psoriasis','Asthma','Atopic Dermatitis','Food Allergy','Transplant Rejection','Immuno Deficiency'],
    'Neuroscience': ['Alzheimers','Parkinsons','Stroke','Epilepsy','Migraine','Multiple Sclerosis','Amyotrophic Lateral Sclerosis','Huntingtons','Brain Injury','Neuropathic Pain'],
    'Psychiatry': ['Depression','Anxiety Disorder','Schizophrenia','Bipolar','PTSD','OCD','Substance Use','Eating Disorder','ADHD','Autism'],
    'Gastroenterology': ['IBD','IBS','Gastro Reflux','Cirrhosis','Hepatitis','Gastro Cancer','Pancreatitis','Celiac','Gastro Disorder','Gastro Bleeding'],
    'Cardiology': ['Coronary Artery','Heart Failure','Hypertension','Arrhythmia','Valvular Heart','Congenital Heart Defect','Peripheral Artery','Aortics','Pulmonary Hypertension','Cardiomyopathy'],
    'Ophthalmology': ['Cataract','Glaucoma','Macular Degeneration','Diabetic Retinopathy','Dry Eye','Myopia','Hyperopia','Astigmatism','Retinal Detachment','Uveitis','Corneals','Optic Nerve Disorder'],
    'Endocrinology': ['Diabetes','Thyroid','Obesity','Osteoporosis','Adrenal Disorder','Pituitary Disorder','Metabolic Syndrome','Infertility','Growth Disorder','Hypogonadism'],
    }

with gr.Blocks() as demo:

    rag = gr.State()
    gr.Markdown(
            '''
        # <center>Research Evidence Portal<center>
        ''')    
    gr.HTML(markdown.markdown(display_markdown("Overview.md")))

    with gr.Row():
        Key = gr.Textbox(label="API_KEY", type="password", value="123xxxxxxxxxx123")
    with gr.Row():
        llm_btn = gr.Radio(llm_options, label="Available LLMs", value = llm_options[0], type="index") 

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                with gr.Row():
                    Area = gr.Dropdown(choices=options_1, label="Therapeutic Area", value='Oncology')
                    SubArea = gr.Dropdown([], label='Sub-Therapeutic Area', value='Breast Cancer', allow_custom_value=True)
    
                    def update_second(first_val):
                        SubArea = gr.Dropdown(options_2[first_val], label='Sub-Therapeutic Area')
                        return SubArea 
    
                    Area.input(update_second, Area, SubArea)

                    outputs = gr.Textbox(visible=False)

                    def print_results(option_1, option_2):
                        return f"You selected '{option_1}' in the first dropdown and '{option_2}' in the second dropdown."
        
                    SubArea.input(print_results, [Area, SubArea], outputs) 

                    Duration = gr.Dropdown(["1 week", "2 weeks", "1 month", "3 months", "6 months", "1 year", "5 years", "10 years"], value="10 years",
                        label="Duration")
                    Keyword = gr.Textbox(label="Search", value="Cancer")
                    submit_btn = gr.Button("Search")
                    output2 = gr.Textbox(label="Selections", visible=False)
                    entities = gr.Dropdown(
                        choices=ms_choices,
                        multiselect=True,
                        label="Entities")

                    def process_selection(selected_options):
                        return "You selected: " + ", ".join(selected_options)
                    entities.change(process_selection, entities, output2)


        with gr.Column(scale=2):
            output = gr.HTML()
            chatbot = gr.Chatbot(height=305)
            with gr.Row():
                    qa_init_btn = gr.Button("Initialize Chatbot")
            with gr.Row():
                    llm_progress = gr.Textbox(value="Not initialized", show_label=False)
            with gr.Row():
                    msg = gr.Textbox(placeholder="Ask a question", container=True)
            with gr.Row():
                    qa_btn = gr.Button("Submit")
#                    clear_btn = gr.ClearButton([msg, chatbot], value="Clear")

        # Preprocessing events
    submit_btn.click(fn=run_show_markdown, inputs=[Area, SubArea, Keyword, Duration, Key, entities, llm_btn], outputs=output, api_name="run_show_markdown")

        # Chatbot events
    qa_init_btn.click(init_RAG, \
            inputs=[llm_btn], \
            outputs=[rag, llm_progress], \
            queue=False)
    msg.submit(ask_RAG, \
            inputs=[rag, msg, chatbot], \
            outputs=[rag, msg, chatbot], \
            queue=False)
    qa_btn.click(ask_RAG, \
            inputs=[rag, msg, chatbot], \
            outputs=[rag, msg, chatbot], \
            queue=False)
#    clear_btn.click(lambda:[None],inputs=None,outputs=[chatbot],queue=False)

    gr.HTML(f"<img src='{filePathPre}=./Multi-Agent-Workflow.png' width='400' height='400' alt='image Two'>")

#demo.launch()
#server_port=8080, 
demo.launch(share=False, server_name='0.0.0.0', server_port=7860, allowed_paths=["./", f"{pngPath}", f"{filePath}"])