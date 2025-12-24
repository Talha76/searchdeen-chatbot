from typing import Any, Generator, List
import os
from dotenv import load_dotenv
from langchain_elasticsearch import ElasticsearchRetriever
from langchain_groq import ChatGroq
from elasticsearch import Elasticsearch
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.messages import SystemMessage, AnyMessage
from langsmith import traceable
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

_retriever = ElasticsearchRetriever(
    client=Elasticsearch(hosts=["http://localhost:9200"]),
    index_name="*",
    body_func=lambda query: {
        "query": {
            "multi_match": {
                "query": query.replace('"', '\\"'),
                "fuzziness": 1,
            }
        },
        "highlight": {
            "phrase_limit": 512,
            "fields": {
                "content": {}, "content_arabic": {}, "title": {}
            }
        }
    },
    content_field="content",
)
_llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"), # type: ignore
    model="openai/gpt-oss-120b",
    reasoning_effort="medium",
    temperature=0,
)
_query_translation_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="""You are an expert at query understanding and reformulation.

Your task is to rewrite the user's original query into a single improved version that is clearer, more specific, and better optimized for high-quality search retrieval response.

Instructions:
- Preserve the original intent — do NOT change the meaning.
- Remove ambiguity, vagueness, and unnecessary words.
- Add helpful context or constraints ONLY if they are logically implied.
- Do not answer the query; only reformulate it.

Output the reformulation as a search-optimized version ideal for retrieval."""),
        HumanMessagePromptTemplate.from_template("Original query: {question}"),
    ]
)
_final_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="""You are an Islamic knowledge assistant.

Your role is to answer questions about Islam strictly based on the provided sources
retrieved from Elasticsearch. You must not rely on memory, general knowledge,
assumptions, or inference beyond what is explicitly stated in the sources.

Core Rules (Mandatory):
- Use ONLY the provided retrieved sources as your knowledge base.
- Do NOT add, infer, assume, or extrapolate information.
- If the sources do not contain enough information to answer fully, say so clearly.
- Never guess, speculate, or fill gaps.
- Do not merge information from different sources unless they explicitly align.
- Do not issue personal opinions or modern interpretations unless directly cited.

Scholarly Accuracy Rules:
- Quote Qur’anic verses, Hadith, or scholarly opinions only if they appear verbatim
  or clearly referenced in the sources.
- Preserve wording and meaning exactly as presented.
- If multiple scholarly views are present, list them separately without preference.
- Do not claim consensus (ijmāʿ) unless explicitly stated in the sources.

Answering Guidelines:
- Begin by checking whether the retrieved sources are sufficient.
- If sufficient, answer using clear, neutral, and respectful language.
- Cite sources inline using the provided identifiers (e.g., [Source 1], [Hadith A]).
- If insufficient, respond with:
  "The provided sources do not contain enough information to answer this question."

Prohibited Behavior:
- No assumptions about the user’s intent, belief, or level of knowledge.
- No reasoning beyond textual evidence.
- No synthesis that introduces new meaning.
- No religious verdicts (fatwas) unless explicitly labeled as such in the sources.

Output Format:
- Short direct answer (if possible)
- Bullet points or paragraphs strictly tied to sources
- Source citations after each factual claim"""),
        ("placeholder", "{history}"),  # Conversation history placeholder
        HumanMessagePromptTemplate.from_template("User question: {question}"),
        HumanMessagePromptTemplate.from_template("Retrieved sources: {context}"),
    ]
)
def _retrieved_docs_formatter(docs):
    formatted_docs = []
    for i, doc in enumerate(docs, 1):  # Limit to top 5 documents
        content = f"{i}. {doc.page_content}"
        try:
            arabic = doc.metadata.get('_source', {}).get('extras', {}).get('arabic')
            if not arabic:
                arabic = doc.metadata.get('content_arabic', '')
            if arabic:
                content += f"\nArabic: {arabic}"
        except (AttributeError, KeyError):
            pass  # Skip Arabic text if not available
        formatted_docs.append(content)
    return "\n\n".join(formatted_docs)
_context_generator = (
    _query_translation_prompt
    | _llm
    | StrOutputParser()
    | _retriever
    | _retrieved_docs_formatter
)
_final_pipeline = (
    {
        "question": lambda x: x["question"],
        "context": _context_generator,
        "history": lambda x: x["history"],
    }
    | _final_prompt
    | _llm
    | StrOutputParser()
)

@traceable
def get_response(question: str, history: list[AnyMessage] = [], *args, **kwargs) -> Generator[str]:
    """Get response from LLM based on the question."""
    # Invoke the pipeline
    stream = _final_pipeline.stream({
        'question': question,
        'history': history,
    })
    for chunk in StrOutputParser().transform(stream):
        yield chunk  # type: ignore


if __name__ == "__main__":
    query = input("Enter your question: ")
    print("Generating response...")
    for resp in get_response(query):
        print(resp, end="", flush=True)
