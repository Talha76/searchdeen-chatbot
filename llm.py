from typing import Generator
from langchain_elasticsearch import ElasticsearchRetriever
from langchain_groq import ChatGroq
from elasticsearch import Elasticsearch
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.messages import SystemMessage, AnyMessage
from langsmith import traceable
from langchain_core.output_parsers import StrOutputParser


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
_llm_medium_reasoning = ChatGroq(
    model="openai/gpt-oss-120b",
    reasoning_effort="medium",
    temperature=0,
)
_query_translation_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="""You are an expert at query understanding and reformulation.

Task: Your task is to rewrite the user's original query into a single improved version that is clearer, more specific, and better optimized for high-quality AI responses and search retrieval.

Instructions:
- Preserve the original intent — do NOT change the meaning.
- Remove ambiguity, vagueness, and unnecessary words.
- Add helpful context or constraints ONLY if they are logically implied.
- Add relevant synonyms or closely related terms for key words to improve search and retrieval performance (without changing meaning).
- Do not answer the query; only reformulate it."""),
        HumanMessagePromptTemplate.from_template("Original query: {question}"),
    ]
)
_final_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="""You are an Islamic knowledge assistant.

Task: Answer questions strictly and only using the provided sources. Do not rely on prior knowledge, assumptions, inference, or guesswork.

Rules:
- Use ONLY the retrieved sources.
- Consider ONLY sources that are directly relevant to the question.
- Do NOT add, infer, interpret, or extrapolate beyond the text.
- If the sources are insufficient, clearly state that the question cannot be answered.
- Do not issue opinions or fatwas unless explicitly stated in the sources.
- Do not claim consensus unless explicitly mentioned.

Scholarly Accuracy:
- Quote Qur’an, Hadith, or scholars only if present in the sources.
- Preserve wording and meaning exactly.
- If multiple views exist, list them neutrally without preference.

Answering:
- Provide a concise, neutral answer.
- Cite sources after each factual claim.
- If no clear answer exists, say so explicitly."""),
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
    | _llm_medium_reasoning
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
    | _llm_medium_reasoning
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
