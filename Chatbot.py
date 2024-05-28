import os
import sys
import spacy
import re
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.vectorstores import Chroma
import constants
import time


nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


transition_phrases = ["Regarding", "Furthermore", "Additionally", "Note:"]
transition_phrases_pattern = re.compile(r"(?<!\.)\s(" + "|".join(transition_phrases) + ")")

os.environ["OPENAI_API_KEY"] = constants.APIKEY

PERSIST = False

#(placeholder cache)
cache = {}


def preprocess_question(question):

    doc = nlp(question)
    cleaned_question = " ".join(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)
    return cleaned_question


def format_answer_with_bullets(answer):
    answer = transition_phrases_pattern.sub(r". \1", answer)

    items = re.split(r'(?<=[^A-Z].[.?]) (?=[A-Z])', answer)
    bullet_formatted_answer = "\n".join(f"• {item.strip()}" for item in items if item)

    return "Here are the responses with bullet points:\n\n" + bullet_formatted_answer


def construct_prompt(user_input):
    instructions = [
        "Offer thorough explanations to ensure users understand the reasoning behind each decision.",
        "Maintain clarity and brevity in your responses, avoiding unnecessary elaboration.",
        "You only answer based on the context you know.",
        "When responding to queries about the characters' names or any literary references, include a brief explanation about the origin of the name if it relates to literature.",
        "Use a playful tone to highlight the irony or humor in the situation, especially when discussing the characters' indifference to whales.",
        "Offer additional literary trivia or related topics if the user seems engaged or interested in literature.",
        "Encourage user interaction by asking open-ended questions like, 'Would you like to hear more about how these names influence their behavior?'"
    ]
    instruction = " ".join(instructions)
    return f"{instruction}\n\nUser: {user_input}\nAI:"




if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else None

    if PERSIST and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        loader = DirectoryLoader("data/")
        embedding_function = OpenAIEmbeddings()
        if PERSIST:
            index = VectorstoreIndexCreator(embedding=embedding_function,
                                            vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator(embedding=embedding_function).from_loaders([loader])

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-4o"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    chat_history = []
    while True:

        user_input = input("Prompt: ") if not query else query
        if user_input.lower() in ['quit', 'q', 'exit']:
            sys.exit()

        start_time = time.time()

        # Check cache before processing
        if user_input in cache:
            formatted_answer = cache[user_input]
        else:
            prompt = construct_prompt(user_input)
            query = preprocess_question(user_input)
            result = chain({"question": prompt, "chat_history": chat_history})
            formatted_answer = format_answer_with_bullets(result['answer'])
            cache[user_input] = formatted_answer

        print(formatted_answer)
        end_time = time.time()
        result = end_time - start_time
        print("Time: ", result)
        chat_history.append((user_input, formatted_answer))
        query = None