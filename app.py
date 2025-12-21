from rag_qa import initialize_embeddings, initialize_llm, initialize_vector_store, create_rag_chain, get_response

import time
import streamlit as st

st.title("Telecom Q&A RAG")

# Step 1: Initialize embeddings
embeddings = initialize_embeddings()

# Step 2: Initialize LLM
llm = initialize_llm()

# Step 3: Initialize vector store
vectorstore = initialize_vector_store(embeddings)

# Step 4: Create RAG chain (returns chain and retriever)
rag_chain, retriever = create_rag_chain(vectorstore, llm)

print(1)
def response_generator(response):
    
    for word in response.split():
        yield word + " "
        time.sleep(0.05)
import time

def character_stream_generator(response):
    """Streams the response character by character."""
    for char in response:
        yield char
        # A much smaller sleep time is needed for smooth character-level stream
        time.sleep(0.01) 

# Example usage in Streamlit:
# st.write_stream(character_stream_generator("This is **bold** text.\nAnd a new line."))


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = get_response(rag_chain, retriever, prompt)
        response = st.write_stream(character_stream_generator(response))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
        
#if __name__ == "__main__":
#    main()