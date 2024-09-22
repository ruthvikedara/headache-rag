# HeadacheBot: A Simple Assistant for Neurologists

HeadacheBot is a minimal chatbot designed to assist neurologists by answering common questions about headaches. It uses a small dataset of 50 question-answer pairs, along with natural language processing techniques, to retrieve the closest matching answer based on user input, and optionally, provides a more detailed response using a large language model (LLM).

**Motivation:** Specialist physicians like neurologists often have very high patient loads and not enough time to answer all of their questions. This project is a proof of concept that could help allievate some workload off of busy neurologists by answering simple questions related to headaches.

## Features
- Efficient Answer Retrieval: Uses sentence embeddings to find the most relevant answer from a pre-defined set of questions.
- Optional LLM-Generated Responses: Users can request a more detailed answer powered by a large language model, based on the retrieved information.
- Minimalist Design: Focused on simplicity and functionality, with a clear use case in healthcare.

## How It Works
- Dataset: A set of 50 Q&A pairs focused on headache-related topics, covering common questions neurologists may encounter.
- Sentence Embeddings & Vector Search: Each question is embedded into a vector space using a pre-trained sentence embedding model. When a user asks a question, the system computes cosine similarity to find the closest match.
- LLM Integration (Optional): After retrieving an answer, users have the option to get a more detailed explanation from a large language model, which uses both the original question and the retrieved answer as context.

## Discussion
The idea of using only vector-based retrieval was to ensure that only correct information is presented to the user.

However, the LLM often gives more empathetic and satisfactory answers. Its use in a clinical setting has to be tightly monitored.

## Future Improvements
1. Expand Dataset: Add more clinical questions to broaden the scope.
2. Extracting Semantic Information: Use techniques like named entity recognition and relation extraction to better understand more complicated questions from the user and provide more accurate answers.
3. Model Refinement: Explore fine-tuning or using specialized medical models for improved accuracy.
