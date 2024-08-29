## Context-Aware RAG Pipeline Fine-Tuned on <b>google/gemma-2b-it</b> for Financial Questions

<figure markdown="span">
  ![Sole Gemma](../images/Gemma-image.png){ width="300" }
  <figcaption>Sole Gemma</figcaption>
</figure>

### Abstract

Large language models (LLMs) have shown remarkable capabilities in understanding context and providing relevant responses to prompts. In the financial domain, optimizing context selection can significantly enhance the accuracy and relevance of answers. This project aims to improve the performance of an LLM, specifically the “google/gemma-2b-it” model, for financial queries by leveraging context from renowned financial books, including “An Intelligent Investor,” “Rich Dad Poor Dad,” “The Bond King,” “Patient Capital,” and “The Ultimate Day Trader.”

## Github code

Please find the github code at [link](https://github.com/mrunalmania/FinancialBot) 

### Methodology

PDF text extraction


```python
import fitz
from tqdm import tqdm

def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """
    Open a PDF file and extract text from each page.

    Args:
    - pdf_path (str): Path to the PDF file.

    Returns:
    - list[dict]: A list of dictionaries containing information about each page.
    """
    doc = fitz.open(pdf_path)
    pages_and_texts = []

    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        # Additional text preprocessing can be done here (e.g., text normalization)
        pages_and_texts.append({
            "page_number": page_number - 14,  # Adjust page number to start from 0
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count_raw": len(text.split(".")),
            "page_token_count": len(text) / 4,  # Assuming an average word length of 4 characters
            "text": text
        })

    return pages_and_texts

# Example usage
pdf_path = "path/to/your/pdf/file.pdf"
pages_and_texts = open_and_read_pdf(pdf_path)
```

### Data Processing


```python
from spacy.lang.en import English

# Initialize English tokenizer
nlp = English()

# Add the sentencizer pipeline to the tokenizer
nlp.add_pipe("sentencizer")

# Example text
text = "We have 2 cars. We want to purchase one more. We don't have enough money."

# Tokenize the text into sentences
doc = nlp(text)

# Check if the tokenizer has correctly split the text into sentences
assert len(list(doc.sents)) == 3

# Print the sentences
sentences = list(doc.sents)
print(sentences)
```

### Chunking sentences

```python
def split_list(input_list: list[str], slice_size: int) -> list[list[str]]:
    """
    Split a list of texts into smaller chunks.

    Args:
    - input_list (list[str]): List of texts to be split.
    - slice_size (int): Size of each chunk.

    Returns:
    - list[list[str]]: A list of smaller chunks of texts.
    """
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

# Example usage
test_list = list(range(25))
split_list(test_list, 10)
```

### Enhancing Text Chunk Analysis for Improved Data Insights

In the realm of text analysis, breaking down large blocks of text into smaller, more manageable chunks can significantly enhance our ability to derive insights. Whether it’s analyzing customer feedback, parsing through research papers, or summarizing literature, text chunking plays a pivotal role.

In this article, we’ll explore how to process text chunks extracted from PDF documents, ensuring that the data is cleaned and organized for further analysis. We’ll focus on key steps, including splitting chunks, joining sentences, and calculating statistics, to provide a comprehensive guide for anyone looking to dive into text analysis.

### Splitting Chunks and Joining Sentences

To begin, we start with a collection of text chunks, each representing a section of text from a PDF document. Our goal is to break down these chunks further into individual sentences and then join them back together into cohesive paragraphs.

```python
import re

# Split each chunk into its own item
pages_and_chunks = []
for item in tqdm(pages_and_texts):
    for sentence_chunk in item["sentences_chunks"]:
        chunk_dict = {}
        chunk_dict["page_number"] = item["page_number"]

        # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
        joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
        joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk) # ".A" -> ". A" for any full-stop/capital letter combo
        chunk_dict["sentence_chunk"] = joined_sentence_chunk

        # Get stats about the chunk
        chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
        chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
        chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters

        pages_and_chunks.append(chunk_dict)

# How many chunks do we have?
len(pages_and_chunks)
```

### Embedding those chunks

```python
from sentence_transformers import SentenceTransformer

# Initialize the SentenceTransformer model
embedding_model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

# Create a list of sentences
sentences = ["This framework generates embeddings for each input sentence",
             "Sentences are passed as a list of string.",
             "The quick brown fox jumps over the lazy dog."]

# Encode the sentences to get the embeddings
embeddings = embedding_model.encode(sentences)
embeddings_dict = dict(zip(sentences, embeddings))

# Print the embeddings for each sentence
for sentence, embedding in embeddings_dict.items():
    print(f"Sentence: {sentence}")
    print(f"Embedding: {embedding}\n")
```
### Advanced Retrieval Techniques

We employed advanced retrieval techniques, including vector search with dot product similarity, to identify relevant text chunks for context selection. This approach ensures that the model focuses on the most relevant information when generating responses to financial questions.

### Hyperparameter Tuning

Hyperparameter tuning plays a crucial role in optimizing the performance of our model. Through grid search, we identified the optimal hyperparameters, including a temperature value of 0.3, do_sample set to True, and the use of flash_attn_v2 for attention implementation.

### Retrieval of Relevant Resources

Our retrieval mechanism uses embeddings to compare the query with all other text chunks, calculating dot product scores to identify the most relevant resources. This process ensures that the model retrieves accurate and contextually appropriate information for each query.

### Response Generation

Using the “google/gemma-2b-it” model, we generate responses to financial queries based on the retrieved resources and the context provided by the financial books. The model’s ability to understand context and generate coherent responses enables it to provide valuable insights into complex financial questions.

```python
def prompt_formmater(query: str, context_items: list[dict]) -> str:
   context = "-" + "\n -".join([item["sentence_chunk"] for item in context_items])

   base_prompt = """Please use the following context to get idea of what is the context about and try to answer the question: \n\n
   query:
   {query}
   context:
   {context}
   """


   base_prompt = base_prompt.format(query=query, context=context)
   dialogue_template = [
       {"role": "user",
        "content": base_prompt}
   ]

   # Apply the chat template
   prompt = tokenizer.apply_chat_template(conversation=dialogue_template, tokenize=False, add_generation_prompt=True)
   return prompt

query = "What is the intelligent investor?"
scores, indices = retrieval_relevant_resources(query, embeddings)

context_items = [pages_and_chunks[i] for i in indices]
prompt = prompt_formmater(query, context_items)
print(prompt)

def ask_me(query: str,
           temperature: float = 0.5,
           max_new_token: int = 256,
           format_answer = True,
           return_answer_only = True):
  scores, indices = retrieval_relevant_resources(query, embeddings)

    # create the context item
  context_items = [pages_and_chunks[i] for i in indices]

  # Add score to context item
  for i, item in enumerate(context_items):
    item["score"] = scores[i].cpu()

  prompt = prompt_formmater(query, context_items)

  input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

  outputs = model.generate(**input_ids, max_new_tokens = max_new_token, temperature = temperature, do_sample=True)

  outputs_decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)


  return outputs_decoded

query = "What is foreign policy"
final_ans = ask_me(query)
print(f"Final output: \n {final_ans.replace(prompt, '')}")
```

### Future Work

In future iterations of our project, we plan to enhance the context retrieval phase by implementing reranking algorithms. These algorithms can further improve the relevance and accuracy of the retrieved context, ensuring that the model is presented with the most relevant information for generating responses.

Additionally, we aim to explore the use of more advanced language models, such as “google/gemma-7b-it,” to further enhance the performance of our system. These larger models have the potential to capture more nuanced relationships in the data and generate even more accurate and contextually appropriate responses.

By incorporating reranking algorithms and leveraging more advanced language models, we aim to continue improving the accuracy and effectiveness of our financial question answering system, ultimately providing users with more valuable insights and information.

### Conclusion

Enhancing a language model for financial question answering requires a combination of context enrichment, advanced retrieval techniques, and hyperparameter tuning. By leveraging these strategies, we have developed a powerful tool for extracting insights and answering queries in the financial domain. 