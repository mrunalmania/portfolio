## Multi-Modal Product Insight: A Comprehensive Guide

<figure markdown="span">
  ![Sole Gemma](../images/sun_painting.jpg){ width="300" }
  <figcaption>Whether the sun rises or sets is up to our imagination.</figcaption>
</figure>

### Introduction

In January and February of 2024, I embarked on an exciting project to engineer a multi-modal instruction system using BLIP (Bootstrapped Language-Image Pre-training) and PyTorch. This project aimed to generate descriptive product information by leveraging neural networks and integrating with AWS for scalable cloud processing. Throughout this journey, I showcased my skills in statistics, applied mathematics, and algorithms for data science using Python, PyTorch, TensorFlow, and Scikit-Learn, demonstrating expertise in neural networks and machine learning.

## Github code

Please find the github code at [link](https://github.com/mrunalmania/InstructBLIP) 

### Project Overview

#### Objective

The primary objective was to develop a system capable of generating rich, detailed descriptions of product images, utilizing a combination of state-of-the-art machine learning models and scalable cloud infrastructure.

#### Tech Stack

PyTorch: For building and training neural networks.
FAISS: For efficient similarity search and clustering of dense vectors.
BLIP: To integrate language and vision models for multimodal tasks.
AWS: For scalable cloud processing and storage.

### Implementation

#### Code and Description

Setting Up the Model

The project began with setting up the BLIP model for conditional generation. We used the `Salesforce/instructblip-vicuna-7b` model with 4-bit precision to balance performance and resource usage.

```python
import torch
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

model = InstructBlipForConditionalGeneration.from_pretrained(
    "Salesforce/instructblip-vicuna-7b",
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
)

processor = InstructBlipProcessor.from_pretrained(
    "Salesforce/instructblip-vicuna-7b",
    load_in_4bit=True,
)

```


#### Loading Datasets

We loaded various datasets to train and test our model, including fashion, sports, planes, snacks, pets, and Pokémon.

```python
from datasets import load_dataset

datasets = [
    ("detection-datasets/fashionpedia", None, "val"),
    ("keremberke/nfl-object-detection", "mini", "test"),
    ("keremberke/plane-detection", "mini", "train"),
    ("Matthijis/snacks", None, "validation"),
    ("romkmr/mini_pets", None, "test"),
    ("keremberke/pokemon-classification", "mini", "train")
]

```

#### Generating Descriptions

For each image in the datasets, we generated detailed descriptions using two different prompts.

```python
prompt1 = "describe this image in full detail"
prompt2 = "create an extensive description of this image"

counter = 0

for name, config, split in datasets:
    d = load_dataset(name, config, split=split)
    for idx in range(len(d)):
        image = d[idx]['image']
        desc = ""

        for _prompt in [prompt1, prompt2]:
            inputs = processor(
                images=image,
                text=_prompt,
                return_tensors='pt'
            ).to(model.device, torch.bfloat16)
            outputs = model.generate(
                **inputs,
                do_samples=False,
                num_beams=10,
                max_length=512,
                min_length=16,
                top_p=0.9,
                repetition_penalty=1.5,
                temperature=1
            )
            generated_text = processor.batch_decode(
                outputs,
                skip_special_tokens=True
            )[0].strip()
            
            desc += generated_text + " "
        desc = desc.strip() # remove \n, \t 
        image.save(f"images/{counter}.jpg")

        print(counter, desc)

        with open("description.csv", "a") as f:
            f.write(f"{counter},{desc}\n")
        
        counter += 1
        torch.cuda.empty_cache()

```

#### Creating Embeddings

We then created embeddings for the generated descriptions using the sentence-transformers library.

```python
from sentence_transformers import SentenceTransformer
import pickle
import pandas as pd

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

with open("description.csv", 'r') as f:
    lines = f.readlines()

lines = [line.strip().split(",") for line in lines]

for idx, line in enumerate(lines):
    lines[idx] = [line[0], ",".join(line[1:])]

df = pd.DataFrame(lines, columns=['id', 'desc'])

embeddings = model.encode(df['desc'].tolist(), show_progress_bar=True)

with open('embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

print(embeddings.shape)
```

#### Building the Search Application

To enable users to search for images based on text queries, we built a search application using Gradio and FAISS for efficient similarity search.

```python
import gradio as gr
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)
embeddings = embeddings.astype("float32")

embedding_size = embeddings.shape[1]
n_clusters = 1
num_results = 1

quantizer = faiss.IndexFlatIP(embedding_size)

index = faiss.IndexIVFFlat(
    quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT,
)

index.train(embeddings)
index.add(embeddings)

def _search(query):
    query_embedding = model.encode(query)
    query_embedding = np.array(query_embedding).astype("float32")
    query_embedding = query_embedding.reshape(1, -1)

    _, indices = index.search(query_embedding, num_results)
    print(indices)
    images = [f"images/{i}.jpg" for i in indices[0]]
    print(images)
    return images

with gr.Blocks() as demo:
    query = gr.Textbox(lines=1, label="Search Query")
    outputs = gr.Gallery(preview=True)
    submit = gr.Button(value="Search")
    submit.click(_search, inputs=query, outputs=outputs)

demo.launch()
```

### Conclusion

This project not only demonstrated my proficiency in neural networks and machine learning but also highlighted my ability to integrate complex systems using advanced technologies and cloud infrastructure. The multi-modal instruction system we built is a testament to the power of combining vision and language models to create meaningful and scalable applications.