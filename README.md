# Learn how to use the OpenAI API and GPT-4o for Data Science

This guide explains how to create specialized AI assistants tailored to data science tasks using the OpenAI API and GPT-4o. It covers custom instructions, file processing, data analysis, and multi-modal capabilities.

## **Before you begin:**

1. Ensure you have an OpenAI developer account.
2. Define an environment variable called `OPENAI_API_KEY` that contains your API key.

---

## **Task 0: Set up**

Install the latest version of the OpenAI package:

```python
!pip install openai==1.33.0
```

Import required packages:

```python
import os
import openai
import pandas as pd
```

Define an OpenAI client:

```python
OPENAI_API_KEY="your_api_key_here"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = openai
```

---

## **Task 1: Upload the Papers**

To inform GPT about the latest AGI research, upload Arxiv papers.

Define the dataset:

```python
papers = pd.DataFrame({
    "filename": [
        "2405.10313v1.pdf", "2401.03428v1.pdf", "2401.09395v2.pdf",
        "2401.13142v3.pdf", "2403.02164v2.pdf", "2403.12107v1.pdf",
        "2404.10731v1.pdf", "2312.11562v5.pdf", "2311.02462v2.pdf",
        "2310.15274v1.pdf"
    ],
    "title": [
        "How Far Are We From AGI?",
        "Exploring Large Language Model Based Intelligent Agents",
        "Caught in the Quicksand of Reasoning, Far From AGI Summit",
        "Unsocial Intelligence: Assumptions of AGI Discourse",
        "Cognition is All You Need: Next Layer of AI Above LLMs",
        "Scenarios for the Transition to AGI",
        "What is Meant by AGI? On the Definition of Artificial General Intelligence",
        "A Survey of Reasoning with Foundation Models",
        "Levels of AGI: Operationalizing Progress on the Path to AGI",
        "Systematic AI Approach for AGI"
    ]
})
papers["filename"] = "papers/" + papers["filename"]
```

Define a function to upload files:

```python
def upload_file_for_assistant(file_path):
    uploaded_file = client.files.create(
        file=open(file_path, "rb"),
        purpose='assistants'
    )
    return uploaded_file.id
```

Upload the papers:

```python
uploaded_file_ids = papers["filename"].apply(upload_file_for_assistant).to_list()
```

---

## **Task 2: Add the Files to a Vector Store**

Create a vector store and associate uploaded files:

```python
vstore = client.beta.vector_stores.create(
    file_ids=uploaded_file_ids,
    name="arxiv_agi_papers"
)
```

---

## **Task 3: Create the Assistant**

Define an assistant prompt:

```python
assistant_prompt = """
When explaining the contents of the papers, follow these guidelines:
- **Introduction:** Briefly state the title, authors, and research question.
- **Abstract Summary:** Highlight key points and findings.
- **Key Sections:** Summarize methodology, results, and significance.
- **Conclusion:** Provide the authors' conclusions and future research directions.
- **Critical Analysis:** Discuss strengths, weaknesses, and innovative contributions.
- **Contextual Understanding:** Relate the paper to broader AGI research.
- **Practical Takeaways:** Identify useful methodologies and insights for data scientists.
- **Q&A Readiness:** Be prepared for follow-up questions.
"""
```

Create the assistant:

```python
aggie = client.beta.assistants.create(
    name="Aggie",
    instructions=assistant_prompt,
    model="gpt-4o",
    tools=[{"type": "file_search"}],
    tool_resources={"file_search": {"vector_store_ids": [vstore.id]}}
)
```

---

## **Task 4: Create a Conversation Thread**

Create a conversation thread:

```python
conversation = client.beta.threads.create()
```

Send a user message:

```python
msg_what_is_agi = client.beta.threads.messages.create(
    thread_id=conversation.id,
    role="user",
    content="What are the most common definitions of AGI?"
)
```

---

## **Task 5: Run the Assistant**

Define an event handler:

```python
from typing_extensions import override
from openai import AssistantEventHandler

class EventHandler(AssistantEventHandler):
    @override
    def on_text_created(self, text) -> None:
        print(f"\nassistant > ", end="", flush=True)
    @override
    def on_text_delta(self, delta, snapshot):
        print(delta.value, end="", flush=True)
```

Run the assistant:

```python
def run_aggie():
    with client.beta.threads.runs.stream(
        thread_id=conversation.id,
        assistant_id=aggie.id,
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()

run_aggie()
```

---

## **Task 6: Add Another Message and Run Again**

Send a follow-up message:

```python
msg_how_close_is_agi = client.beta.threads.messages.create(
    thread_id=conversation.id,
    role="user",
    content="How close are we to developing AGI?"
)
```

Run the assistant again:

```python
run_aggie()
```

---

## **Conclusion**

This guide walks through setting up an AI assistant that processes and explains AGI research papers. By following these steps, you can create a powerful data science assistant tailored to your needs.

