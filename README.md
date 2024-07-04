# ZaniAGENT

This is a PDF extractor agent that allows us to choose the models while performing QnA.

## Run Locally

Clone the project

```bash
  git clone git@github.com:yadavlukish/zaniagent.git
```
OR

```bash
  git clone git@github.com:yadavlukish/zaniagent.git
```


Create .env file and add keys in below format

```bash
  OPENAI_API_KEY=<>
  SLACK_TOKEN=<>
  HUGGINGFACEHUB_API_TOKEN=<>
```

Go to the project directory

```bash
  cd zaniagent
```

Install dependencies

```bash
  pip requirements.txt
```

Start the server

```bash
  streamlit run app.py
```

Gpt4 docker issue

```bash
  https://github.com/zylon-ai/private-gpt/issues/729
```

