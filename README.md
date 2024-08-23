# QA BOT 

QA Bot to provide answers based on a given context. 🚀


## DEMO VIDEO 
https://drive.google.com/file/d/10u4fYrbESEjL3-q4fgWz7Pry9y1I3j4N/view?usp=sharing

## SETUP
- pip3 install -r requirements.txt
- export OPENAI_API_KEY="{OPENAI_API_KEY}"

## RUN QA BOT
- fastapi run
- Server run on [0.0.0.0:8000](http://0.0.0.0:8000)

## API DOCS
- Docs are at on [0.0.0.0:8000/docs](http://0.0.0.0:8000/docs)

## Fetch Answers API
```http
POST /bot/qa
```

#### Payload

```json
{
  "document_path": "{absolute_document_path}",
  "document_type": "{pdf|json}",
  "questions": [
    "{question_1}",
    "{question_2}",
    "{question_3}"
    ....
  ]
}
```

#### Responses

```json
{
  "result": {
    "{question}": "{answer}",
  }
}
```

### Note: 
- Python3 and pip3 are pre-requisities that are assumed to be present in the os already 
- libmagic and tesseract are pre-requisities that are assumed to be present in the os already
- Python3.12 was used for the developement and testing
