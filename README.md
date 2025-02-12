# PrivateAssistantLite

Note that this project is for learning purposes, and it involves using an LLM-Agent.

## introduction

This is a project focused on LLM-Agent. 

I am building a RAG workflow that will help us learn about web security.

I’m currently running the DeepSeek-R1: 7B model locally using Ollama. 

This model is distilled from QWen 2.5. 

You can replace it with any suitable large model by simply modifying the ``self.llm`` part in ``pipeline.py``.

## TODO

1. Instead of conducting a full web search using Google, searching on a few specific professional websites.
2. Configure a local knowledge base to accelerate the model’s loading speed. (Maybe follow ``ask.py`` to use DuckDB)
3. Use Streamlit to create a visual front-end interface.

## How to Customize your LLM-Agent.

Configure your Google Custom Search API, and set up domain restrictions within it. 

Note that you should follow the rules from Google Custom Search API.

For example, if you’re building an expert system for cybersecurity, these websites are all you need

Maybe there are some limits, you can choose what you need to configue.

``` text
   security.stackexchange.com/
   www.reddit.com/r/netsec/
   www.securityweek.com/
   thehackernews.com/
   zhihu.com
   wikipedia.org
   stackoverflow.com
   github.com
   gitee.com
   cdsn.net
```

## reference

1. [pengfeng/ask.py](https://github.com/pengfeng/ask.py?tab=readme-ov-file)
2. [Alicloud Developer class](https://developer.aliyun.com/article/1266585)
3. [deepseek api status](https://status.deepseek.com/#)

## quick-start

1. python environment

   ``` shell
   ## It’s recommended to use a virtual environment.
    pip install conda

    conda create --name test python=3.10
    conda activate test

   ## If you plan to use your local environment directly, 
   ## please disregard the previous instructions.
    pip install -r requirements.txt
   ```
2. env config
   ```
    SEARCH_API_KEY=<your_api_key>
    SEARCH_PROJECT_KEY=<your_cx>
    SEARCH_API_URL=https://www.googleapis.com/customsearch/v1
   ```

[google custom search](https://developers.google.com/custom-search/v1/overview?hl=zh-cn)

3. 
   ``` shell
    python demo.py
   ```