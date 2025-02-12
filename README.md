## simple spider

NOTE that this project is for learning.

### reference

1. [pengfeng/ask.py](https://github.com/pengfeng/ask.py?tab=readme-ov-file)
2. [阿里云社区开发者课程](https://developer.aliyun.com/article/1266585)

### quick-start

1. python environment
   It's recommended to use virtual env.

   ``` shell
    pip install conda

    conda create --name test python=3.10
    conda activate test

    pip install -r requirements.txt
   ```
2. env config
   ```
    SEARCH_API_KEY=<your_api_key>
    SEARCH_PROJECT_KEY=<your_cx>
    SEARCH_API_URL=https://www.googleapis.com/customsearch/v1
   ```

[google custom search](https://developers.google.com/custom-search/v1/overview?hl=zh-cn)

3. run it
   ``` shell
    python demo.py
   ```