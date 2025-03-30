#relik/azure
el_model='relik'
# Azure Entity Linking: https://learn.microsoft.com/en-us/azure/ai-services/language-service/entity-linking/overview
# key&endpoint
Azure_key = '8cClqtqeXC9C49Q01UMwtkqifGbABXrbO0mljNQqp7fnCEfXAiWKJQQJ99ALACNns7RXJ3w3AAAaACOGKoAq'
Azure_endpoint = 'https://el.cognitiveservices.azure.com/'
#----------------If you use relik to Entity Linking,state it here--------------------------------
# relik: https://github.com/SapienzaNLP/relik
relik_server_url = 'http://210.40.16.12:48361/api/relik'
# If you use proxies to access the Internet, please set the proxy address.
proxies={
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}
#----------------For embedding model,you can choose local or online service----------------------
nv_embed_v2_url = 'http://210.40.16.12:24449/embeddings'
# ----------------For LLM model config,state it here----------------------
# ----deepseekV3/qwen2.5-instruct-14b/gpt3.5-turbo----------
model_name = "deepseekV3_ali"
# If you use qwen2.5-instruct-14b,you need to set aliApiKey,https://help.aliyun.com/en/model-studio/developer-reference/get-api-key
aliApiKey = "sk-aeb0ed24d0c045e78d73fff879da5f07"
# If you use gpt3.5turbo,you need to set openaiApiKey,https://platform.openai.com/account/api-keys
openaiApiKey = ""
# If you use deepseekV3,you need to set deepseekApiKey,https://platform.deepseek.com/
deepseekApiKey = "sk-eead0d1fed4e45eb9fdacfdc6a1df855"

# Get article chunks by nv-embed-v2/text-embedding-3-small
RetrieveModelName = "nv-embed-v2"
# If you use chroma to store wikipedia articles,please set Persistent Client Path, chroma collection name.
Chroma_store = False
PersistentClient_Path = "D:\Chroma"
chroma_collection_name = "wikipediaNV"# You can set any name you like, but chroma_collection_name needs to correspond to the retrieval model.

# Use ELTop_k, RLTop_k to set the top k of the entity and relation linking results for graph search.
ELTop_k = 2
RLTop_k = 1
# Get top_k of text blocks to self-adaptive
article_top_k = 2
