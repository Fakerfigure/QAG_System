import httpx
from openai import OpenAI
from .config import get_model_config  # 从配置模块导入

def chatbot(content):
    """动态获取配置的chatbot实现"""
    config = get_model_config()
    
    if not config.get("api_key"):
        return "请先在配置页面设置API Key", 0
    
    try:
        client = OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            http_client=httpx.Client(
                # proxies="http://127.0.0.1:7897",  
                timeout=45  # 增加超时设置
            )
        )

        completion = client.chat.completions.create(
            model=config["model"],
            messages=[{'role': 'user', 'content': content}],
            temperature=config["temperature"]
        )
        
        answer = completion.choices[0].message.content
        used_tokens = completion.usage.total_tokens
        print(answer,used_tokens)
        return answer, used_tokens
    except httpx.ConnectError:
        return "连接超时，请检查网络设置", 0
    except Exception as e:
        return f"API请求失败: {str(e)}", 0