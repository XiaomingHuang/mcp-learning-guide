# Model Context Protocol (MCP) 学习指南

## 目录
- [前言](#前言)
- [第一阶段：MCP 基础概念](#第一阶段mcp-基础概念)
  - [1. MCP 协议概述](#1-mcp-协议概述)
  - [2. MCP 核心组件](#2-mcp-核心组件)
  - [3. Claude 与 MCP](#3-claude-与-mcp)
- [第二阶段：MCP 实践应用](#第二阶段mcp-实践应用)
  - [1. 基础工具集成](#1-基础工具集成)
  - [2. 数据源集成](#2-数据源集成)
  - [3. 上下文管理](#3-上下文管理)
- [第三阶段：高级应用开发](#第三阶段高级应用开发)
  - [1. Agent 开发](#1-agent-开发)
  - [2. 系统集成](#2-系统集成)
  - [3. 性能优化](#3-性能优化)
- [开发资源](#开发资源)
  - [官方资源](#官方资源)
  - [开发工具](#开发工具)
  - [实践项目](#实践项目)
- [学习建议](#学习建议)

## 前言

Model Context Protocol (MCP) 是由 Anthropic 在 2024 年推出的开放协议，旨在标准化大型语言模型（LLM）与外部数据源、工具之间的交互方式。本学习指南将帮助您系统地学习 MCP，从基础概念到高级应用，让您能够有效地利用 MCP 构建强大的 AI 应用。

## 第一阶段：MCP 基础概念

### 1. MCP 协议概述

#### 设计目的和核心理念

MCP 协议的主要目标是解决 LLM 在与外部世界交互时的标准化问题。其核心理念包括：

a) 统一接口标准
- 为 LLM 提供标准化的工具调用接口
- 统一的上下文管理方式
- 一致的数据交换格式

b) 结构化交互
- 明确定义工具的输入输出格式
- 规范化的错误处理机制
- 可追踪的调用链路

#### 标准化交互实现

MCP 通过以下机制实现 LLM 与外部工具的标准化交互：

a) 工具注册和发现
```python
# 工具定义示例
tools = [
    {
        "name": "calculator",
        "description": "Performs basic calculations",
        "parameters": {
            "operation": "string",
            "numbers": "array"
        },
        "returns": "number"
    }
]
```

b) 上下文传递
```python
# 上下文管理示例
context = {
    "conversation_id": "conv_123",
    "user_query": "计算 2+3",
    "available_tools": tools,
    "previous_results": []
}
```

c) 标准化调用流程
1. LLM 理解用户意图
2. 选择合适的工具
3. 构建标准格式的调用请求
4. 处理工具返回结果
5. 将结果整合到响应中

#### MCP 相比传统 prompt 的优势

a) 结构化能力对比：
- 传统方式：
```python
# 传统 prompt 方式
prompt = """使用计算器工具计算 2+3
可用工具：calculator(numbers, operation)
"""
```

- MCP 方式：
```python
# MCP 方式
tool_call = {
    "tool": "calculator",
    "parameters": {
        "operation": "add",
        "numbers": [2, 3]
    }
}
```

b) 主要优势：
1. 更好的可控性
   - 明确的工具能力边界
   - 可预测的交互流程
   - 标准化的错误处理

2. 更高的可靠性
   - 减少提示词工程的不确定性
   - 提供类型安全
   - 便于调试和监控

3. 更强的扩展性
   - 易于添加新工具
   - 支持复杂的工具组合
   - 便于版本管理

#### 实际应用示例

```python
from anthropic import Anthropic
from mcp_client import MCPClient

# 初始化客户端
client = MCPClient(
    model="claude-3",
    tools=[calculator, web_search, database]
)

# 处理用户请求
response = client.process({
    "query": "计算网页上提到的数字之和",
    "context": current_context,
    "tools": {
        "allowed": ["web_search", "calculator"],
        "preferred": "calculator"
    }
})
```

### 2. MCP 核心组件

#### MCP 的基本架构

MCP 的架构主要包含以下核心组件：

a) Context Manager（上下文管理器）
```python
class MCPContext:
    def __init__(self):
        self.conversation_history = []
        self.active_tools = {}
        self.current_state = {}
        self.memory = {}

    def update(self, new_context):
        # 更新上下文状态
        self.current_state.update(new_context)
```

b) Tool Registry（工具注册器）
```python
class ToolRegistry:
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, tool_name, tool_config):
        # 注册新工具
        self.tools[tool_name] = tool_config
```

#### 上下文管理机制

MCP 的上下文管理包含三个关键部分：

a) 对话历史管理
```python
def manage_history(context, new_message):
    # 添加新消息到历史记录
    context.conversation_history.append({
        'timestamp': datetime.now(),
        'content': new_message,
        'type': 'user_message'
    })
    
    # 维护历史记录大小
    if len(context.conversation_history) > MAX_HISTORY:
        context.conversation_history = context.conversation_history[-MAX_HISTORY:]
```

b) 状态追踪
```python
def track_state(context, current_action):
    # 记录当前状态
    context.current_state.update({
        'last_action': current_action,
        'tool_calls': context.current_state.get('tool_calls', 0) + 1,
        'timestamp': datetime.now()
    })
```

c) 内存管理
```python
def manage_memory(context, key, value):
    # 存储关键信息
    context.memory[key] = {
        'value': value,
        'created_at': datetime.now()
    }
```

#### 工具调用接口规范

MCP 定义了标准的工具调用接口：

a) 工具定义
```python
tool_definition = {
    "name": "weather_api",
    "description": "Get weather information for a location",
    "parameters": {
        "location": {
            "type": "string",
            "description": "City name or coordinates"
        },
        "units": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "default": "celsius"
        }
    },
    "returns": {
        "type": "object",
        "properties": {
            "temperature": "number",
            "condition": "string"
        }
    }
}
```

b) 调用格式
```python
async def tool_call(context, tool_name, parameters):
    try:
        tool = context.active_tools.get(tool_name)
        if not tool:
            raise ToolNotFoundError(f"Tool {tool_name} not found")
            
        result = await tool.execute(parameters)
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
```

#### 数据源集成

MCP 提供了标准化的数据源集成方式：

a) 数据源接口
```python
class DataSource:
    def __init__(self, config):
        self.config = config
        self.connection = None
    
    async def connect(self):
        # 建立连接
        pass
    
    async def query(self, query_params):
        # 执行查询
        pass
    
    async def stream(self, stream_params):
        # 流式数据处理
        pass
```

b) 数据转换器
```python
class DataTransformer:
    def to_mcp_format(self, data):
        # 将数据转换为MCP标准格式
        return {
            "type": "data_response",
            "format": "json",
            "content": data
        }
```

### 3. Claude 与 MCP

#### Claude API 中的 MCP 支持

首先需要从 Anthropic 导入必要的组件：

```python
from anthropic import Anthropic
from anthropic.tools import Tool, ToolCall

# 初始化 Anthropic 客户端
client = Anthropic(api_key="your_api_key")
```

定义工具示例：
```python
calculator_tool = Tool(
    name="calculator",
    description="执行基本的数学计算",
    parameters={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"]
            },
            "numbers": {
                "type": "array",
                "items": {"type": "number"}
            }
        },
        "required": ["operation", "numbers"]
    }
)

search_tool = Tool(
    name="search",
    description="搜索互联网信息",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "number"}
        },
        "required": ["query"]
    }
)
```

#### 基本调用方式

a) 基础消息发送：
```python
def send_message_with_tools(user_input):
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        temperature=0.7,
        messages=[{
            "role": "user",
            "content": user_input
        }],
        tools=[calculator_tool, search_tool]
    )
    return message
```

b) 处理工具调用：
```python
def handle_tool_calls(message):
    if message.tool_calls:
        for tool_call in message.tool_calls:
            if tool_call.tool_name == "calculator":
                result = execute_calculation(tool_call.parameters)
            elif tool_call.tool_name == "search":
                result = execute_search(tool_call.parameters)
            
            # 将工具执行结果发送回 Claude
            response = client.messages.create(
                model="claude-3-opus-20240229",
                messages=[{
                    "role": "tool_response",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                }]
            )
```

#### 上下文处理最佳实践

a) 维护对话历史：
```python
class ConversationManager:
    def __init__(self):
        self.messages = []
        self.tool_states = {}
        
    def add_message(self, role, content):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })
    
    def add_tool_result(self, tool_name, result):
        self.tool_states[tool_name] = {
            "last_result": result,
            "last_used": datetime.now()
        }
```

b) 智能上下文裁剪：
```python
def trim_context(messages, max_tokens=8000):
    """智能裁剪上下文以适应模型限制"""
    total_tokens = 0
    trimmed_messages = []
    
    for msg in reversed(messages):
        tokens = estimate_tokens(msg['content'])
        if total_tokens + tokens <= max_tokens:
            trimmed_messages.insert(0, msg)
            total_tokens += tokens
        else:
            break
            
    return trimmed_messages
```

c) 错误处理和重试机制：
```python
async def safe_tool_execution(tool_call, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            result = await execute_tool(tool_call)
            return result
        except TemporaryError as e:
            retries += 1
            await asyncio.sleep(2 ** retries)  # 指数退避
        except PermanentError as e:
            return {"error": str(e)}
```

## 第二阶段：MCP 实践应用

### 1. 基础工具集成

以下是一个实际的示例，创建一个包含计算器和天气查询的简单应用。

#### 工具定义和集成

```python
from anthropic import Anthropic
from typing import Dict, List, Any
import json
import aiohttp
from datetime import datetime

# 定义工具
calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "执行基本的数学计算",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"]
                },
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"}
                }
            },
            "required": ["operation", "numbers"]
        }
    }
}

weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "country": {"type": "string"}
            },
            "required": ["city"]
        }
    }
}
```

#### 工具执行器

```python
class ToolExecutor:
    def __init__(self, weather_api_key: str):
        self.weather_api_key = weather_api_key
        
    async def execute_calculator(self, params: Dict[str, Any]) -> Dict[str, Any]:
        operation = params["operation"]
        numbers = params["numbers"]
        
        try:
            if operation == "add":
                result = sum(numbers)
            elif operation == "subtract":
                result = numbers[0] - sum(numbers[1:])
            elif operation == "multiply":
                result = 1
                for num in numbers:
                    result *= num
            elif operation == "divide":
                result = numbers[0]
                for num in numbers[1:]:
                    if num == 0:
                        raise ValueError("Division by zero")
                    result /= num
                    
            return {
                "status": "success",
                "result": result
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def execute_weather(self, params: Dict[str, Any]) -> Dict[str, Any]:
        city = params["city"]
        country = params.get("country", "")
        
        async with aiohttp.ClientSession() as session:
            try:
                # 这里使用示例 API，实际使用时需替换为真实的天气 API
                url = f"https://api.weatherapi.com/v1/current.json?key={self.weather_api_key}&q={city}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "status": "success",
                            "temperature": data["current"]["temp_c"],
                            "condition": data["current"]["condition"]["text"]
                        }
                    else:
                        return {
                            "status": "error",
                            "error": f"API error: {response.status}"
                        }
            except Exception as e:
                return {
                    "status": "error",
                    "error": str(e)
                }
```

#### 上下文管理

```python
class ConversationContext:
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.tool_results: Dict[str, Any] = {}
        self.last_update: datetime = datetime.now()
    
    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })
        self.last_update = datetime.now()
    
    def add_tool_result(self, tool_name: str, result: Dict[str, Any]):
        self.tool_results[tool_name] = {
            "result": result,
            "timestamp": datetime.now()
        }
        self.last_update = datetime.now()
    
    def get_recent_context(self, max_messages: int = 10) -> List[Dict[str, Any]]:
        return self.messages[-max_messages:]
```

#### 应用主类

```python
class MCPApplication:
    def __init__(self, api_key: str, weather_api_key: str):
        self.client = Anthropic(api_key=api_key)
        self.tool_executor = ToolExecutor(weather_api_key)
        self.context = ConversationContext()
        self.tools = [calculator_tool, weather_tool]
    
    async def process_user_input(self, user_input: str) -> str:
        # 添加用户输入到上下文
        self.context.add_message("user", user_input)
        
        # 创建 Claude 消息
        response = await self.client.messages.create(
            model="claude-3-opus-20240229",
            messages=self.context.get_recent_context(),
            tools=self.tools
        )
        
        # 处理工具调用
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_result = await self.execute_tool(tool_call)
                self.context.add_tool_result(
                    tool_call.function.name,
                    tool_result
                )
            
            # 将工具结果发送回 Claude
            final_response = await self.client.messages.create(
                model="claude-3-opus-20240229",
                messages=self.context.get_recent_context() + [{
                    "role": "tool_response",
                    "content": json.dumps(tool_result)
                }]
            )
            
            self.context.add_message("assistant", final_response.content)
            return final_response.content
        
        self.context.add_message("assistant", response.content)
        return response.content
    
    async def execute_tool(self, tool_call) -> Dict[str, Any]:
        tool_name = tool_call.function.name
        params = tool_call.function.arguments
        
        if tool_name == "calculator":
            return await self.tool_executor.execute_calculator(params)
        elif tool_name == "get_weather":
            return await self.tool_executor.execute_weather(params)
        else:
            return {
                "status": "error",
                "error": f"Unknown tool: {tool_name}"
            }
```

### 2. 数据源集成

#### 数据库连接器

```python
from typing import Dict, Any, List
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import aiofiles
from datetime import datetime

class DatabaseConnector:
    def __init__(self, connection_url: str):
        self.engine = create_async_engine(connection_url)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
    
    async def query_database(self, query: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            async with self.async_session() as session:
                result = await session.execute(query, params or {})
                data = result.fetchall()
                return {
                    "status": "success",
                    "data": [dict(row) for row in data],
                    "timestamp": datetime.now()
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def execute_write(self, query: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            async with self.async_session() as session:
                await session.execute(query, params or {})
                await session.commit()
                return {"status": "success"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
```

#### 文档检索系统

```python
from elasticsearch import AsyncElasticsearch
import tiktoken
from typing import List, Dict, Any

class DocumentRetrieval:
    def __init__(self, elasticsearch_url: str, index_name: str):
        self.es_client = AsyncElasticsearch([elasticsearch_url])
        self.index_name = index_name
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    async def search_documents(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        try:
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title", "content"],
                        "fuzziness": "AUTO"
                    }
                },
                "size": max_results
            }
            
            results = await self.es_client.search(
                index=self.index_name,
                body=search_body
            )
            
            documents = []
            for hit in results["hits"]["hits"]:
                doc = hit["_source"]
                documents.append({
                    "title": doc["title"],
                    "content": doc["content"],
                    "score": hit["_score"]
                })
            
            return {
                "status": "success",
                "documents": documents
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def chunk_document(self, content: str, max_tokens: int = 500) -> List[str]:
        tokens = self.tokenizer.encode(content)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for token in tokens:
            if current_length + 1 > max_tokens:
                chunks.append(self.tokenizer.decode(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(token)
            current_length += 1
            
        if current_chunk:
            chunks.append(self.tokenizer.decode(current_chunk))
            
        return chunks
```

#### 结构化数据处理器

```python
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import json

class DataProcessor:
    def __init__(self):
        self.supported_formats = ["json", "csv", "excel"]
    
    async def process_structured_data(
        self, 
        data: Any, 
        format_type: str,
        operations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        try:
            # 转换数据为 DataFrame
            df = self._convert_to_dataframe(data, format_type)
            
            # 应用操作
            for operation in operations:
                df = self._apply_operation(df, operation)
            
            return {
                "status": "success",
                "data": df.to_dict(orient="records"),
                "summary": self._generate_summary(df)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _convert_to_dataframe(self, data: Any, format_type: str) -> pd.DataFrame:
        if format_type == "json":
            return pd.DataFrame(json.loads(data))
        elif format_type == "csv":
            return pd.read_csv(data)
        elif format_type == "excel":
            return pd.read_excel(data)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _apply_operation(self, df: pd.DataFrame, operation: Dict[str, Any]) -> pd.DataFrame:
        op_type = operation["type"]
        
        if op_type == "filter":
            return df.query(operation["condition"])
        elif op_type == "sort":
            return df.sort_values(operation["column"], ascending=operation.get("ascending", True))
        elif op_type == "group":
            return df.groupby(operation["columns"]).agg(operation["aggregations"])
        elif op_type == "transform":
            return df.assign(**{operation["column"]: df.eval(operation["expression"])})
        
        return df
    
    def _generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
            "missing_values": df.isnull().sum().to_dict()
        }
```

#### 数据源集成器

```python
class DataSourceIntegration:
    def __init__(
        self,
        db_url: str,
        es_url: str,
        es_index: str
    ):
        self.db_connector = DatabaseConnector(db_url)
        self.doc_retrieval = DocumentRetrieval(es_url, es_index)
        self.data_processor = DataProcessor()
        
    async def handle_data_request(
        self,
        request_type: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            if request_type == "database":
                return await self.db_connector.query_database(
                    params["query"],
                    params.get("parameters")
                )
            elif request_type == "document":
                return await self.doc_retrieval.search_documents(
                    params["query"],
                    params.get("max_results", 5)
                )
            elif request_type == "structured":
                return await self.data_processor.process_structured_data(
                    params["data"],
                    params["format"],
                    params.get("operations", [])
                )
            else:
                return {
                    "status": "error",
                    "error": f"Unsupported request type: {request_type}"
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
```

### 3. 上下文管理

#### 动态上下文管理系统

```python
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
import asyncio
from collections import deque

@dataclass
class ContextItem:
    content: Any
    timestamp: datetime
    importance: float = 1.0
    type: str = "general"
    
    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "type": self.type
        }

class DynamicContext:
    def __init__(self, max_size: int = 1000):
        self.context_items: deque = deque(maxlen=max_size)
        self.importance_threshold: float = 0.5
        self.context_types: Dict[str, float] = {
            "user_input": 1.0,
            "tool_result": 0.8,
            "system_message": 0.6,
            "background": 0.4
        }
        
    async def add_item(self, content: Any, type: str = "general", importance: Optional[float] = None):
        """添加新的上下文项"""
        if importance is None:
            importance = self.context_types.get(type, 0.5)
            
        item = ContextItem(
            content=content,
            timestamp=datetime.now(),
            importance=importance,
            type=type
        )
        self.context_items.append(item)
        await self._cleanup_old_items()
        
    async def _cleanup_old_items(self):
        """清理过期或低重要性的项目"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=24)
        
        self.context_items = deque(
            [item for item in self.context_items 
             if (item.timestamp > cutoff_time or item.importance > self.importance_threshold)],
            maxlen=self.context_items.maxlen
        )
```

#### 长对话历史管理

```python
class ConversationHistory:
    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.history: List[Dict[str, Any]] = []
        self.summary: Optional[str] = None
        
    async def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """添加新消息到历史记录"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }
        self.history.append(message)
        
        # 如果超出令牌限制，