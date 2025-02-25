## 第二阶段：MCP 实践应用（续）

### 3. 上下文管理（续）

#### 长对话历史管理（续）

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
        
        # 如果超出令牌限制，进行压缩
        if len(self.history) > 10:  # 简单的触发条件示例
            await self._compress_history()
            
    async def _compress_history(self):
        """压缩历史记录，保持重要信息"""
        if not self.history:
            return
            
        # 提取关键消息
        key_messages = []
        current_summary = []
        
        for msg in self.history:
            if self._is_key_message(msg):
                key_messages.append(msg)
            else:
                current_summary.append(msg["content"])
                
        # 更新摘要
        if current_summary:
            self.summary = f"Previous discussion summary: {' '.join(current_summary)}"
            
        # 重置历史记录
        self.history = (
            [{"role": "system", "content": self.summary}] +
            key_messages[-5:]  # 保留最后5条关键消息
        )
        
    def _is_key_message(self, message: Dict[str, Any]) -> bool:
        """判断是否为关键消息"""
        # 示例判断逻辑
        importance_markers = [
            "结论", "总结", "决定", "重要", "关键",
            "conclusion", "summary", "decision", "important"
        ]
        
        content = message["content"].lower()
        return any(marker in content for marker in importance_markers)
```

#### 内存优化管理器

```python
class MemoryOptimizer:
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.cache_size: int = 0
        self.max_cache_size: int = 1024 * 1024 * 100  # 100MB
        self.last_access: Dict[str, datetime] = {}
        
    async def add_to_cache(self, key: str, value: Any):
        """添加项目到缓存"""
        value_size = self._estimate_size(value)
        
        # 如果需要，清理缓存空间
        while self.cache_size + value_size > self.max_cache_size:
            await self._evict_least_used()
            
        self.cache[key] = value
        self.last_access[key] = datetime.now()
        self.cache_size += value_size
        
    async def get_from_cache(self, key: str) -> Optional[Any]:
        """从缓存获取项目"""
        if key in self.cache:
            self.last_access[key] = datetime.now()
            return self.cache[key]
        return None
        
    async def _evict_least_used(self):
        """清除最少使用的缓存项"""
        if not self.last_access:
            return
            
        oldest_key = min(
            self.last_access.items(),
            key=lambda x: x[1]
        )[0]
        
        value_size = self._estimate_size(self.cache[oldest_key])
        del self.cache[oldest_key]
        del self.last_access[oldest_key]
        self.cache_size -= value_size
        
    def _estimate_size(self, obj: Any) -> int:
        """估算对象大小"""
        return len(json.dumps(obj)) if isinstance(obj, (dict, list)) else 1024
```

#### 上下文管理整合

```python
class ContextManager:
    def __init__(self):
        self.dynamic_context = DynamicContext()
        self.conversation_history = ConversationHistory()
        self.memory_optimizer = MemoryOptimizer()
        
    async def process_interaction(
        self,
        user_input: str,
        tool_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        # 添加用户输入
        await self.dynamic_context.add_item(
            content=user_input,
            type="user_input"
        )
        
        # 添加到对话历史
        await self.conversation_history.add_message(
            role="user",
            content=user_input
        )
        
        # 处理工具结果
        if tool_results:
            await self.dynamic_context.add_item(
                content=tool_results,
                type="tool_result"
            )
            
        # 获取当前上下文
        current_context = self._prepare_context()
        
        # 缓存处理结果
        cache_key = f"context_{datetime.now().timestamp()}"
        await self.memory_optimizer.add_to_cache(cache_key, current_context)
        
        return current_context
        
    def _prepare_context(self) -> Dict[str, Any]:
        """准备当前上下文"""
        context = {
            "timestamp": datetime.now().isoformat(),
            "dynamic_items": [
                item.to_dict() for item in self.dynamic_context.context_items
            ],
            "conversation_summary": self.conversation_history.summary,
            "recent_messages": self.conversation_history.history[-5:]
        }
        return context
```

#### 高级上下文管理技术

##### 会话分支管理

```python
class ConversationBranchManager:
    def __init__(self):
        self.branches: Dict[str, List[Dict]] = {}
        self.active_branch: str = "main"
        
    async def create_branch(self, branch_name: str, from_branch: str = "main"):
        """创建新的对话分支"""
        if from_branch in self.branches:
            # 从现有分支复制上下文
            self.branches[branch_name] = self.branches[from_branch].copy()
        else:
            self.branches[branch_name] = []
            
    async def switch_branch(self, branch_name: str):
        """切换到指定分支"""
        if branch_name in self.branches:
            self.active_branch = branch_name
        else:
            raise ValueError(f"Branch {branch_name} does not exist")
            
    async def merge_branches(self, source_branch: str, target_branch: str):
        """合并两个分支的上下文"""
        if source_branch in self.branches and target_branch in self.branches:
            # 智能合并，避免重复信息
            merged_context = self._smart_merge(
                self.branches[source_branch],
                self.branches[target_branch]
            )
            self.branches[target_branch] = merged_context
```

##### 上下文优先级管理

```python
class PriorityContextManager:
    def __init__(self):
        self.priority_levels = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.5,
            "low": 0.3
        }
        self.context_items: List[Dict] = []
        
    async def add_item_with_priority(
        self,
        content: Any,
        priority: str = "medium",
        lifetime: Optional[timedelta] = None
    ):
        """添加带优先级的上下文项"""
        expiry = datetime.now() + lifetime if lifetime else None
        
        item = {
            "content": content,
            "priority": self.priority_levels.get(priority, 0.5),
            "created_at": datetime.now(),
            "expires_at": expiry
        }
        
        self.context_items.append(item)
        await self._rebalance_context()
        
    async def _rebalance_context(self):
        """重新平衡上下文，确保高优先级信息得到保留"""
        current_time = datetime.now()
        
        # 清理过期项目
        self.context_items = [
            item for item in self.context_items
            if not item["expires_at"] or item["expires_at"] > current_time
        ]
        
        # 按优先级排序
        self.context_items.sort(key=lambda x: x["priority"], reverse=True)
```

## 第三阶段：高级应用开发

### 1. Agent 开发

#### Agent 基础架构

```python
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import asyncio
from datetime import datetime

class BaseAgent(ABC):
    def __init__(self, name: str, capabilities: List[str]):
        self.name = name
        self.capabilities = capabilities
        self.state = "idle"
        self.task_queue = asyncio.Queue()
        
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """处理任务的主要流程"""
        try:
            self.state = "processing"
            
            # 任务分析
            task_plan = await self.analyze_task(task)
            
            # 执行计划
            result = await self.execute_plan(task_plan)
            
            self.state = "idle"
            return result
        except Exception as e:
            self.state = "error"
            return {"status": "error", "error": str(e)}
            
    @abstractmethod
    async def analyze_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分析任务并创建执行计划"""
        pass
        
    @abstractmethod
    async def execute_plan(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """执行任务计划"""
        pass
```

#### 多工具协调系统

```python
class ToolCoordinator:
    def __init__(self):
        self.tools: Dict[str, Any] = {}
        self.tool_dependencies: Dict[str, List[str]] = {}
        
    def register_tool(self, tool_name: str, tool_instance: Any, dependencies: List[str] = None):
        """注册工具及其依赖"""
        self.tools[tool_name] = tool_instance
        self.tool_dependencies[tool_name] = dependencies or []
        
    async def execute_tool_chain(self, chain: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行工具链"""
        results = []
        for step in chain:
            tool_name = step["tool"]
            if tool_name in self.tools:
                # 检查依赖是否满足
                if await self._check_dependencies(tool_name):
                    result = await self.tools[tool_name].execute(step["parameters"])
                    results.append({
                        "tool": tool_name,
                        "result": result,
                        "timestamp": datetime.now()
                    })
        return results
        
    async def _check_dependencies(self, tool_name: str) -> bool:
        """检查工具依赖是否满足"""
        dependencies = self.tool_dependencies[tool_name]
        return all(dep in self.tools for dep in dependencies)
```

#### 智能 Agent 实现

```python
class SmartAgent(BaseAgent):
    def __init__(self, name: str, capabilities: List[str]):
        super().__init__(name, capabilities)
        self.tool_coordinator = ToolCoordinator()
        self.memory_manager = MemoryManager()
        self.task_planner = TaskPlanner()
        
    async def analyze_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """智能分析任务"""
        # 理解任务需求
        task_understanding = await self._understand_task(task)
        
        # 创建执行计划
        execution_plan = await self.task_planner.create_plan(task_understanding)
        
        # 验证计划可行性
        validated_plan = await self._validate_plan(execution_plan)
        
        return validated_plan
        
    async def execute_plan(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """执行任务计划"""
        results = []
        
        for step in plan:
            # 执行单个步骤
            step_result = await self._execute_step(step)
            results.append(step_result)
            
            # 更新内存
            await self.memory_manager.update(step_result)
            
            # 检查是否需要调整计划
            if await self._should_adjust_plan(results):
                plan = await self._adjust_plan(plan, results)
                
        return self._synthesize_results(results)
```

#### 任务规划器

```python
class TaskPlanner:
    def __init__(self):
        self.planning_strategies = {
            "sequential": self._create_sequential_plan,
            "parallel": self._create_parallel_plan,
            "conditional": self._create_conditional_plan
        }
        
    async def create_plan(self, task_understanding: Dict[str, Any]) -> List[Dict[str, Any]]:
        """创建任务执行计划"""
        strategy = self._select_planning_strategy(task_understanding)
        plan = await self.planning_strategies[strategy](task_understanding)
        return plan
        
    async def _create_sequential_plan(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """创建顺序执行计划"""
        plan = []
        for step in task["steps"]:
            plan.append({
                "type": "sequential",
                "tool": step["tool"],
                "parameters": step["parameters"],
                "validation": step.get("validation", {})
            })
        return plan
        
    async def _create_parallel_plan(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """创建并行执行计划"""
        parallel_steps = []
        for step in task["steps"]:
            if step.get("can_parallel", False):
                parallel_steps.append({
                    "type": "parallel",
                    "tool": step["tool"],
                    "parameters": step["parameters"]
                })
        return parallel_steps
```

#### 应用示例

```python
class DataProcessingAgent(SmartAgent):
    def __init__(self):
        super().__init__(
            name="data_processor",
            capabilities=["data_analysis", "visualization", "reporting"]
        )
        
        # 注册工具
        self.tool_coordinator.register_tool(
            "data_analyzer",
            DataAnalyzer(),
            dependencies=[]
        )
        self.tool_coordinator.register_tool(
            "visualizer",
            Visualizer(),
            dependencies=["data_analyzer"]
        )
        
    async def process_data_task(self, data: Any, requirements: Dict[str, Any]):
        """处理数据任务"""
        task = {
            "type": "data_processing",
            "data": data,
            "requirements": requirements
        }
        
        # 创建执行计划
        plan = await self.analyze_task(task)
        
        # 执行计划
        results = await self.execute_plan(plan)
        
        # 生成报告
        report = await self._generate_report(results)
        
        return report
```

### 2. 系统集成

#### 业务系统集成

```python
from typing import Dict, Any
import aiohttp
import jwt
from datetime import datetime, timedelta
import logging
import prometheus_client as prom

class SystemIntegrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.systems = {}
        self.session = aiohttp.ClientSession()
        
    async def register_system(self, system_name: str, system_config: Dict[str, Any]):
        """注册业务系统"""
        adapter = self._create_system_adapter(system_config)
        self.systems[system_name] = adapter
        
    async def connect(self, system_name: str) -> Dict[str, Any]:
        """连接到业务系统"""
        if system_name not in self.systems:
            raise ValueError(f"System {system_name} not registered")
            
        adapter = self.systems[system_name]
        return await adapter.connect()
        
    def _create_system_adapter(self, config: Dict[str, Any]) -> 'SystemAdapter':
        """创建系统适配器"""
        adapter_type = config.get("type", "rest")
        if adapter_type == "rest":
            return RESTAdapter(config)
        elif adapter_type == "grpc":
            return GRPCAdapter(config)
        elif adapter_type == "graphql":
            return GraphQLAdapter(config)
```

#### 安全认证实现

```python
class SecurityManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.tokens = {}
        
    async def generate_token(self, user_id: str, permissions: List[str]) -> str:
        """生成 JWT token"""
        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        return token
        
    async def validate_token(self, token: str) -> Dict[str, Any]:
        """验证 token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise SecurityError("Token expired")
        except jwt.InvalidTokenError:
            raise SecurityError("Invalid token")

class AuthMiddleware:
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        
    async def authenticate(self, request: Request) -> Dict[str, Any]:
        """认证请求"""
        token = request.headers.get("Authorization")
        if not token:
            raise SecurityError("No token provided")
            
        token = token.replace("Bearer ", "")
        return await self.security_manager.validate_token(token)
```

#### 监控和日志系统

```python
class MonitoringSystem:
    def __init__(self):
        # Prometheus metrics
        self.request_counter = prom.Counter(
            'mcp_requests_total',
            'Total requests processed',
            ['system', 'endpoint']
        )
        self.latency_histogram = prom.Histogram(
            'mcp_request_latency_seconds',
            'Request latency in seconds',
            ['system', 'endpoint']
        )
        self.error_counter = prom.Counter(
            'mcp_errors_total',
            'Total errors encountered',
            ['system', 'error_type']
        )
        
    async def record_request(self, system: str, endpoint: str):
        """记录请求"""
        self.request_counter.labels(system=system, endpoint=endpoint).inc()
        
    async def record_latency(self, system: str, endpoint: str, duration: float):
        """记录延迟"""
        self.latency_histogram.labels(
            system=system,
            endpoint=endpoint
        ).observe(duration)
        
    async def record_error(self, system: str, error_type: str):
        """记录错误"""
        self.error_counter.labels(
            system=system,
            error_type=error_type
        ).inc()

class LoggingSystem:
    def __init__(self):
        self.logger = logging.getLogger("mcp")
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志系统"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 文件处理器
        file_handler = logging.FileHandler('mcp.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
```

#### 集成示例

```python
class MCPSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.integrator = SystemIntegrator(config.get("systems", {}))
        self.security = SecurityManager(config["security_key"])
        self.monitoring = MonitoringSystem()
        self.logging = LoggingSystem()
        
    async def initialize(self):
        """初始化系统"""
        # 注册业务系统
        for system_name, system_config in self.config["systems"].items():
            await self.integrator.register_system(system_name, system_config)
            
        # 设置监控
        self.start_monitoring()
        
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理请求"""
        start_time = datetime.now()
        system_name = request.get("system")
        
        try:
            # 认证
            await self.security.validate_token(request.get("token"))
            
            # 执行请求
            result = await self.integrator.connect(system_name)
            
            # 记录指标
            duration = (datetime.now() - start_time).total_seconds()
            await self.monitoring.record_latency(
                system_name,
                request.get("endpoint"),
                duration
            )
            
            return result
            
        except Exception as e:
            # 记录错误
            self.logging.logger.error(f"Error processing request: {str(e)}")
            await self.monitoring.record_error(system_name, type(e).__name__)
            raise
```

### 3. 性能优化

#### 响应时间优化

```python
from typing import Dict, Any, List
import asyncio
from functools import lru_cache
import aiocache
from collections import deque

class ResponseOptimizer:
    def __init__(self):
        # 配置缓存
        self.cache = aiocache.Cache(aiocache.SimpleMemoryCache)
        self.request_queue = deque(maxlen=1000)
        
    @lru_cache(maxsize=100)
    async def optimize_request(self, request: Dict[str, Any]):
        """优化请求处理"""
        # 预处理请求
        processed_request = await self._preprocess_request(request)
        return processed_request
    
    async def _preprocess_request(self, request: Dict[str, Any]):
        """请求预处理"""
        # 移除不必要的字段
        essential_fields = self._extract_essential_fields(request)
        # 压缩请求数据
        compressed_data = self._compress_request(essential_fields)
        return compressed_data

class ResponseTimeManager:
    def __init__(self):
        self.latency_threshold = 500  # 毫秒
        self.response_times = []
        
    async def track_response_time(self, start_time: float) -> float:
        """跟踪响应时间"""
        elapsed = (time.time() - start_time) * 1000
        self.response_times.append(elapsed)
        
        if elapsed > self.latency_threshold:
            await self._handle_slow_response(elapsed)
        
        return elapsed
```

#### 并发处理实现

```python
class ConcurrencyManager:
    def __init__(self, max_workers: int = 10):
        self.semaphore = asyncio.Semaphore(max_workers)
        self.active_tasks: Dict[str, asyncio.Task] = {}
        
    async def process_concurrent(self, tasks: List[Dict[str, Any]]):
        """并发处理多个任务"""
        async with self.semaphore:
            # 创建任务
            coroutines = [self._process_task(task) for task in tasks]
            # 并发执行
            results = await asyncio.gather(*coroutines, return_exceptions=True)
            return results
            
    async def _process_task(self, task: Dict[str, Any]):
        """处理单个任务"""
        task_id = task.get('id')
        try:
            # 创建新任务
            new_task = asyncio.create_task(self._execute_task(task))
            self.active_tasks[task_id] = new_task
            result = await new_task
            return result
        finally:
            # 清理完成的任务
            self.active_tasks.pop(task_id, None)

class BatchProcessor:
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.batch_queue = asyncio.Queue()
        
    async def add_to_batch(self, item: Any):
        """添加项目到批处理队列"""
        await self.batch_queue.put(item)
        
        if self.batch_queue.qsize() >= self.batch_size:
            await self.process_batch()
            
    async def process_batch(self):
        """处理一个批次"""
        batch = []
        while not self.batch_queue.empty() and len(batch) < self.batch_size:
            item = await self.batch_queue.get()
            batch.append(item)
            
        if batch:
            await self._execute_batch(batch)
```

#### 资源管理

```python
class ResourceManager:
    def __init__(self):
        self.memory_limit = 1024 * 1024 * 1024  # 1GB
        self.cpu_limit = 0.8  # 80% CPU 使用率上限
        self.active_connections = 0
        self.max_connections = 100
        
    async def monitor_resources(self):
        """监控资源使用"""
        while True:
            memory_usage = self._get_memory_usage()
            cpu_usage = self._get_cpu_usage()
            
            if memory_usage > self.memory_limit:
                await self._handle_memory_pressure()
            
            if cpu_usage > self.cpu_limit:
                await self._handle_cpu_pressure()
                
            await asyncio.sleep(5)  # 每5秒检查一次
            
    async def acquire_connection(self):
        """获取连接"""
        if self.active_connections >= self.max_connections:
            await self._wait_for_connection()
            
        self.active_connections += 1
        return ConnectionContext(self)

class ConnectionPool:
    def __init__(self, min_size: int = 5, max_size: int = 20):
        self.min_size = min_size
        self.max_size = max_size
        self.pool = asyncio.Queue()
        self.size = 0
        
    async def get_connection(self):
        """获取连接"""
        if self.pool.empty() and self.size < self.max_size:
            # 创建新连接
            conn = await self._create_connection()
            self.size += 1
            return conn
            
        return await self.pool.get()
        
    async def release_connection(self, conn):
        """释放连接"""
        if self.size > self.min_size and await self._is_idle(conn):
            # 关闭多余的空闲连接
            await self._close_connection(conn)
            self.size -= 1
        else:
            await self.pool.put(conn)
```

#### 性能优化整合

```python
class PerformanceOptimizer:
    def __init__(self):
        self.response_optimizer = ResponseOptimizer()
        self.concurrency_manager = ConcurrencyManager()
        self.resource_manager = ResourceManager()
        self.batch_processor = BatchProcessor()
        
    async def optimize_request_handling(self, request: Dict[str, Any]):
        """优化请求处理"""
        start_time = time.time()
        
        try:
            # 获取资源
            async with self.resource_manager.acquire_connection():
                # 优化请求
                optimized_request = await self.response_optimizer.optimize_request(request)
                
                # 处理请求
                if self._should_batch(request):
                    await self.batch_processor.add_to_batch(optimized_request)
                    result = await self.batch_processor.get_result()
                else:
                    result = await self.concurrency_manager.process_concurrent([optimized_request])
                
                return result
        finally:
            elapsed = await self.response_optimizer.track_response_time(start_time)
```

## 开发资源

### 官方资源

为了获取最新和最准确的 MCP 资源，建议访问以下官方渠道：

1. **Anthropic MCP 协议文档**
   - 访问 Anthropic 的官方开发者文档：https://docs.anthropic.com
   - 在文档中查找 MCP (Model Context Protocol) 相关章节
   - 关注协议的最新更新和变化

2. **Claude API 文档**
   - Claude API 文档地址：https://docs.anthropic.com/claude/
   - 包含了详细的 API 参考和使用指南
   - 提供了不同 API 版本的说明

3. **官方示例代码和教程**
   - 访问 Anthropic 的 GitHub 仓库
   - 查看官方博客获取最新教程
   - 参考开发者社区的讨论

### 开发工具

#### MCP SDK 和客户端库设置

```python
# 安装必要的依赖
'''
pip install anthropic    # Anthropic 官方 Python SDK
pip install pytest      # 测试框架
pip install aiohttp     # 异步 HTTP 客户端
pip install python-dotenv  # 环境变量管理
'''

# 基本配置示例
from anthropic import Anthropic
import os
from dotenv import load_dotenv

load_dotenv()  # 加载环境变量

class MCPClient:
    def __init__(self):
        self.client = Anthropic(
            api_key=os.getenv("ANTHROPIC_API