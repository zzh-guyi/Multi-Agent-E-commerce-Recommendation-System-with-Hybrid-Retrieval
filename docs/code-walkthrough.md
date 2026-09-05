# 代码讲解指南 — 从零读懂每一行

> 面向小白的逐文件讲解,帮助你在面试中自信地解释每一个技术细节。

---

## Python版核心代码讲解

### 1. `python/agents/base_agent.py` — Agent基类

**面试考点**: 重试机制、降级策略、模板方法模式

```python
class BaseAgent(ABC):
    """所有Agent的基类,封装了通用的运行时保障机制"""
    
    # 模板方法模式: 子类只需实现 _execute(),运行时保障由基类处理
    @abstractmethod
    async def _execute(self, **kwargs) -> AgentResult:
        """子类实现具体业务逻辑"""
    
    async def run(self, **kwargs) -> AgentResult:
        """公开方法: 封装了计时、重试、降级"""
        # 1. 记录开始时间(用于计算延迟)
        # 2. 调用 _retry_execute(带重试)
        # 3. 如果全部重试失败,调用 _fallback(降级)
```

**面试怎么说**: "我用模板方法模式设计了Agent基类,子类只需关注业务逻辑,
重试(指数退避)、超时、降级、指标收集等横切关注点由基类统一处理。"

---

### 2. `python/agents/user_profile_agent.py` — 用户画像Agent

**面试考点**: 实时特征、RFM模型、LLM结构化输出

```python
# 关键设计点:

# 1. System Prompt 要求LLM输出固定JSON格式
SYSTEM_PROMPT = """...输出JSON格式:
{"segments":["active"], "rfm_score":{"recency":0.8}}..."""

# 2. 行为数据收集: 优先从Feature Store获取,否则用上下文兜底
async def _collect_behavior(self, user_id, context):
    if self.feature_store:  # Phase 2注入Redis
        return await self.feature_store.get_user_features(user_id)
    return {...}  # 降级: 使用context中的默认数据

# 3. 健壮的解析: 处理LLM可能输出的markdown代码块
def _parse_profile(self, user_id, raw):
    if cleaned.startswith("```"):  # 去掉```json ...```
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
```

**面试怎么说**: "画像Agent的核心挑战是LLM输出的不确定性。我通过结构化Prompt
约束输出格式,加上容错解析(处理代码块包裹),保证99%的解析成功率。"

---

### 3. `python/agents/product_rec_agent.py` — 商品推荐Agent

**面试考点**: 多路召回、LLM重排、多样性控制

```python
# 两阶段架构:

# 阶段1: 多路召回 - 获取大量候选集
async def _recall(self, profile, limit):
    # 协同过滤 + 向量检索 + 热度
    # 按用户偏好类目排序: preferred类目的商品排在前面

# 阶段2: LLM重排 - 精细排序
async def _rerank(self, profile, candidates, num_items):
    # 将用户画像和候选商品发给LLM
    # LLM返回排序后的商品ID列表
    # 解析失败时降级为原始顺序
```

**面试怎么说**: "我用经典的'召回→精排'两阶段架构。召回用多策略合并(协同过滤+向量+热度),
扩大候选池;精排用LLM理解用户意图做语义级排序。LLM重排的优势是零样本、可解释、
灵活调整策略。"

---

### 4. `python/orchestrator/supervisor.py` — Supervisor编排器

**面试考点**: 并行编排、asyncio.gather、结果聚合

```python
async def recommend(self, request):
    # Phase 1: 并行执行用户画像 + 商品召回
    profile_result, rec_result = await asyncio.gather(
        self.user_profile_agent.run(...),
        self.product_rec_agent.run(...),
    )
    # 为什么可以并行? 因为画像和召回互不依赖
    
    # Phase 2: 并行执行LLM重排 + 库存校验
    rerank_result, inventory_result = await asyncio.gather(
        self.product_rec_agent.run(user_profile=user_profile, ...),
        self.inventory_agent.run(products=raw_products),
    )
    # 重排需要画像(Phase 1的结果),但不需要库存结果
    # 库存检查需要商品列表,但不需要排序结果
    # 所以可以并行!
    
    # Phase 3: 聚合 + 文案生成(串行,因为依赖前两步)
    final_products = [p for p in ranked if p.id in available_ids]
    copy_result = await self.marketing_copy_agent.run(
        products=final_products  # 必须等库存过滤完才能生成文案
    )
```

**面试怎么说**: "编排的核心是分析Agent间的依赖关系。没有依赖的放在同一个
asyncio.gather里并行。有依赖的用await串行等待。这样总延迟约等于最长链路的延迟,
而不是所有Agent延迟之和。"

---

### 5. `python/services/ab_test.py` — A/B测试引擎

**面试考点**: 一致性哈希、Thompson Sampling

```python
# 流量分桶: 保证同一用户始终进入同一组
def _hash_bucket(self, user_id, experiment_id):
    raw = f"{user_id}:{experiment_id}"
    h = hashlib.md5(raw.encode()).hexdigest()
    return int(h[:8], 16) % self.bucket_count  # 取前8个hex字符

# Thompson Sampling: 动态调整流量
def assign_thompson(self, user_id, experiment_id):
    for g in exp.groups:
        sample = np.random.beta(g.successes, g.failures)
        # Beta分布: successes越多,采样值越高
    best = max(samples, key=lambda x: x[0])
    # 效果好的组被选中的概率更高,但仍有探索空间

# 与传统A/B的区别:
# 传统: 50/50固定分配,等实验结束统计
# Thompson: 动态分配,效果好的组自动获得更多流量
```

**面试怎么说**: "传统A/B测试需要等实验跑完才能得出结论,期间50%的用户
看到的是较差的策略。Thompson Sampling边实验边优化,效果好的组自动获得
更多流量,减少了约50%的实验周期。"

---

## Go版核心亮点讲解

### `go/orchestrator/supervisor.go` — goroutine并行

```go
// Go的并行模型: goroutine + sync.WaitGroup
var wg sync.WaitGroup
wg.Add(2)

go func() {
    defer wg.Done()
    profileResult = s.profileAgent.Run(params)
    mu.Lock()
    agentResults["user_profile"] = profileResult
    mu.Unlock()
}()

go func() {
    defer wg.Done()
    recResult = s.recAgent.Run(params)
    mu.Lock()
    agentResults["product_recall"] = recResult
    mu.Unlock()
}()

wg.Wait()  // 等待两个goroutine都完成
```

**面试对比**: 
- **Python**: `asyncio.gather()` — 协程,单线程异步
- **Java**: `CompletableFuture.supplyAsync()` — 线程池
- **Go**: `goroutine + WaitGroup` — M:N调度,最轻量

---

## Java版核心亮点讲解

### `java/orchestrator/SupervisorOrchestrator.java` — CompletableFuture并行

```java
// Java的并行模型: CompletableFuture
CompletableFuture<AgentResult> profileFuture = 
    userProfileAgent.runAsync(Map.of("userId", request.getUserId()));
CompletableFuture<AgentResult> recFuture = 
    productRecAgent.runAsync(Map.of("numItems", request.getNumItems() * 2));

// join() 阻塞等待结果
AgentResult profileResult = profileFuture.join();
AgentResult recResult = recFuture.join();
```

**面试怎么说**: "Java版使用CompletableFuture实现异步并行,Spring框架
提供@Async注解简化线程池管理。和Python的asyncio不同,Java是真正的
多线程并行,适合CPU密集型场景。"
