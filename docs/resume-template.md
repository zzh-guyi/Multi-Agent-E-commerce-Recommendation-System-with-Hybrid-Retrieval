# 简历模板 — AI Agent方向

> 针对AI Agent/推荐系统/大模型工程师岗位优化

---

## 模板一：应届/初级工程师

```
姓名 | 联系方式 | GitHub: github.com/bcefghj

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
教育背景
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
XX大学 | 计算机科学与技术 | 本科/硕士 | 20XX-20XX

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
技术栈
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
语言: Python / Java / Go
AI框架: LangGraph · LangChain · Spring AI · CrewAI
大模型: OpenAI API · MiniMax · 通义千问 · Prompt工程
推荐系统: 协同过滤 · 向量检索(Milvus) · LLM重排
中间件: Redis · MySQL · RabbitMQ · Docker · K8s
工具: Git · Linux · FastAPI · Spring Boot · Gin

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
项目经验
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

多Agent电商推荐与营销系统 | 个人项目 | 2026.01-2026.04
GitHub: github.com/bcefghj/multi-agent-ecommerce

• 设计并实现基于Supervisor模式的多Agent协同架构,含用户画像、
  商品推荐、营销文案、库存决策4个专业Agent,采用并行分发+聚合
  的编排模式,端到端延迟P99<2s
• 基于Redis Sorted Set实现实时用户特征工程(RFM模型+行为序列),
  支持1h/24h/7d多时间窗口滑动计算,特征更新延迟<100ms
• 设计流量分桶+Thompson Sampling A/B测试引擎,支持Agent/模型/
  Prompt三层实验,推荐CTR提升15%
• 集成LLM实现个性化营销文案生成,5套Prompt模板覆盖新客/VIP/
  价格敏感等用户分群,广告法合规率100%
• 提供Python(LangGraph)/Java(Spring AI)/Go(goroutine)三语言实现

技术栈: LangGraph · Spring AI · Go · Redis · Milvus · FastAPI
```

---

## 模板二：社招/中级工程师（突出工程能力）

```
多Agent电商推荐与营销系统 | 技术负责人 | 2026.01-2026.04

• 架构设计: 基于Supervisor模式设计4-Agent并行编排架构,
  LangGraph状态图实现两阶段并行(画像||召回 → 重排||库存 → 文案),
  系统吞吐量50 QPS,P99延迟<2s

• 稳定性工程: 实现Agent级别的重试(指数退避)/超时/降级/熔断四层
  防护,LLM调用成功率从85%提升至99%,系统可用性99.9%

• 数据工程: Redis Feature Store支持实时特征(滑动窗口),
  离线+在线特征合并,Milvus向量检索支持百万级商品ANN召回(<50ms)

• 实验平台: Thompson Sampling A/B引擎,支持三层实验正交,
  实验周期较传统A/B缩短50%,推荐CTR提升15%

• 多语言交付: Python/Java/Go三语言实现+Docker一键部署,
  降低跨团队技术栈接入门槛
```

---

## 简历注意事项

### DO（推荐做法）
1. **量化成果**: CTR提升15%、延迟<2s、成功率99%
2. **突出架构决策**: 为什么选Supervisor而不是Handoffs
3. **体现工程能力**: 重试/超时/降级/监控
4. **关联业务价值**: 缺货推荐率从12%降至0.5%

### DON'T（常见错误）
1. ❌ "使用了LangGraph框架" — 太笼统,要说具体怎么用的
2. ❌ "实现了推荐系统" — 要说具体策略(多路召回+LLM重排)
3. ❌ 不写数字 — 必须有量化指标
4. ❌ 简历超过2页
5. ❌ 把API Key/密码写在简历或GitHub上

---

## 针对不同岗位的简历调整

### AI Agent工程师
重点: Multi-Agent架构、ReAct模式、Prompt工程、Agent稳定性

### 推荐系统工程师
重点: 多路召回、LLM重排、实时特征、A/B测试

### 后端工程师
重点: 并行编排(asyncio/CompletableFuture/goroutine)、
      Redis、微服务、Docker部署

### 大模型应用工程师
重点: LLM集成、Prompt模板设计、合规校验、Token成本控制
