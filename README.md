# 基于 Multi-Agent + Hybrid Retrieval 的智能电商推荐系统

基于 **Multi-Agent + LangGraph + ReAct/Tool Calling + Hybrid Retrieval + LLM Rerank** 构建的智能电商推荐系统。通过多个专业 Agent 协同完成用户画像、商品召回、LLM 重排、库存校验和营销文案生成，并由 Supervisor 统一编排推荐流程。

**核心标签**：Python FastAPI LangGraph LangChain Milvus Redis MySQL Docker Hybrid Retrieval RRF LLM Rerank A/B Test

## 📖 目录

1. [这个项目是什么？](#-这个项目是什么)

2. [推荐系统评估体系](#-推荐系统评估体系)

3. [系统架构（看图秒懂）](#-系统架构看图秒懂)

4. [四大核心 Agent 详解](#-四大核心-agent-详解)

5. [关键代码展示](#-关键代码展示)

6. [快速上手运行](#-快速上手运行)

7. [API 接口文档](#-api-接口文档)

8. [项目文件结构](#-项目文件结构)

   

   ------

   ## 🤔 这个项目是什么

   ### 用一句话解释

   > 本项目模拟真实电商推荐场景，构建了一套基于 **Multi-Agent + Hybrid Retrieval + LLM Rerank** 的智能推荐系统，解决用户需求理解、商品召回、智能排序、库存校验以及个性化营销生成等完整推荐链路问题。
   >
   > 不同于传统推荐系统依赖单一召回模型或固定规则，本项目通过 **Supervisor Agent 统一编排多个专业 Agent**，结合 **Hybrid Retrieval（向量检索 + 关键词检索 + RRF 融合）、LLM Rerank、库存 Agent、营销 Agent 以及在线指标监控体系**，实现从用户需求理解到推荐结果生成的端到端智能推荐流程。

   ------

   ## 🎯 解决了什么问题？

   传统电商推荐系统主要存在以下问题：

   | 问题                   | 传统方案                           | 本项目方案                                                   | 验证指标                         |
   | :--------------------- | :--------------------------------- | :----------------------------------------------------------- | :------------------------------- |
   | 🎯 用户需求理解不足     | 依赖关键词匹配，难以理解复杂语义   | **Hybrid Retrieval**：Dense Vector Search + Keyword Search，通过 RRF 融合语义和关键词信息 | Recall@K、Precision@K、MRR、NDCG |
   | 🔍 商品召回覆盖不足     | 单一向量检索可能遗漏关键词相关商品 | **Milvus 向量检索 + Keyword Retrieval 双路召回**，提升候选覆盖能力 | Recall@5、Recall@10              |
   | 📊 召回结果排序不准确   | 召回后直接按照相似度排序           | **LLM Rerank** 对候选商品进行语义理解和重新排序              | NDCG@K、MRR@K、Ranking 提升      |
   | 📦 推荐结果缺少业务约束 | 推荐系统不了解实时库存状态         | **InventoryAgent** 实时检查库存，过滤不可售商品              | 推荐可用率、库存过滤成功率       |
   | ✍️ 推荐内容缺少个性化   | 固定模板生成营销文案               | **MarketingCopyAgent** 根据用户画像生成个性化营销内容        | 文案生成成功率、Agent 调用成功率 |
   | 🤖 多智能体协作复杂     | 各模块独立运行，缺少统一流程管理   | **Supervisor + LangGraph** 管理 Agent 调度和状态流转         | Agent 成功率、平均响应延迟       |
   | 📈 缺少线上效果反馈     | 无法评估推荐是否有效               | **MetricsCollector + A/B Test Engine** 记录曝光、点击、购买行为 | CTR、CVR、GMV、实验指标          |

   ------

   ## 📊 推荐系统评估体系

   为了验证系统效果，本项目构建了一套完整的 Evaluation Framework，从 **Retrieval 离线效果、LLM Rerank 排序效果、Agent 运行质量以及在线业务指标** 等多个维度进行评估。

   ### 1. 离线 Retrieval 评估

   针对 Hybrid Retrieval 和 RRF 召回效果，构建测试 Query 集，对候选商品排序质量进行评估：

   | 指标            | 作用                           |
   | :-------------- | :----------------------------- |
   | **Recall@K**    | 衡量目标商品是否被召回         |
   | **Precision@K** | 衡量 Top-K 推荐结果准确性      |
   | **MRR@K**       | 衡量目标商品的排名位置         |
   | **NDCG@K**      | 衡量排序结果的相关性和位置质量 |

   **评估链路**：

   text

   ```
   User Query
       │
       ▼
   Hybrid Retrieval
       │
       ├── Vector Search
       └── Keyword Search
       │
       ▼
   RRF Fusion
       │
       ▼
   Evaluation Metrics
       │
       ├── Recall@K
       ├── Precision@K
       ├── MRR@K
       └── NDCG@K
   ```

   

   #### Hybrid Retrieval 对比实验

   使用离线 Evaluation Dataset 对 Keyword、Vector 和 Hybrid（RRF）三种召回策略进行对比：

   | Strategy         | Recall@10  | Precision@10 | MRR@10     | NDCG@10    | Avg Latency |
   | :--------------- | :--------- | :----------- | :--------- | :--------- | :---------- |
   | **Keyword**      | 0.9367     | **0.4000**   | 0.9375     | 0.8935     | **3.0 ms**  |
   | **Vector**       | **0.9648** | 0.3950       | 0.9750     | **0.9233** | 91.4 ms     |
   | **Hybrid (RRF)** | 0.9568     | 0.3950       | **1.0000** | 0.9212     | 96.6 ms     |

   **关键发现**：

   - **Vector Retrieval** 的 Recall@10 达到 **0.9648**，三种策略中召回覆盖率最高。
   - **Hybrid (RRF)** 的 MRR@10 达到 **1.0000**，说明相关商品具有较好的前排位置表现。
   - **Hybrid (RRF)** 的 NDCG@10 达到 **0.9212**，与 Vector Retrieval 的 **0.9233** 接近。
   - **Keyword Retrieval** 平均延迟仅 **3.0 ms**，具有明显的低延迟优势。
   - **Hybrid Retrieval** 通过融合语义检索与关键词检索结果，在保持较高召回能力的同时改善相关商品的前排排序表现。

   ------

   ### 2. LLM Rerank 离线评估

   在 Hybrid Retrieval 召回候选商品后，引入 **LLM Rerank** 对候选商品进行语义重排序，并与不经过 Rerank 的原始排序结果进行对比。

   #### 评估指标

   | 指标            | 作用                                |
   | :-------------- | :---------------------------------- |
   | **Recall@5**    | 衡量 Top-5 结果对相关商品的覆盖能力 |
   | **MRR@5**       | 衡量 Top-5 中首个相关商品的排名质量 |
   | **NDCG@5**      | 衡量 Top-5 综合排序质量             |
   | **Precision@5** | 衡量 Top-5 推荐结果准确率           |
   | **Avg Latency** | 衡量 LLM Rerank 平均耗时            |

   #### LLM Rerank 对比实验

   | Strategy       | Recall@5   | MRR@5      | NDCG@5     | Precision@5 | Avg Latency  |
   | :------------- | :--------- | :--------- | :--------- | :---------- | :----------- |
   | **LLM Rerank** | **0.8000** | **0.9700** | **0.8600** | **0.6400**  | 23084.7 ms   |
   | **No Rerank**  | 0.7821     | 0.9500     | 0.8404     | 0.6200      | **336.7 ms** |

   #### Rerank 效果

   相比 No Rerank，LLM Rerank 在当前 Evaluation Dataset 上取得以下提升：

   | Metric          | No Rerank | LLM Rerank | Improvement |
   | :-------------- | :-------- | :--------- | :---------- |
   | **Recall@5**    | 0.7821    | **0.8000** | **+2.29%**  |
   | **MRR@5**       | 0.9500    | **0.9700** | **+2.11%**  |
   | **NDCG@5**      | 0.8404    | **0.8600** | **+2.33%**  |
   | **Precision@5** | 0.6200    | **0.6400** | **+3.23%**  |

   > Improvement 按 `(LLM Rerank - No Rerank) / No Rerank` 计算。

   #### 实验结论

   - LLM Rerank 在 Recall@5、MRR@5、NDCG@5 和 Precision@5 四项指标上均取得提升。
   - **NDCG@5 从 0.8404 提升至 0.8600**，说明整体排序质量得到改善。
   - **Precision@5 从 0.6200 提升至 0.6400**，说明 Top-5 推荐结果的相关性进一步提高。
   - 主要代价是推理延迟明显增加：**23.1s vs 336.7ms**，因此在实际系统中需要结合超时控制、重试和降级策略进行权衡。
   - 系统实现了 **Timeout + Retry + Fallback** 机制，在 Rerank 服务异常或超时时返回原始排序结果，避免单次 LLM 调用阻塞整个推荐链路。

   ------

   ### 3. Agent 系统监控指标

   针对 Multi-Agent 协作流程，系统增加运行时 Metrics 监控。

   | 指标                 | 描述                  |
   | :------------------- | :-------------------- |
   | **Agent Call Count** | Agent 调用次数        |
   | **Success Rate**     | Agent 执行成功率      |
   | **Average Latency**  | Agent 平均响应耗时    |
   | **Error Count**      | Agent 异常次数        |
   | **Tool Call Count**  | Agent 工具调用次数    |
   | **Fallback Count**   | Rerank 等模块降级次数 |

   **Metrics 监控链路**：

   text

   ```
   Supervisor
       │
       ├── UserProfileAgent
       ├── ProductRecAgent
       ├── InventoryAgent
       └── MarketingCopyAgent
            │
            ▼
       MetricsCollector
            │
            ▼
       /api/v1/metrics
   ```

   

   ------

   ### 4. 推荐业务指标

   模拟真实线上推荐系统，引入用户反馈闭环。

   | 指标         | 描述               | 计算方式                               |
   | :----------- | :----------------- | :------------------------------------- |
   | **Exposure** | 商品曝光次数       | 事件上报                               |
   | **Click**    | 用户点击次数       | 事件上报                               |
   | **CTR**      | 点击率             | `Click / Exposure`                     |
   | **Purchase** | 用户购买次数       | 事件上报                               |
   | **CVR**      | 点击后的购买转化率 | `Purchase / Click`                     |
   | **GMV**      | 商品成交金额       | `Σ(Purchase Quantity × Product Price)` |

   **用户行为闭环**：

   text

   ```
   Recommendation
       │
       ▼
     Exposure
       │
       ▼
      Click
       │
       ▼
     Purchase
       │
       ▼
   CTR / CVR / GMV
   ```

   

   通过以下 API 记录用户行为：

   bash

   ```
   POST /api/v1/events/exposure
   POST /api/v1/events/click
   POST /api/v1/events/purchase
   ```

   

   MetricsCollector 对用户行为数据进行汇总，提供在线业务指标查询能力。

   ------

   ### 5. A/B Test Evaluation

   系统实现基于用户 ID 的流量分组机制，对不同推荐策略进行 A/B Test。

   核心实验指标包括：

   | 指标         | 描述                           |
   | :----------- | :----------------------------- |
   | **CTR**      | 点击率                         |
   | **CVR**      | 转化率                         |
   | **GMV**      | 成交金额                       |
   | **CTR Lift** | 实验组相对对照组的点击率提升   |
   | **CVR Lift** | 实验组相对对照组的转化率提升   |
   | **GMV Lift** | 实验组相对对照组的成交金额提升 |

   **Lift 计算**：

   text

   ```
                 Treatment - Control
   Lift =        --------------------
                     Control
   ```

   

   **整体 Evaluation Framework**：

   text

   ```
             Offline Evaluation
                    │
       ┌────────────┴────────────┐
       ▼                         ▼
   Retrieval Quality       Ranking Quality
       │                         │
       ├── Recall            ├── LLM Rerank
       ├── MRR               └── Evaluation
       └── NDCG
       │                         │
       └────────────┬────────────┘
                    ▼
           Agent Runtime Metrics
                    │
                    ▼
          Online Business Metrics
                    │
              CTR / CVR / GMV
                    │
                    ▼
                A/B Test
   ```

   

   ------

   ## 🏗 系统架构（看图秒懂）

   text

   ```
                           User Query
                               │
                               ▼
                     ┌──────────────────┐
                     │ Supervisor Agent │
                     │    (LangGraph)   │
                     └────────┬─────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
   │ UserProfile     │ │ ProductRec      │ │ MarketingCopy   │
   │ Agent           │ │ Agent           │ │ Agent           │
   │                 │ │                 │ │                 │
   │ 用户画像构建     │ │ 商品召回 + 重排  │ │ 个性化文案生成   │
   └─────────────────┘ └────────┬────────┘ └─────────────────┘
                                │
                                ▼
                      ┌──────────────────┐
                      │ Hybrid Retrieval │
                      └────────┬─────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌─────────────────┐         ┌─────────────────┐
        │ Vector Search   │         │ Keyword Search  │
        │     Milvus      │         │      MySQL      │
        └────────┬────────┘         └────────┬────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                      ┌─────────────────┐
                      │    RRF Fusion   │
                      │      k = 60     │
                      └────────┬────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │   LLM Rerank    │
                      └────────┬────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │ InventoryAgent  │
                      │   库存校验       │
                      └────────┬────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │  Final Result   │
                      │   + Metrics     │
                      └─────────────────┘
   ```

   

   ------

   ## 🤖 四大核心 Agent 详解

   ### 1. UserProfileAgent

   **职责**：基于用户行为数据生成用户画像。

   | 属性     | 说明                                                         |
   | :------- | :----------------------------------------------------------- |
   | **输入** | `user_id`                                                    |
   | **处理** | 查询 MySQL 用户行为表，计算 RFM 特征、7 日行为统计、偏好类目和价格区间 |
   | **输出** | `UserProfile`：偏好类目、用户分群（`new_user` / `active` / `vip`）、行为摘要 |
   | **特点** | 纯规则计算，不依赖 LLM，延迟 < 1ms                           |

   ------

   ### 2. ProductRecAgent（核心 Agent，双模式设计）

   ProductRecAgent 负责商品候选召回和语义重排序，采用 **Recall + Rerank** 两阶段设计。

   #### Mode 1：Recall（ReAct + Tool Calling）

   text

   ```
   LLM
    │
    │ 制定搜索策略
    ▼
   Tool Calling × N 轮
    │
    ▼
   search_products(query, limit)
    │
    │ LangChain Tool
    ▼
   Hybrid Retrieval
    ├── Milvus Vector Search
    ├── MySQL Keyword Search
    └── RRF Fusion
    │
    ▼
   候选商品（10 ~ 20 个）
   ```

   

   **主要特点**：

   - LLM 根据用户需求和用户画像自主决定搜索策略。
   - 支持多轮 Tool Calling。
   - 每次搜索通过 Hybrid Retrieval 获取候选商品。
   - LLM 不直接访问数据库，而是通过 LangChain Tool 间接调用检索能力。

   #### Mode 2：Rerank（LLM 重排序）

   text

   ```
   Recall Candidates
          │
          ▼
   LLM Rerank Prompt
          │
          ├── 用户画像
          ├── 用户需求
          └── 候选商品摘要
          │
          ▼
   LLM 输出商品 ID 排序
          │
          ▼
   Top-K Products
          │
          ▼
   Timeout + Retry + Fallback
   ```

   

   **主要特点**：

   - 将用户画像、用户需求和候选商品信息交给 LLM。
   - 通过语义理解对候选商品重新排序。
   - 超时或调用失败时自动 fallback 到 RRF 原始排序。
   - 避免 Rerank 服务异常影响整个推荐链路。

   ------

   ### 3. InventoryAgent

   **职责**：对 Rerank 后的候选商品进行库存校验。

   | 属性     | 说明                                       |
   | :------- | :----------------------------------------- |
   | **输入** | `ranked_products`                          |
   | **处理** | 查询 MySQL 库存表，过滤 `stock > 0` 的商品 |
   | **输出** | `available_ids`                            |
   | **特点** | 只检查最终候选集，不负责召回或排序         |

   ------

   ### 4. MarketingCopyAgent

   **职责**：为最终推荐商品生成个性化营销文案。

   | 属性     | 说明                                              |
   | :------- | :------------------------------------------------ |
   | **输入** | `user_profile` + `final_products`                 |
   | **处理** | 按用户分群选择不同 Prompt 模板，调用 LLM 生成文案 |
   | **输出** | `marketing_copies`：商品 ID → 营销文案            |
   | **特点** | 支持多用户类型 Prompt，并进行简单合规过滤         |

   ------

   ## 💻 关键代码展示

   ### Supervisor 路由（LangGraph 状态机）

   python

   ```
   # orchestrator/graph.py
   
   graph.add_conditional_edges(
       "supervisor",
       route_supervisor,
       {
           "user_profile": "user_profile",
           "product_recall": "product_recall",
           "rerank": "rerank",
           "inventory": "inventory",
           "marketing_copy": "marketing_copy",
           "finish": "aggregate",
       },
   )
   ```

   

   Supervisor 根据 `PipelineState` 中各阶段的执行状态和产出结果决定下一步路由。

   同时设置：

   python

   ```
   MAX_ITERATIONS = 20
   MAX_AGENT_VISITS = 3
   ```

   

   用于：

   - 防止 LangGraph 无限循环。
   - 防止单个 Agent 被重复调度过多次。

   ------

   ### ReAct / Tool Calling

   python

   ```
   # agents/product_rec_agent.py
   
   tools = [
       search_products,
       get_product_detail,
   ]
   
   agent = create_react_agent(
       llm,
       tools,
       prompt,
   )
   
   result = await agent.ainvoke(
       {
           "messages": [
               HumanMessage(content=user_intent)
           ]
       }
   )
   ```

   

   ProductRecAgent 的 Recall 模式中：

   text

   ```
   LLM
    │
    ├── search_products()
    │
    ├── get_product_detail()
    │
    └── search_products()
           │
           ▼
       Final Recall
   ```

   

   LLM 不直接访问数据库，而是通过 LangChain Tool 调用后端检索能力。

   ------

   ### Hybrid Retrieval（RRF Fusion）

   python

   ```
   # tools/product_tools.py
   
   # 1. MySQL 关键词搜索
   keyword_results = product_store.keyword_search_products(
       query,
       limit=40,
   )
   
   # 2. Milvus 向量搜索
   embedding = embedding_service.embed(query)
   
   vector_hits = vector_store.search(
       query_embedding=embedding,
       limit=40,
   )
   
   # 3. RRF Fusion
   k = 60
   
   for rank, pid in enumerate(vector_results, 1):
       scores[pid] += 1.0 / (k + rank)
   
   for rank, row in enumerate(keyword_results, 1):
       scores[row["product_id"]] += 1.0 / (k + rank)
   
   # 4. 按 RRF 分数排序
   ```

   

   RRF 核心公式：

   text

   ```
                1
   Score(d) = Σ -------------
                k + rank(d)
   ```

   

   本项目使用：

   text

   ```
   k = 60
   ```

   

   通过融合：

   text

   ```
   Milvus Vector Search
           +
   MySQL Keyword Search
           ↓
       RRF Fusion
           ↓
     Top-K Candidates
   ```

   

   降低单一检索方式的局限性。

   ------

   ### Metrics 线程安全

   python

   ```
   # services/metrics.py
   
   class MetricsCollector:
   
       def __init__(self):
           self._lock = threading.Lock()
           self._agent_metrics = {}
   
       def record_recall(
           self,
           latency_ms,
           success,
           ...
       ):
           with self._lock:
               self._recall_metrics["call_count"] += 1
   ```

   

   所有 `record_*` 方法使用 `threading.Lock` 保护共享统计数据。

   这样可以避免并发请求下多个线程同时修改计数器导致的数据竞争。

   ------

   ## 🚀 快速上手运行

   ### 1. 克隆项目并配置环境

   bash

   ```
   git clone <repository-url>
   cd multi-agent-ecommerce-system
   cp python/.env.example python/.env
   ```

   

   编辑 `python/.env`，配置：

   env

   ```
   LLM_API_KEY=your_llm_api_key
   EMBEDDING_API_KEY=your_embedding_api_key
   LLM_BASE_URL=https://api.deepseek.com/v1
   LLM_MODEL=deepseek-v4-flash
   ```

   

   ### 2. 启动服务

   bash

   ```
   docker compose up -d
   ```

   

   主要服务包括：

   text

   ```
   FastAPI  → 端口 8000
   MySQL    → 端口 3306
   Redis    → 端口 6379
   Milvus   → 端口 19530
   etcd     → 端口 2379
   MinIO    → 端口 9091
   ```

   

   ### 3. 测试推荐接口

   bash

   ```
   curl -X POST http://localhost:8000/api/v1/recommend \
     -H "Content-Type: application/json" \
     -d '{"user_id": "U001", "num_items": 5}'
   ```

   

   ### 4. 查看 Metrics

   bash

   ```
   curl http://localhost:8000/api/v1/metrics
   ```

   

   ------

   ## 📡 API 接口文档

   | 方法   | 接口                      | 作用                     |
   | :----- | :------------------------ | :----------------------- |
   | `POST` | `/api/v1/recommend`       | 执行完整推荐流程         |
   | `POST` | `/api/v1/recommend/graph` | 返回 LangGraph 状态快照  |
   | `GET`  | `/api/v1/metrics`         | 查看系统运行指标         |
   | `GET`  | `/api/v1/experiments`     | 查看 A/B Test 配置和统计 |
   | `POST` | `/api/v1/events/exposure` | 上报曝光事件             |
   | `POST` | `/api/v1/events/click`    | 上报点击事件             |
   | `POST` | `/api/v1/events/purchase` | 上报购买事件             |
   | `GET`  | `/api/v1/products`        | 商品列表                 |
   | `GET`  | `/health`                 | 健康检查                 |

   ### 推荐请求示例

   json

   ```
   POST /api/v1/recommend
   Content-Type: application/json
   
   {
     "user_id": "U001",
     "num_items": 5
   }
   ```

   

   ### Metrics 返回示例

   json

   ```
   {
     "supervisor": {
       "call_count": 1,
       "avg_latency_ms": 0.6
     },
     "recall": {
       "call_count": 1,
       "avg_latency_ms": 7843.5,
       "avg_tool_calls": 3
     },
     "rerank": {
       "call_count": 1,
       "avg_candidate_count": 10
     },
     "marketing": {
       "call_count": 1,
       "success_count": 1
     },
     "request": {
       "count": 1,
       "avg_total_latency_ms": 14773
     }
   }
   ```

   

   > Metrics 中的延迟会受到 LLM API、网络以及 Tool Calling 次数等因素影响，因此实际运行值会随请求变化。

   ------

   ## 📁 项目文件结构

   text

   ```
   multi-agent-ecommerce-system/
   ├── README.md
   ├── docker-compose.yml      # Docker Compose 服务编排
   ├── .env.example
   ├── docs/                   # 补充文档
   ├── scripts/                # 数据导入脚本
   └── python/
       ├── main.py             # FastAPI 入口
       ├── streamlit_app.py    # 可视化界面
       ├── Dockerfile
       ├── requirements.txt
       │
       ├── agents/             # 4 个核心 Agent
       │   ├── user_profile_agent.py
       │   ├── product_rec_agent.py
       │   ├── inventory_agent.py
       │   └── marketing_copy_agent.py
       │
       ├── orchestrator/       # Supervisor + LangGraph
       │   ├── graph.py
       │   └── supervisor.py
       │
       ├── services/           # 基础设施
       │   ├── metrics.py      # 线程安全 Metrics
       │   ├── ab_test.py      # A/B Test
       │   ├── vector_store.py
       │   └── embedding_service.py
       │
       ├── tools/              # LangChain Tools
       │   └── product_tools.py
       │
       ├── models/
       │   └── schemas.py
       │
       ├── config/
       │   └── settings.py
       │
       └── tests/
   ```

   

   ------

   ## 🛠 技术栈总览

   | 分类           | 技术                                  |
   | :------------- | :------------------------------------ |
   | **语言**       | Python 3.11+                          |
   | **Web 框架**   | FastAPI + Uvicorn                     |
   | **Agent 框架** | LangGraph + LangChain                 |
   | **LLM**        | DeepSeek / MiniMax（ChatOpenAI 兼容） |
   | **Embedding**  | BGE-M3（SiliconFlow API，1024 维）    |
   | **向量数据库** | Milvus（Cosine 相似度）               |
   | **关系数据库** | MySQL（生产）/ SQLite（开发）         |
   | **缓存**       | Redis                                 |
   | **A/B Test**   | Thompson Sampling（自实现）           |
   | **测试**       | pytest                                |
   | **容器化**     | Docker + Docker Compose               |

   
