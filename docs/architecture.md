# Fraud Copilot: Architecture Document
## LangGraph-on-Databricks with Analyst Augmentation

**Lovelytics | Gen AI Engineer Technical Assessment**  
**Diego Cortes | March 2026**

---

## Table of Contents

1. [Business Understanding](#1-business-understanding)
2. [Fundamental Architectural Decisions](#2-fundamental-architectural-decisions)
3. [System Architecture](#3-system-architecture)
4. [Machine Learning Models](#4-machine-learning-models)
5. [MLOps Pipeline](#5-mlops-pipeline)
6. [Multi-Tool Agent Design](#6-multi-tool-agent-design)
7. [Concurrency and Scalability](#7-concurrency-and-scalability)
8. [LLM Evaluation and Control](#8-llm-evaluation-and-control)
9. [Analyst Augmentation: Cognitive Load](#9-analyst-augmentation-cognitive-load)
10. [Cost Analysis](#10-cost-analysis)
11. [Implementation Roadmap](#11-implementation-roadmap)
12. [Implementation Iterations and Lessons Learned](#12-implementation-iterations-and-lessons-learned)
13. [Trade-offs and Future Improvements](#13-trade-offs-and-future-improvements)

---

## 1. Business Understanding

### 1.1 The Real Problem

Fraud Copilot is not a chatbot. It is an **operational augmentation system** for financial fraud analysts. Business value is measured across fice axes:

| Value Axis | Impact |
|---|---|
| Average investigation time | operational time |
| Detection rate (recall) | more frauds detected |
| False positives investigated/day | less unnecessary work |
| Compliance errors per month | less regulatory risk |
| Average cost per investigation | reduction total cost |

### 1.2 Guiding Principle: Analyst Augmentation, Not Replacement

The system does not replace analysts with agents. It enables each analyst to handle more cases at higher quality. This manifests in three concrete capabilities:

**Capability 1 — Unified information access:** The analyst asks a natural language question and receives data, predictions, and regulatory knowledge in a single response, instead of consulting 3-4 separate tools.

**Capability 2 — Operational explainability:** Every fraud model prediction includes the features that contributed most to the decision and a recommended action ("verify identity", "block international transactions"), reducing interpretation time.

**Capability 3 — Cognitive load adaptation:** The system detects signals of operational fatigue and adjusts response complexity, protecting the quality of analyst decisions during high-load periods.

### 1.3 Analyst Workflows

A typical analyst executes a lot of investigations daily. Each investigation combines:

1. **Data queries** for transaction context 
2. **Risk evaluation** using predictive models
3. **Regulatory verification** against KYC/AML/PCI DSS documentation 
4. **Complex analyses** combining data + prediction + knowledge

---

## 2. Fundamental Architectural Decisions

### 2.1 Orchestration Framework: LangGraph

**Decision:** Use LangGraph as the agent authoring framework, wrapped with `ResponsesAgent` from MLflow for native Databricks compatibility.

**Alternatives evaluated:**

| Framework | Pros | Cons | Verdict |
|---|---|---|---|
| **LangGraph** | Visual graph, explicit state machine, native checkpointing, automatic MLflow tracing | Learning curve, inherited LangChain abstractions | **Selected** |
| OpenAI Agents SDK | Simple, native Databricks template, elegant handoffs | Tied to OpenAI as LLM provider, less flow control | Rejected |
| Agno | Lightweight, intuitive, native "teams" concept | Newer, less battle-tested in production | Future alternative |
| Pure Python (ChatAgent) | Maximum control, no external dependencies | No flow visualization, more boilerplate code | Rejected |

**Technical justification — three capabilities that alternatives do not match for this use case:**

1. **Graph visualization:** During a live demo, the exact flow of each query type through nodes can be shown — which decisions are made, where the flow branches. This is difficult to demonstrate with agents based solely on tool calling.
2. **Explicit state machine:** Conditional transitions between nodes are declarative and unit-testable, unlike a ReAct agent where the LLM decides the flow implicitly.
3. **Native Databricks integration:** MLflow Tracing automatically captures LangGraph execution via `mlflow.langchain.autolog()`, with no manual instrumentation.

**Risk mitigation:** If LangGraph causes production issues, migration cost is low because `ResponsesAgent` decouples the authoring framework from Databricks infrastructure. LangGraph can be replaced by Agno or pure Python by changing only the internal agent implementation, without touching serving, tracing, or evaluation.

### 2.2 Platform: Databricks-Centric

**Decision:** Concentrate data, models, agent, evaluation, and observability on Databricks.

**Justification:** Maintainability (single provider), security (Unity Catalog centralizes permissions), and compatibility (MLflow 3 traces the entire stack natively). The explicit trade-off is greater Databricks coupling in exchange for smaller operational surface.

### 2.3 MLOps Pattern: Deploy Code

**Decision:** Use the "deploy code" pattern recommended by Databricks, where what is promoted to production is the training code, not the model artifact.

**Justification:** Supports automated retraining in a controlled environment. Code is versioned in Git, executed in production against production data, and the resulting model is automatically validated before serving.

---

## 3. System Architecture

### 3.1 High-Level View

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FRAUD ANALYSTS (up to 100)                       │
│                    Natural Language Interface                       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│              DATABRICKS APPS - Chat UI + Auth                       │
│              (WebSocket streaming, OAuth, session management)       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│          MODEL SERVING ENDPOINT: fraud-copilot-agent                │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │            LangGraph StateGraph                               │  │
│  │                                                               │  │
│  │  [classify_intent] ──→ [data_query]  ─────────┐               │  │
│  │        │                                      │               │  │
│  │        ├──────────→ [predict_fraud]  ──┐      │               │  │
│  │        │                               │      │               │  │
│  │        ├──────────→ [predict_purchase] ┤      │               │  │
│  │        │                               ├→ [assess_load]       │  │
│  │        ├──────────→ [search_knowledge] ┤      │               │  │
│  │        │                               │      ▼               │  │
│  │        └──────────→ [complex_analysis] ┘  [synthesize] → OUT  │  │
│  │                                                               │  │
│  │  mlflow.langchain.autolog() → automatic tracing               │  │
│  └───────────────────────────────────────────────────────────────┘  │
└───────────┬──────────┬──────────┬──────────┬────────────────────────┘
            │          │          │          │
            ▼          ▼          ▼          ▼
┌──────────────┐ ┌──────────┐ ┌─────────┐ ┌──────────────────────┐
│ SQL Warehouse│ │Model Srv:│ │Model Srv│ │ Vector Search Index  │
│ Delta Tables │ │fraud_mod │ │purch_mod│ │ Financial KB         │
│ fraud_dataset│ │XGBoost   │ │LightGBM │ │ RAG with citations   │
│ purchase_data│ │Pipeline  │ │Pipeline │ │ + TF-IDF fallback    │
└──────────────┘ └──────────┘ └─────────┘ └──────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    OBSERVABILITY LAYER                               │
│                                                                     │
│  MLflow 3 Traces ──→ Production Monitoring ──→ Delta Tables         │
│  Agent Evaluation ──→ Golden Set (30 queries) ──→ Quality Scores    │
│  Lakehouse Monitoring ──→ Model Drift ──→ Retrain Triggers          │
│  Analyst Session Metrics ──→ Cognitive Load Dashboard               │
└─────────────────────────────────────────────────────────────────────┘
```
### Production Deployment: 3-Level Concurrency & AI Gateway

```mermaid
flowchart TB
    subgraph analysts["100 Fraud Analysts"]
        direction LR
        a1["Analyst 1"]
        a2["Analyst 2"]
        a3["..."]
        a4["Analyst 100"]
    end

    subgraph presentation["Presentation — Databricks Apps"]
        ui["Chat UI<br/><b>WebSocket Streaming</b><br/><i>OAuth + session persistence<br/>Conversation history → Delta</i>"]
    end

    subgraph level1["LEVEL 1 — Agent Endpoint (LangGraph)"]
        lb1["Load Balancer"]
        subgraph replicas_agent["Auto-Scaling Replicas"]
            direction LR
            r1["Replica 1<br/><i>LangGraph<br/>process</i>"]
            r2["Replica 2"]
            r3["..."]
            r4["Replica N"]
        end
        config1["min_concurrency: 8<br/>max_concurrency: 64<br/>scale_to_zero: false<br/>workload_type: CPU"]
    end

    subgraph level2["LEVEL 2 — ML Model Endpoints"]
        subgraph fraud_ep["Fraud Model Endpoint"]
            direction LR
            fr1["Replica"]
            fr2["Replica"]
            fr3["..."]
        end
        config_f["min: 4 · max: 16<br/>CPU · XGBoost Pipeline"]

        subgraph purchase_ep["Purchase Model Endpoint"]
            direction LR
            pr1["Replica"]
            pr2["Replica"]
        end
        config_p["min: 4 · max: 12<br/>CPU · LightGBM Pipeline"]
    end

    subgraph gateway["AI GATEWAY — Mosaic AI"]
        gw["AI Gateway<br/><b>Rate Limiting + Routing</b>"]
        
        subgraph llm_providers["LLM Providers (Failover Chain)"]
            direction LR
            primary["Primary<br/><b>Llama 3.1 405B</b><br/><i>Databricks-hosted</i>"]
            fallback1["Fallback 1<br/><b>Llama 3.1 70B</b><br/><i>Lower cost</i>"]
            fallback2["Fallback 2<br/><b>External API</b><br/><i>Claude / GPT-4</i>"]
        end

        gw_features["Features:<br/>• Rate limiting + retry with backoff<br/>• Prompt caching (static system prompts)<br/>• Token usage tracking per endpoint<br/>• Automatic failover on 429/5xx"]
    end

    subgraph level3["LEVEL 3 — SQL Warehouse"]
        sql["Serverless SQL Warehouse<br/><b>Auto-Scaling</b>"]
        sql_config["min_clusters: 1<br/>max_clusters: 4<br/>auto_stop: 10 min"]
    end

    subgraph vs_layer["Vector Search"]
        vs["Vector Search Endpoint<br/><b>fraud-copilot-vs-endpoint</b><br/><i>STANDARD type</i>"]
    end

    subgraph session["Session State Management"]
        sticky["Option A: Sticky Sessions<br/><i>Load balancer affinity<br/>Same analyst → same replica</i>"]
        delta_state["Option B: Delta Persistence<br/><i>Read conversation history<br/>per analyst from Delta<br/>~100ms overhead</i>"]
    end

    analysts --> ui
    ui --> lb1
    lb1 --> replicas_agent
    replicas_agent -->|"Model inference<br/>+ SHAP"| level2
    replicas_agent -->|"Intent classify<br/>+ Synthesize"| gw
    gw --> primary
    primary -.->|"429 / timeout"| fallback1
    fallback1 -.->|"429 / timeout"| fallback2
    replicas_agent -->|"Data queries<br/>(production path)"| level3
    replicas_agent -->|"RAG retrieval"| vs
    replicas_agent -.->|"Session context"| session

    style level1 fill:#0d1117,stroke:#e94560,color:#c9d1d9
    style level2 fill:#0d1117,stroke:#1f6feb,color:#c9d1d9
    style gateway fill:#0d1117,stroke:#d29922,color:#c9d1d9
    style level3 fill:#0d1117,stroke:#3fb950,color:#c9d1d9
    style session fill:#0d1117,stroke:#8b949e,color:#c9d1d9
```
### CI/CD Pipeline: Deploy Code Pattern with Catalog Promotion

```mermaid
flowchart TB
    subgraph trigger["Triggers"]
        direction LR
        push["Git Push<br/><i>PR merged to main</i>"]
        drift["Drift Alert<br/><i>PSI > 0.2 or<br/>AUC < 0.85</i>"]
        sched["Scheduled<br/><i>Weekly (fraud)<br/>Bi-weekly (purchase)</i>"]
    end

    subgraph ci["CI — GitHub Actions"]
        lint["Lint & Format<br/><i>ruff, black</i>"]
        unit["Unit Tests<br/><i>pytest: 40 tests<br/>tools, policies,<br/>routing, extraction</i>"]
        build["Build DAB Bundle<br/><i>databricks bundle validate</i>"]
        lint --> unit --> build
    end

    subgraph dev_cat["DEV Catalog"]
        dev_train["Execute Training Code<br/><i>Same notebooks as dev<br/>Against dev data</i>"]
        dev_log["MLflow Experiment<br/><i>Log params, metrics,<br/>SHAP artifacts</i>"]
        dev_reg["Register Model<br/><i>dev.default.fraud_model<br/>dev.default.purchase_model</i>"]
        dev_train --> dev_log --> dev_reg
    end

    subgraph gates["Validation Gates (Automated)"]
        direction TB
        gate1["Gate 1: Performance<br/><b>AUC > 0.88</b> (fraud)<br/><b>R² > 0.80</b> (purchase)<br/><i>Evaluated on holdout set</i>"]
        gate2["Gate 2: Fairness<br/><b>Disparity < 0.05</b><br/><i>Check by country,<br/>device_type subgroups</i>"]
        gate3["Gate 3: Latency<br/><b>P95 < 200ms</b><br/><i>Load test: 100 requests<br/>against staging endpoint</i>"]
        gate4["Gate 4: A/B Comparison<br/><b>Challenger >= Champion</b><br/><i>Same holdout set,<br/>paired comparison</i>"]
        gate5["Gate 5: Integration<br/><b>Agent invokes correctly</b><br/><i>5 smoke queries<br/>per intent type</i>"]
        gate6["Gate 6: Agent Quality<br/><b>Golden set > 90% routing</b><br/><i>30 queries against<br/>staging agent</i>"]
        gate1 --> gate2 --> gate3 --> gate4 --> gate5 --> gate6
    end

    subgraph staging_cat["STAGING Catalog"]
        stg_deploy["Deploy to Staging<br/><i>databricks bundle deploy<br/>--target staging</i>"]
        stg_ep["Staging Endpoints<br/><i>fraud-copilot-agent-staging<br/>fraud-model-staging<br/>purchase-model-staging</i>"]
        stg_deploy --> stg_ep
    end

    subgraph prod_cat["PROD Catalog"]
        promote["Promote Model<br/><i>Copy model version<br/>dev → prod catalog<br/>Set alias @champion</i>"]
        prod_ep["Production Endpoints<br/><i>fraud-copilot-agent<br/>fraud-model<br/>purchase-model</i>"]
        prod_deploy["Deploy Agent<br/><i>databricks bundle deploy<br/>--target production</i>"]
        promote --> prod_deploy --> prod_ep
    end

    subgraph fail_path["Gate Failure"]
        alert["Notify Team<br/><i>Slack + email</i>"]
        challenger["Model stays as<br/><b>@challenger</b><br/><i>Available for A/B<br/>but not serving</i>"]
        report["Generate Report<br/><i>Which gate failed?<br/>Metric values vs thresholds</i>"]
        alert --> challenger
        alert --> report
    end

    trigger --> ci
    ci -->|"Tests pass"| dev_cat
    dev_cat -->|"Model registered"| staging_cat
    staging_cat --> gates
    gates -->|"All pass"| prod_cat
    gates -->|"Any fails"| fail_path

    style ci fill:#0d1117,stroke:#8b949e,color:#c9d1d9
    style dev_cat fill:#0d1117,stroke:#3fb950,color:#c9d1d9
    style staging_cat fill:#0d1117,stroke:#d29922,color:#c9d1d9
    style gates fill:#0d1117,stroke:#d29922,color:#c9d1d9
    style prod_cat fill:#0d1117,stroke:#1f6feb,color:#c9d1d9
    style fail_path fill:#2d1117,stroke:#e94560,color:#f85149

```

### Production Monitoring: Drift Detection & Retrain Loop

```mermaid
flowchart TB
    subgraph serving["Model Serving (Production)"]
        fraud_ep["Fraud Model<br/>Endpoint"]
        purchase_ep["Purchase Model<br/>Endpoint"]
        agent_ep["Agent Endpoint"]
    end

    subgraph inference_logs["Inference Tables (Automatic)"]
        inf_fraud[("fraud_model_inference_log<br/><i>Every prediction:<br/>features, probability,<br/>risk_tier, timestamp,<br/>latency_ms</i>")]
        inf_purchase[("purchase_model_inference_log<br/><i>Every prediction:<br/>features, predicted_amount,<br/>timestamp, latency_ms</i>")]
        inf_agent[("agent_inference_log<br/><i>Every query:<br/>intent, tools_used,<br/>latency, token_count</i>")]
    end

    subgraph traces["MLflow Traces"]
        trace_store[("MLflow Trace Store<br/><i>Full execution graph<br/>per query: classify →<br/>tool → assess → synth</i>")]
    end

    subgraph monitoring["Lakehouse Monitoring (Scheduled Jobs)"]
        subgraph fraud_mon["Fraud Model Monitors"]
            psi_fraud["PSI Monitor<br/><b>Weekly</b><br/><i>Top-10 features<br/>vs training baseline</i>"]
            auc_fraud["AUC Rolling Window<br/><b>Daily</b><br/><i>30-day window<br/>against labeled data</i>"]
            dist_fraud["Prediction Distribution<br/><b>Daily</b><br/><i>P(fraud) histogram<br/>shift vs baseline</i>"]
        end

        subgraph purchase_mon["Purchase Model Monitors"]
            psi_purchase["PSI Monitor<br/><b>Bi-weekly</b><br/><i>Top-10 features</i>"]
            rmse_purchase["RMSE by Tier<br/><b>Weekly</b><br/><i>Per membership_tier<br/>rolling window</i>"]
            mape_purchase["MAPE Monitor<br/><b>Weekly</b><br/><i>Alert if > 20%<br/>in any tier</i>"]
        end

        subgraph agent_mon["Agent Monitors"]
            routing_acc["Routing Accuracy<br/><b>Weekly</b><br/><i>Golden set re-eval<br/>30 queries</i>"]
            latency_mon["Latency Monitor<br/><b>Continuous</b><br/><i>P50 / P95 / P99<br/>per intent type</i>"]
            token_mon["Token Usage<br/><b>Daily</b><br/><i>Avg tokens per query<br/>by intent type</i>"]
            error_mon["Error Rate<br/><b>Continuous</b><br/><i>HTTP 5xx rate<br/>per endpoint</i>"]
        end
    end

    subgraph alerts["Alert Engine (Databricks SQL Alerts)"]
        direction TB
        alert_drift["DRIFT ALERT<br/><i>PSI > 0.2 any feature<br/>or AUC < 0.85</i>"]
        alert_perf["PERFORMANCE ALERT<br/><i>Latency P95 > 200ms<br/>or error rate > 1%</i>"]
        alert_quality["QUALITY ALERT<br/><i>Routing accuracy < 90%<br/>or token usage spike</i>"]
        alert_cost["COST ALERT<br/><i>Daily spend > threshold<br/>or DBU anomaly</i>"]
    end

    subgraph actions["Automated Actions"]
        retrain["Trigger Retrain<br/><b>Databricks Workflow</b><br/><i>Execute training notebooks<br/>against prod data</i>"]
        scale["Scale Endpoint<br/><i>Increase max_concurrency<br/>or add replicas</i>"]
        notify["Notify Team<br/><i>Slack + PagerDuty</i>"]
        dashboard["Update Dashboard<br/><i>SQL Dashboard:<br/>drift trends,<br/>quality metrics,<br/>cost attribution</i>"]
    end

    subgraph feedback["Feedback Loop"]
        labels["Ground Truth Labels<br/><i>Analyst confirms/rejects<br/>fraud predictions<br/>(delayed 24-72h)</i>"]
        join["Join Predictions<br/>with Labels<br/><i>inference_log JOIN<br/>confirmed_fraud</i>"]
        eval_real["Real AUC Computation<br/><i>Actual performance<br/>vs predicted</i>"]
    end

    serving --> inference_logs
    agent_ep --> traces
    inference_logs --> monitoring
    traces --> agent_mon

    psi_fraud & auc_fraud & dist_fraud --> alert_drift
    psi_purchase & rmse_purchase & mape_purchase --> alert_drift
    latency_mon & error_mon --> alert_perf
    routing_acc --> alert_quality
    token_mon --> alert_cost

    alert_drift --> retrain
    alert_drift --> notify
    alert_perf --> scale
    alert_perf --> notify
    alert_quality --> notify
    alert_cost --> notify

    monitoring --> dashboard

    labels --> join
    inf_fraud --> join
    join --> eval_real
    eval_real --> auc_fraud

    retrain -->|"New model version<br/>→ validation gates"| serving

    style serving fill:#0d1117,stroke:#1f6feb,color:#c9d1d9
    style inference_logs fill:#0d1117,stroke:#3fb950,color:#c9d1d9
    style monitoring fill:#0d1117,stroke:#d29922,color:#c9d1d9
    style alerts fill:#0d1117,stroke:#e94560,color:#c9d1d9
    style actions fill:#0d1117,stroke:#8b949e,color:#c9d1d9
    style feedback fill:#0d1117,stroke:#a371f7,color:#c9d1d9

```
### Production Cognitive Load: Real-Time Analyst Monitoring

```mermaid
flowchart TB
    subgraph analysts["Analyst Sessions"]
        direction LR
        s1["Analyst A<br/><i>Session active<br/>3.5 hours</i>"]
        s2["Analyst B<br/><i>Session active<br/>1 hour</i>"]
        s3["Analyst C<br/><i>Session active<br/>6 hours</i>"]
    end

    subgraph capture["Signal Capture (Per Query)"]
        direction TB
        sig1["Query Counter<br/><i>queries_last_hour++</i>"]
        sig2["Routing Tier<br/><i>Update avg_routing_tier<br/>from classify_intent</i>"]
        sig3["Interval Timer<br/><i>timestamp_now -<br/>timestamp_last_query</i>"]
        sig4["Follow-up Detector<br/><i>Semantic similarity<br/>vs previous query<br/>> 0.8 = follow-up</i>"]
        sig5["Session Clock<br/><i>first_query_ts → now</i>"]
        sig6["Circadian Factor<br/><i>Current hour of day</i>"]
    end

    subgraph delta_session["analyst_session_metrics — Delta Table"]
        direction TB
        delta_desc["Partitioned by analyst_id<br/>Updated per query"]
        schema["Schema:<br/>analyst_id STRING<br/>session_id STRING<br/>queries_last_hour INT<br/>avg_routing_tier FLOAT<br/>session_duration_hours FLOAT<br/>avg_query_interval_sec FLOAT<br/>followup_rate FLOAT<br/>hour_of_day INT<br/>load_score INT<br/>load_level STRING<br/>updated_at TIMESTAMP"]
    end

    subgraph compute["Load Score Computation"]
        heuristic["Heuristic Scorer<br/><b>v1 (Prototype)</b><br/><i>6 weighted signals<br/>score 0-100</i>"]
        ml_model["ML Model<br/><b>v2 (Future)</b><br/><i>Trained on real session<br/>data + analyst feedback<br/>+ escalation outcomes</i>"]
        heuristic -.->|"Replace when<br/>data available"| ml_model
    end

    subgraph levels["Load Level Actions"]
        normal["NORMAL (0-30)<br/><i>Standard detailed responses<br/>Full SHAP explanations<br/>Complete citations</i>"]
        elevated["ELEVATED (31-60)<br/><i>Executive summary first<br/>Structured bullet points<br/>Key metrics highlighted</i>"]
        high["HIGH (61-80)<br/><i>Concise responses only<br/>Skip secondary details<br/>Offer: 'Need more detail?'</i>"]
        critical["CRITICAL (81-100)<br/><i>Minimal 2-3 sentence answers<br/>Suggest break<br/>Offer handoff to colleague</i>"]
    end

    subgraph supervisor_layer["Supervisor Dashboard (SQL Dashboard)"]
        heatmap["Heatmap<br/><b>load_score × analyst × hour</b><br/><i>Visual: who is overloaded<br/>and when</i>"]
        trend["Weekly Trend<br/><b>Avg load by team</b><br/><i>Is overall load<br/>increasing?</i>"]
        correlation["Correlation Analysis<br/><b>load_score vs quality</b><br/><i>Escalation rate,<br/>follow-up rate,<br/>error rate</i>"]
        alerts_sup["Real-Time Alerts<br/><b>Analyst > 60 for 2+ hours</b><br/><i>Slack notification<br/>to supervisor</i>"]
    end

    subgraph ops_actions["Operational Actions"]
        redistribute["Redistribute Caseload<br/><i>Move complex cases<br/>from overloaded analyst<br/>to available analysts</i>"]
        break_suggest["Suggest Break<br/><i>System note appended<br/>to agent response</i>"]
        shift_review["Shift Review<br/><i>Supervisors review<br/>high-load periods<br/>for staffing decisions</i>"]
    end

    analysts --> capture
    capture --> delta_session
    delta_session --> compute
    compute --> levels

    levels -->|"normal/elevated"| analysts
    levels -->|"high"| break_suggest
    levels -->|"critical"| redistribute

    delta_session --> supervisor_layer
    alerts_sup --> redistribute
    correlation --> shift_review

    style capture fill:#0d1117,stroke:#3fb950,color:#c9d1d9
    style delta_session fill:#0d1117,stroke:#1f6feb,color:#c9d1d9
    style compute fill:#0d1117,stroke:#d29922,color:#c9d1d9
    style levels fill:#0d1117,stroke:#e94560,color:#c9d1d9
    style supervisor_layer fill:#0d1117,stroke:#a371f7,color:#c9d1d9
    style ops_actions fill:#0d1117,stroke:#8b949e,color:#c9d1d9
```

### Cost Optimization: Tier Routing — LLM Bypass for Simple Queries

```mermaid
flowchart TB
    subgraph input["Incoming Queries (5,000/day)"]
        queries["All Analyst Queries"]
    end

    subgraph classifier["Intent Classifier"]
        classify["LLM Call #1<br/><b>classify_intent</b><br/><i>~500 tokens<br/>$0.005/query</i>"]
    end

    subgraph tier1["TIER 1 — Direct SQL (40% traffic)"]
        direction TB
        t1_desc["Data Analysis Queries<br/><i>'How many international?'<br/>'Average amount by fraud?'<br/>'Compare Gold vs Silver'</i>"]
        t1_exec["Execute SQL<br/><i>Pattern-match → DataFrame<br/>or Text-to-SQL → Warehouse</i>"]
        t1_response["Return Formatted Result<br/><b>No LLM synthesis needed</b><br/><i>Deterministic response</i>"]
        t1_cost["Cost: $0.005<br/><i>1 LLM call (classify only)<br/>~500 tokens</i>"]
        t1_desc --> t1_exec --> t1_response
    end

    subgraph tier2["TIER 2 — Model + Light Synthesis (25% traffic)"]
        direction TB
        t2_desc["Prediction Queries<br/><i>'Predict fraud for $2,500...'<br/>'Expected purchase for 45yo...'</i>"]
        t2_exec["Model Inference<br/><i>Pipeline predict + SHAP<br/>Risk tier + action</i>"]
        t2_synth["LLM Call #2<br/><b>synthesize</b><br/><i>Format SHAP explanation<br/>~1,200 tokens</i>"]
        t2_cost["Cost: $0.02<br/><i>2 LLM calls<br/>~1,700 tokens total</i>"]
        t2_desc --> t2_exec --> t2_synth
    end

    subgraph tier3["TIER 3 — RAG + Full Synthesis (20% traffic)"]
        direction TB
        t3_desc["Knowledge Queries<br/><i>'KYC procedures for...'<br/>'PCI DSS requirements?'</i>"]
        t3_exec["RAG Retrieval<br/><i>Vector Search top-5<br/>+ context building</i>"]
        t3_synth["LLM Call #2<br/><b>synthesize with citations</b><br/><i>Context + format<br/>~2,000 tokens</i>"]
        t3_cost["Cost: $0.04<br/><i>2 LLM calls<br/>~2,500 tokens total</i>"]
        t3_desc --> t3_exec --> t3_synth
    end

    subgraph tier4["TIER 4 — Complex Multi-Step (15% traffic)"]
        direction TB
        t4_desc["Complex Queries<br/><i>'Top 5 suspicious + why?'<br/>'Fraud patterns intl vs dom?'</i>"]
        t4_data["Step 1: Data Retrieval"]
        t4_model["Step 2: Model × 5 txns"]
        t4_synth["LLM Call #2<br/><b>full synthesis</b><br/><i>Data + predictions + SHAP<br/>~3,000 tokens</i>"]
        t4_cost["Cost: $0.08<br/><i>3 LLM calls<br/>~3,500 tokens total</i>"]
        t4_desc --> t4_data --> t4_model --> t4_synth
    end

    subgraph cost_summary["Cost Summary"]
        direction TB
        with_routing["WITH Tier Routing<br/><b>Weighted avg: $0.03/query</b><br/><i>5,000 queries/day = $150/day<br/>= ~$4,500/month LLM cost</i>"]
        without_routing["WITHOUT Routing<br/><b>All queries full pipeline</b><br/><i>$0.05 avg × 5,000 = $250/day<br/>= ~$7,500/month LLM cost</i>"]
        savings["SAVINGS: 40%<br/><b>$3,000/month</b><br/><i>From bypassing LLM synthesis<br/>on Tier 1 queries</i>"]
    end

    subgraph future_opt["Future: Semantic Cache"]
        cache["Vector Similarity Cache<br/><i>If query embedding matches<br/>previous query > 0.95 similarity:<br/>return cached response</i>"]
        cache_rate["Estimated Cache Hit: 25-40%<br/><i>Regulatory queries are<br/>highly repetitive</i>"]
        cache_savings["Additional Savings: 20-30%<br/><i>$1,000-$2,000/month</i>"]
        cache --> cache_rate --> cache_savings
    end

    queries --> classifier
    classify --> tier1
    classify --> tier2
    classify --> tier3
    classify --> tier4

    tier1 & tier2 & tier3 & tier4 --> cost_summary
    cost_summary -.-> future_opt

    style tier1 fill:#0d1117,stroke:#3fb950,color:#c9d1d9
    style tier2 fill:#0d1117,stroke:#1f6feb,color:#c9d1d9
    style tier3 fill:#0d1117,stroke:#d29922,color:#c9d1d9
    style tier4 fill:#0d1117,stroke:#e94560,color:#c9d1d9
    style cost_summary fill:#161b22,stroke:#3fb950,color:#c9d1d9
    style future_opt fill:#161b22,stroke:#8b949e,color:#8b949e

```



### 3.2 Component Responsibilities

**Databricks Apps (Frontend):** Chat UI with WebSocket streaming, OAuth authentication integrated with Databricks, and conversation history persistence in Delta tables. For the prototype, the `agent-openai-agents-sdk` template provides a functional chat UI with no additional frontend development.

**Model Serving Endpoint (Agent):** The LangGraph agent is packaged as an MLflow model (via `ResponsesAgent`), registered in Unity Catalog, and deployed as a serverless endpoint in Model Serving. Databricks manages auto-scaling, logging, version control, and access control.

**SQL Warehouse (Data Queries):** Delta tables `fraud_dataset` and `product_purchase_dataset` are queried via pattern-matching data tools. For the prototype, CSVs are loaded as Delta tables with schema enforcement.

**Model Serving Endpoints (ML Models):** Two separate endpoints for the classification and regression models. Each with its own independent MLOps lifecycle. The agent invokes them as tools.

**Vector Search Index (RAG):** Vector index over the financial knowledge base documents. Embedding: `databricks-gte-large-en` (Databricks-hosted). Retrieval: top-5 chunks. Agent responses include source document citations. A TF-IDF fallback guarantees RAG functionality even when Vector Search is unavailable.

---

## 4. Machine Learning Models

### 4.1 Data Analysis

**fraud_dataset.csv:**
- 100 records, 32 columns, target: `fraud` (binary 0/1)
- Balance: 50% fraud / 50% legitimate (artificially balanced)
- 82 unique customers, 51 countries, 15 merchant categories
- Amount range: $67.89 - $9,876.00 (mean: $2,339.96)
- 49% international transactions

**product_purchase_dataset.csv:**
- 100 records, 30 columns, target: `purchase_amount` (continuous)
- Tier distribution: silver (37), gold (30), platinum (20), bronze (13)
- Purchase amount range: $98.60 - $612.45 (mean: $342.04)

**financial_documents/ (Knowledge Base):**
- 6 markdown documents (from a set of 20)
- Topics: credit fraud indicators, AML, transaction monitoring, CLV, PCI DSS, KYC

### 4.2 Fraud Detection Model

| Attribute | Specification |
|---|---|
| **Problem** | Binary classification (fraud: 0/1) |
| **Algorithm** | XGBoost wrapped in sklearn Pipeline |
| **Pipeline** | `ColumnTransformer(passthrough + OrdinalEncoder) → XGBClassifier` |
| **Features** | 28 total: 18 numeric + 6 binary + 4 categorical (raw strings) |
| **Validation** | Stratified 5-Fold Cross Validation |
| **Primary metric** | ROC-AUC (target: > 0.90) |
| **Secondary metrics** | F1-Score, PR-AUC, Precision, Recall |
| **Explainability** | SHAP TreeExplainer: waterfall per prediction + global summary |
| **Output** | `probability`, `risk_score` (0-100), `risk_tier` (H/M/L), `top_5_features`, `suggested_action` |
| **Registration** | MLflow Model Registry in Unity Catalog, alias `champion` |

**Key design decision:** All preprocessing is packaged inside the sklearn Pipeline so the registered model accepts raw data directly. This prevents train-serve skew — the same data format flows from the analyst's query through the agent to the model without intermediate encoding steps.

### 4.3 Purchase Prediction Model

| Attribute | Specification |
|---|---|
| **Problem** | Regression (purchase_amount: continuous) |
| **Algorithm** | LightGBM wrapped in sklearn Pipeline |
| **Pipeline** | `ColumnTransformer(passthrough + OrdinalEncoder) → LGBMRegressor` |
| **Features** | 28 total: 14 numeric + 5 binary + 9 categorical (raw strings) |
| **Validation** | GroupKFold by `membership_tier` |
| **Primary metric** | RMSE |
| **Secondary metrics** | MAE, R², MAPE segmented by tier |
| **Target transform** | Log1p if skewness > 2 (conditional) |
| **Output** | `predicted_amount`, `confidence_interval` (±20%), `top_features` |
| **Registration** | MLflow Model Registry in Unity Catalog, alias `champion` |

### 4.4 Small Dataset Considerations

With only 100 rows per dataset, the following practices are mandatory:

1. **Report variance:** Not just the mean of AUC/RMSE, but the standard deviation across folds. With 20 samples per test fold, variance will be high.
2. **Avoid over-parameterized models:** XGBoost with `max_depth=3` and `n_estimators=150` is sufficient. No deep learning.
3. **Document the limitation:** The model is a functional prototype. Hyperparameters and thresholds will change with production data (10,000-100,000+ records).
4. **Balanced dataset caveat:** The 50/50 fraud balance is artificial. In production, fraud rate is typically 0.1-2%. The model will need recalibration.

---

## 5. MLOps Pipeline

### 5.1 Full Lifecycle

```
┌──────────────────────────────────────────────────────────────────┐
│                     DEVELOPMENT ENVIRONMENT                      │
│                                                                  │
│  [Notebook] → [Experiment] → [Compare] → [Select Best]           │
│       │              │                         │                 │
│   mlflow.autolog()   │                         ▼                 │
│   log params,    MLflow Tracking        Register Model           │
│   metrics,       Server                 in Unity Catalog         │
│   artifacts                             fraud_agent.default.*    │
└─────────────────────────────────────────┬────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                     STAGING / VALIDATION                         │
│                                                                  │
│  [Validation Pipeline]                                           │
│       ├── Gate 1: Performance ── AUC > 0.88 / R² > 0.80          │
│       ├── Gate 2: Fairness ── Disparity check by subgroup        │
│       ├── Gate 3: Latency ── Inference P95 < 200ms               │
│       ├── Gate 4: A/B Test ── Challenger >= Champion             │
│       └── Gate 5: Integration ── Agent invokes model correctly   │
│                                                                  │
│  All gates pass → Promote alias "champion"                       │
│  Any gate fails → Alert + model stays as "challenger"            │
└─────────────────────────────────────────┬────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                     PRODUCTION                                   │
│                                                                  │
│  [Model Serving Endpoint]                                        │
│       ├── Real-time inference (< 200ms P95)                      │
│       ├── Inference table → prediction logs                      │
│       └── Lakehouse Monitoring                                   │
│            ├── PSI weekly (top-10 features)                      │
│            ├── Rolling AUC/RMSE window                           │
│            └── Prediction distribution shift                     │
│                                                                  │
│  Drift detected → Trigger retrain                                │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 Monitoring in Production

| Model | Drift Metric | Alert Threshold | Action |
|---|---|---|---|
| Fraud | PSI on top-10 features | PSI > 0.2 | SQL Alert → Trigger retrain |
| Fraud | Rolling AUC (30 days) | AUC < 0.85 | Team notification |
| Purchase | Wasserstein distance | > calibrated threshold | SQL Alert → Trigger retrain |
| Purchase | MAPE per tier | > 20% any tier | Team notification |
| Both | Inference latency P95 | > 200ms | Scale-up endpoint |

---

## 6. Multi-Tool Agent Design

### 6.1 LangGraph StateGraph

```python
class CopilotState(TypedDict):
    user_query: str
    intent_type: Optional[str]           # data_query | prediction_fraud | prediction_purchase | knowledge | complex
    intent_confidence: Optional[float]    # 0.0 - 1.0
    data_result: Optional[dict]
    prediction_result: Optional[dict]
    knowledge_result: Optional[dict]
    load_assessment: Optional[dict]       # cognitive load score + level
    final_response: Optional[str]
    tools_used: Optional[list]
    sources_cited: Optional[list]
```

The graph follows this execution flow:
```
classify_intent → [tool_node] → assess_load → synthesize → END
```

Where `[tool_node]` is one of: `data_query`, `predict_fraud`, `predict_purchase`, `search_knowledge`, or `complex_analysis`.

### 6.2 Agent Tools

| Tool | Input | Output | Source |
|---|---|---|---|
| `query_fraud_data` | Natural language query | Formatted results + counts | Delta table (in-memory pandas) |
| `query_purchase_data` | Natural language query | Tier comparison + statistics | Delta table (in-memory pandas) |
| `run_fraud_model` | Transaction features dict | risk_score, risk_tier, SHAP top-5, action | MLflow model (Pipeline) |
| `run_purchase_model` | Customer features dict | predicted_amount, confidence interval, SHAP | MLflow model (Pipeline) |
| `search_financial_docs` | Query string | Top-5 chunks + source citations | Vector Search / TF-IDF |
| `assess_analyst_load` | Session metrics dict | load_score (0-100), level | Heuristic algorithm |

### 6.3 Intent Classification

The `classify_intent` node uses the LLM (`databricks-meta-llama-3-1-405b-instruct`) with a structured prompt that defines 5 categories in priority order: `complex` > `prediction_fraud` > `prediction_purchase` > `data_query` > `knowledge`.

Decision rules are explicit in the prompt:
- Query asks to FIND + EXPLAIN or ANALYZE + ASSESS → `complex`
- Query describes ONE specific transaction/customer to predict → `prediction_*`
- Query asks for simple numbers or lists → `data_query`
- Query is about regulations/procedures/theory → `knowledge`

The LLM returns JSON with intent and confidence. If confidence < 0.8, the system could request clarification (guardrail, not yet implemented in prototype).

### 6.4 Complex Query Handling

Complex queries are not simply routed to a single tool. The `complex_analysis` node chains multiple tools:

1. **Data retrieval:** Query the fraud dataset with a composite suspicion score
2. **Model predictions:** Run the fraud model on the top-5 riskiest transactions
3. **Aggregation:** Count HIGH/MEDIUM/LOW risk tiers across analyzed transactions

This multi-step approach is implemented as a dedicated node rather than letting the LLM decide the sequence, which provides deterministic behavior and testability.

### 6.5 Policy and Guardrails

1. **Regulatory questions** (KYC, AML, PCI DSS): Force exclusive use of `search_financial_docs`. Never generate regulatory answers from LLM knowledge without citing source documents.
2. **Predictions:** Always accompany with SHAP explanation. Never present a probability without context of which features drive it.
3. **Sensitive data:** Do not expose individual customer data unless the analyst provides a specific `customer_id`.
4. **Low confidence:** If `classify_intent` confidence < 0.8, request clarification instead of assuming.

---

## 7. Concurrency and Scalability

### 7.1 Concurrency Model

Concurrency is resolved at three independent levels:

**Level 1 — Model Serving (Agent Endpoint):** Each replica is an independent Python process executing the LangGraph graph. Databricks auto-scales horizontally. For 100 analysts at ~1 query per 2 minutes, expected peak is ~50 concurrent requests.

**Level 2 — Model Serving (ML Endpoints):** ML endpoints scale independently from the agent. If prediction queries increase relative to analytics, only ML endpoints scale.

**Level 3 — SQL Warehouse:** Native auto-scaling with cluster policies.

### 7.2 Real Bottleneck: LLM API

The bottleneck is not LangGraph or Databricks — it is LLM latency. Each agent invocation makes 1-3 LLM calls (classify + synthesize), with 1-5 seconds latency each.

**Mitigation:** Mosaic AI Gateway for rate limiting and fallback, prompt caching for static system prompts, and tier routing where simple queries (40% of traffic) bypass the full LLM pipeline.

### 7.3 Session State

**Prototype:** In-memory state (single instance).  
**Production options:** (1) Persist conversation history in Delta table (~100ms overhead per read) or (2) sticky sessions via load balancer configuration.

---

## 8. LLM Evaluation and Control

### 8.1 Evaluation Framework

The golden set contains 30 queries distributed across 5 intent types. Each query specifies:
- Expected intent classification
- Expected tools to be invoked
- Keywords that should appear in the response

### 8.2 Metrics

| Metric | Source | Target |
|---|---|---|
| Intent routing accuracy | Golden set vs expected | > 90% |
| Tool selection accuracy | Golden set vs expected tools | > 90% |
| Content relevance | Keyword match scoring | > 80% |
| Response rate | Non-empty responses > 50 chars | > 95% |
| E2E latency P50 | MLflow Traces | < 3s |
| E2E latency P95 | MLflow Traces | < 8s |

### 8.3 Future: LLM-as-Judge

The prototype uses keyword matching for content evaluation. With more time, `mlflow.genai.evaluate()` with scorers (Correctness, RetrievalGroundedness, Safety) would provide more robust quality assessment.

---

## 9. Analyst Augmentation: Cognitive Load

### 9.1 Concept

Not a sentiment model. It is an operational metrics system that estimates analyst cognitive load based on observable interaction signals.

### 9.2 Signals and Inputs

| Signal | How Captured | Weight |
|---|---|---|
| Queries in last hour | Session counter | 20% |
| Recent case complexity | Average routing tier (1-4) | 20% |
| Time in session | First query timestamp - now | 15% |
| Query velocity | Decreasing interval between queries | 15% |
| Follow-up rate | % queries that are clarifications | 15% |
| Time of day | Circadian fatigue factor | 15% |

### 9.3 Actions by Level

| Score | Level | Copilot Action |
|---|---|---|
| 0-30 | Normal | Standard responses, full detail |
| 31-60 | Elevated | More structured, executive summary first |
| 61-80 | High | Simplified format, offer "need more detail?" |
| 81-100 | Critical | Minimal responses, suggest break, offer handoff |

---

## 10. Cost Analysis

### 10.1 Estimation by Component

Cost estimates are based on published Databricks DBU rates per service type (Model Serving CPU at ~$0.07/DBU, Serverless SQL at ~$0.70/DBU, Foundation Model APIs at token-based pricing), multiplied by estimated concurrency levels and operating hours for each component. The production scenario assumes 100 analysts generating ~5,000 queries/day with the traffic distribution described in the routing section; actual costs require validation through a proof-of-concept with representative workload, as real DBU consumption depends on query complexity, auto-scaling behavior, and scale-to-zero configurations that cannot be predicted from documentation alone. The tier routing optimization (Section 10.2) is the primary cost lever — routing 40% of traffic as direct SQL queries that bypass the LLM reduces the weighted average cost per query from ~$0.05 to ~$0.03.

| Component | Prototype (month) | Production 100 analysts (month) |
|---|---|---|
| Databricks Workspace | $0 (trial) | $2,000 - $5,000 |
| Agent Serving Endpoint | $0 (trial) | $1,500 - $3,000 |
| Fraud + Purchase Model Endpoints | $0 (trial) | $500 - $1,000 |
| LLM API (via AI Gateway) | $50 - $150 | $3,000 - $7,000 |
| Vector Search Index | $0 (trial) | $100 - $300 |
| **Total** | **$50 - $150** | **$7,200 - $16,600** |

### 10.2 Cost per Query Type

| Type | % Traffic | LLM Calls | Estimated Cost/Query |
|---|---|---|---|
| Data analysis (SQL) | 40% | 1 (classify) | $0.005 |
| Prediction | 25% | 2 (classify + synthesize) | $0.02 |
| Regulatory knowledge | 20% | 2 (classify + RAG synthesize) | $0.04 |
| Complex query | 15% | 3 (classify + tools + synthesize) | $0.08 |
| **Weighted average** | 100% | ~1.7 | **$0.03** |

---

## 11. Implementation Roadmap

### Phase 1: Foundation (Days 1-2)
- Load CSVs as Delta tables in Databricks
- Train fraud model (XGBoost Pipeline) with MLflow
- Train purchase model (LightGBM Pipeline) with MLflow
- Register both models in Unity Catalog
- Create Vector Search index over financial documents
- Implement tools as Python functions with `@mlflow.trace`

### Phase 2: Agent (Days 2-3)
- Implement LangGraph StateGraph with defined nodes
- Connect tools to graph (data queries, model invocation, RAG)
- Implement intent classification with structured LLM prompt
- Enable `mlflow.langchain.autolog()` for tracing

### Phase 3: Evaluation and Polish (Day 3)
- Create golden set of 30 queries
- Run evaluation with routing accuracy + content scoring
- Implement `assess_analyst_load` node (heuristic)
- Prepare 5-10 minute demo walkthrough
- Document trade-offs and future improvements

---

## 12. Implementation Iterations and Lessons Learned

This section documents the key iterations, bugs, and workarounds encountered during prototype development.

### 12.1 Library Installation: pip Doesn't Work on Free Tier

**Problem:** The `%pip install` magic command is unreliable on the Databricks Community Edition. Packages would fail silently, install partial dependencies, or not persist after cluster restart.

**Impact:** Initial attempts to install `xgboost`, `lightgbm`, `shap`, `langgraph`, and `databricks-langchain` via `%pip` inside notebooks failed or produced inconsistent environments.

**Solution:** All libraries were installed manually through the Databricks cluster UI: **Cluster > Libraries > Install New > PyPI**. This method is persistent across cluster restarts and does not suffer from the notebook-level pip issues.

**Lesson:** In constrained environments, always verify the package management path before writing code. The "obvious" approach (pip in notebook) was the wrong one here.

### 12.2 Train-Serve Skew: Data Type Mismatch

**Problem:** The initial model implementation applied `OrdinalEncoder` outside the sklearn Pipeline during training. When the model was registered and the agent sent raw string data (`"electronics"` for `merchant_category`), MLflow rejected the input because the model signature expected integers.

**Root cause:** Preprocessing was decoupled from the model. The training notebook encoded categoricals to integers, then trained the model on integers. But the agent had no way to apply the same encoding at inference time.

**Fix:** Both models were refactored to use `sklearn.Pipeline` with `ColumnTransformer` as the first step:
```
Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", "passthrough", numeric_features),
        ("bin", "passthrough", binary_features),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_features),
    ])),
    ("classifier", XGBClassifier(...))
])
```

The Pipeline is what gets registered in MLflow. The model signature now reflects the raw input types (strings for categoricals), and encoding happens internally at both training and inference time.

**Lesson:** Never decouple preprocessing from the model in production systems. The Pipeline pattern is non-negotiable for ML systems that will be served via API.

### 12.3 SHAP Extraction from Pipeline Models

**Problem:** `shap.TreeExplainer` cannot receive a full sklearn Pipeline. It needs the underlying tree model (`XGBClassifier` or `LGBMRegressor`) directly.

**Solution:** Load the model twice:
1. As `mlflow.pyfunc.load_model()` for inference — this uses the full Pipeline and accepts raw data.
2. As `mlflow.sklearn.load_model()` for SHAP access — extract `pipeline.named_steps["classifier"]` and `pipeline.named_steps["preprocessor"]`.

For SHAP computation: transform the input using the preprocessor, then pass the transformed data to `TreeExplainer(underlying_model)`.

### 12.4 int64 Casting Issue

**Problem:** Pandas defaults integer columns to `int64`, but the MLflow model signature (inferred during training) sometimes records `int32`. At prediction time, the signature validator rejects `int64` inputs.

**Fix:** Both model tools explicitly cast `int64` columns to `int32` and `float64` stays as `float64` before calling `model.predict()`.

### 12.5 Vector Search Index Sync Delay

**Problem:** The Vector Search index creation in notebook 00 is asynchronous. When notebook 03 executes immediately after, the index may still be syncing.

**Solution:** The RAG tool probes Vector Search availability at startup. If the index is not ready, it falls back to a TF-IDF-based retrieval path (pre-built from the same document chunks). This ensures the agent demo works regardless of index state.

### 12.6 LLM Rate Limits During Evaluation

**Problem:** Each agent invocation requires 2+ LLM calls (classify + synthesize). Running 30 evaluation queries sequentially exceeds the free-tier rate limit (~20 requests/minute).

**Solution:** The evaluation notebook splits the golden set into 5 batches of 6 queries, with manual cell execution between batches. In production with provisioned throughput endpoints, all queries would run in a single batch.

---

## 13. Trade-offs and Future Improvements

### 13.1 Explicit Trade-offs

| Decision | Benefit | Cost |
|---|---|---|
| Databricks-centric | Maintainability, security, native observability | Vendor lock-in |
| LangGraph over pure Python | Visual graph, testable state machine | Framework dependency |
| XGBoost over deep learning | Works with 100 rows, SHAP-compatible | Lower ceiling with massive data |
| Heuristic cognitive load | Implementable without historical data | Requires real-data validation |
| Pattern-matching data tools | Fast to implement, deterministic | Limited query flexibility |
| LLM-based intent classifier | No training data needed | Higher latency/cost per query |

### 13.2 Improvements With More Time

1. **Text-to-SQL data tools** — Replace keyword matching with LLM-generated SQL for arbitrary analytical queries, with validation guardrails.
2. **Semantic caching** — Vector similarity on previous queries to avoid repeated LLM calls. Estimated 25-40% cache hit rate on regulatory queries.
3. **Fine-tuned intent classifier** — Dedicated distilbert model for routing. Reduces latency and cost of the first hop.
4. **Conversation memory** — Persist conversation context per analyst session for multi-turn investigations.
5. **LLM-based feature extraction** — Replace regex parsing with structured LLM output for extracting transaction/customer features from natural language.
6. **Graph database (Neo4j)** — Model entity relationships for organized fraud ring detection.
7. **Real-time streaming** — Databricks Structured Streaming for proactive alerts.
8. **Production CI/CD** — Databricks Asset Bundles + GitHub Actions for automated deployment.
9. **mlflow.genai.evaluate()** — Full LLM-as-judge evaluation with Correctness, Groundedness, Safety scorers.
10. **Cognitive load ML model** — Train on real analyst session data instead of heuristic.

---
