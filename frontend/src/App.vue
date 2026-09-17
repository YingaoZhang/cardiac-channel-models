<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import heroImage from './assets/hero.png'

const defaultSmiles = 'CCN(CC)CCOC(=O)c1ccc(N)cc1'
const pollDelay = 1200

const smiles = ref(defaultSmiles)
const loading = ref(false)
const reindexing = ref(false)
const error = ref('')
const result = ref(null)
const task = ref(null)
const health = ref(null)
const reindexResult = ref(null)
const pollTimer = ref(null)

const sampleMolecules = [
  { label: '默认示例', value: defaultSmiles },
  { label: 'Dofetilide', value: 'CS(=O)(=O)NCCCOc1ccc(CCNCCc2ccc(OS(=O)(=O)C)cc2)cc1' },
  { label: 'Verapamil', value: 'COc1ccc(CCN(C)CCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC' },
  { label: '咖啡因', value: 'Cn1cnc2n(C)c(=O)n(C)c(=O)c12' },
  { label: '普萘洛尔', value: 'CC(C)NCC(O)COc1cccc2ccccc12' },
]

const agents = [
  { key: 'Toxicity', title: '初始毒性评估', desc: '三通道预测与 MICE 风险', short: 'Risk' },
  { key: 'Optimization', title: '分子结构优化', desc: '生成候选结构与改造建议', short: 'Design' },
  { key: 'ReToxicity', title: '候选毒性复测', desc: '复查优化后的风险变化', short: 'Verify' },
  { key: 'Synthesis', title: '多目标复审', desc: '毒性、活性代理、类药性、相似性与合成校验', short: 'Review' },
]

const activeResult = computed(() => task.value?.result ?? result.value)
const hasResult = computed(() => Boolean(activeResult.value))
const progress = computed(() => task.value?.progress ?? (result.value ? 100 : 0))
const statusText = computed(() => task.value?.message ?? (result.value ? '分析完成' : '等待任务'))
const currentRound = computed(() => activeResult.value?.optimization_round ?? 0)
const maxRounds = computed(() => activeResult.value?.max_optimization_rounds ?? 3)
const agentTrace = computed(() => activeResult.value?.agent_trace ?? [])
const candidateHistory = computed(() => activeResult.value?.candidate_history ?? [])
const ragItems = computed(() => activeResult.value?.rag_context ?? [])
const candidateRagItems = computed(() => activeResult.value?.modified_rag_context ?? [])
const displayedChannels = computed(() =>
  activeResult.value?.modified_channel_predictions ?? activeResult.value?.channel_predictions ?? {},
)
const molecularReview = computed(() => activeResult.value?.molecular_review ?? {})

const structureCases = computed(() =>
  ragItems.value.filter((item) => item.kind === 'structure_case'),
)

const knowledgeChunks = computed(() =>
  ragItems.value.filter((item) => item.kind === 'knowledge_chunk'),
)

const metrics = computed(() => [
  {
    label: 'MICE TdP 概率',
    value: formatPercent(activeResult.value?.modified_toxicity_prob ?? activeResult.value?.toxicity_prob),
    tone: riskTone.value,
    hint: activeResult.value?.modified_tdp ? '候选分子' : '原始分子',
  },
  {
    label: 'hERG 阻滞概率',
    value: formatPercent(displayedChannels.value?.herg?.blocker_probability),
    tone: 'neutral',
    hint: formatChannelPotency(displayedChannels.value?.herg),
  },
  {
    label: 'Cav1.2 阻滞概率',
    value: formatPercent(displayedChannels.value?.['cav1.2']?.blocker_probability),
    tone: 'neutral',
    hint: formatChannelPotency(displayedChannels.value?.['cav1.2']),
  },
  {
    label: 'Nav1.5 阻滞概率',
    value: formatPercent(displayedChannels.value?.['nav1.5']?.blocker_probability),
    tone: 'neutral',
    hint: formatChannelPotency(displayedChannels.value?.['nav1.5']),
  },
  {
    label: 'SA Score',
    value: formatValue(activeResult.value?.sa_score),
    tone: activeResult.value?.synthesis_feasible === false ? 'warning' : 'neutral',
    hint: '合成难度',
  },
  {
    label: '结构相似度',
    value: formatValue(molecularReview.value?.structure_similarity?.score),
    tone: molecularReview.value?.structure_similarity?.passed === false ? 'warning' : 'neutral',
    hint: 'Morgan-Tanimoto · 活性保持代理',
  },
  {
    label: 'QED',
    value: formatValue(molecularReview.value?.drug_likeness?.qed),
    tone: molecularReview.value?.drug_likeness?.passed === false ? 'warning' : 'neutral',
    hint: `Lipinski ${molecularReview.value?.drug_likeness?.lipinski_violation_count ?? '-'} 项违例`,
  },
  {
    label: '优化结论',
    value: conclusionText.value,
    tone: conclusionTone.value,
    hint: `${currentRound.value}/${maxRounds.value} 轮`,
  },
])

const workflowSteps = computed(() =>
  agents.map((agent, index) => ({
    ...agent,
    index: index + 1,
    state: stepState(agent.key),
  })),
)

const riskTone = computed(() => toneFromRisk(activeResult.value?.risk_level))
const candidateTone = computed(() => toneFromRisk(activeResult.value?.modified_risk_level))

const conclusionText = computed(() => {
  if (!activeResult.value) return '-'
  if (activeResult.value.optimization_success) return '成功'
  if (activeResult.value.optimization_failed) return '失败'
  if (loading.value) return '迭代中'
  return '待判断'
})

const conclusionTone = computed(() => {
  if (activeResult.value?.optimization_success) return 'success'
  if (activeResult.value?.optimization_failed) return 'danger'
  if (loading.value) return 'warning'
  return 'neutral'
})

const systemRows = computed(() => [
  {
    label: '毒性模型',
    value: health.value?.toxicity_backend === 'xgboost-three-channel' ? '三通道 XGBoost' : '-',
  },
  {
    label: 'Agent 模型',
    value: health.value?.agent_model ?? '-',
  },
  {
    label: 'RAG 后端',
    value: health.value?.rag_backend ?? '-',
  },
  {
    label: 'pgvector',
    value: health.value?.pgvector?.status ?? '-',
  },
  {
    label: '文档切片',
    value: health.value?.pgvector?.document_count ?? 0,
  },
  {
    label: '文献目录',
    value: shortPath(health.value?.rag_source_dir),
  },
])

function toneFromRisk(level) {
  const value = String(level ?? '').toLowerCase()
  if (value.includes('high')) return 'danger'
  if (value.includes('medium')) return 'warning'
  if (value.includes('low')) return 'success'
  return 'neutral'
}

function stepState(agentKey) {
  if (!task.value && !result.value) return 'idle'
  if (task.value?.current_step === agentKey) return 'active'
  if (agentTrace.value.some((item) => agentNameMatches(item.agent, agentKey))) return 'done'
  return 'idle'
}

function agentNameMatches(name, key) {
  const text = String(name ?? '')
  const map = {
    Toxicity: '初始毒性',
    Optimization: '药物化学',
    ReToxicity: '毒性复测',
    Synthesis: '合成评估',
  }
  return text.includes(map[key]) || text.includes(key)
}

function formatValue(value) {
  if (value === undefined || value === null || value === '') return '-'
  if (typeof value === 'number') return value.toFixed(2)
  return value
}

function formatPercent(value) {
  if (value === undefined || value === null || value === '') return '-'
  const numeric = Number(value)
  if (Number.isNaN(numeric)) return value
  return `${(numeric * 100).toFixed(1)}%`
}

function formatChannelPotency(channel) {
  if (!channel) return 'pIC50 -'
  return `pIC50 ${Number(channel.pIC50).toFixed(2)} · IC50 ${Number(channel.ic50_uM).toFixed(2)} μM`
}

function shortPath(path) {
  if (!path) return 'unknown'
  return String(path).split(/[\\/]/).slice(-2).join('/')
}

function useSample(value) {
  smiles.value = value
}

function stopPolling() {
  if (pollTimer.value) {
    clearTimeout(pollTimer.value)
    pollTimer.value = null
  }
}

async function loadHealth() {
  try {
    const response = await fetch('/api/health')
    health.value = await response.json()
  } catch (err) {
    health.value = { status: 'error', message: err.message }
  }
}

async function pollTask(taskId) {
  try {
    const response = await fetch(`/api/discover/${taskId}`)
    if (!response.ok) throw new Error(`任务状态读取失败：${response.status}`)

    task.value = await response.json()
    if (task.value.result) result.value = task.value.result

    if (task.value.status === 'completed') {
      loading.value = false
      stopPolling()
      return
    }

    if (task.value.status === 'failed') {
      throw new Error(task.value.error || '任务执行失败')
    }

    pollTimer.value = setTimeout(() => pollTask(taskId), pollDelay)
  } catch (err) {
    loading.value = false
    error.value = err.message
    stopPolling()
  }
}

async function runDiscovery() {
  const value = smiles.value.trim()
  if (!value) {
    error.value = '请输入 SMILES。'
    return
  }

  stopPolling()
  loading.value = true
  error.value = ''
  result.value = null
  task.value = null

  try {
    const response = await fetch('/api/discover', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ smiles: value }),
    })

    if (!response.ok) {
      throw new Error(`请求失败：${response.status}`)
    }

    task.value = await response.json()
    pollTimer.value = setTimeout(() => pollTask(task.value.task_id), 300)
  } catch (err) {
    loading.value = false
    error.value = err.message
  }
}

async function rebuildIndex() {
  reindexing.value = true
  error.value = ''
  reindexResult.value = null

  try {
    const response = await fetch('/api/rag/reindex', { method: 'POST' })
    if (!response.ok) {
      throw new Error(`索引重建失败：${response.status}`)
    }
    reindexResult.value = await response.json()
    await loadHealth()
  } catch (err) {
    error.value = err.message
  } finally {
    reindexing.value = false
  }
}

onMounted(loadHealth)
onBeforeUnmount(stopPolling)
</script>

<template>
  <main class="app-shell">
    <aside class="sidebar">
      <section class="brand-card">
        <div>
          <p class="eyebrow">AI Drug Discovery</p>
          <h1>三通道心脏安全工作台</h1>
          <p>hERG、Cav1.2、Nav1.5 基线模型与多智能体闭环优化。</p>
        </div>
        <div class="brand-mark">CiPA</div>
      </section>

      <section class="visual-card" aria-label="平台能力概览">
        <img :src="heroImage" alt="分子研发平台层级示意" />
        <div class="visual-meta">
          <span>Local Model</span>
          <span>Hybrid RAG</span>
          <span>3 Rounds</span>
        </div>
      </section>

      <section class="panel input-panel">
        <div class="section-title">
          <div>
            <span>分子输入</span>
            <p>输入或选择一个 SMILES，开始毒性评估与结构优化。</p>
          </div>
          <small>SMILES</small>
        </div>

        <textarea v-model="smiles" spellcheck="false" placeholder="输入待评估分子的 SMILES" />

        <div class="sample-row" aria-label="示例分子">
          <button
            v-for="sample in sampleMolecules"
            :key="sample.label"
            class="sample-chip"
            type="button"
            @click="useSample(sample.value)"
          >
            {{ sample.label }}
          </button>
        </div>

        <div class="actions">
          <button class="primary" type="button" :disabled="loading" @click="runDiscovery">
            <svg class="icon" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M8 5v14l11-7z" />
            </svg>
            {{ loading ? '运行中' : '运行分析' }}
          </button>
          <button class="secondary" type="button" :disabled="reindexing" @click="rebuildIndex">
            <svg class="icon" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M17.7 6.3A8 8 0 1 0 20 12h-2a6 6 0 1 1-1.76-4.24L13 11h8V3z" />
            </svg>
            {{ reindexing ? '重建中' : '重建索引' }}
          </button>
        </div>

        <p v-if="error" class="error">{{ error }}</p>
      </section>

      <section class="panel status-panel">
        <div class="section-title">
          <div>
            <span>系统状态</span>
            <p>当前后端、向量库与文献索引状态。</p>
          </div>
          <small>{{ health?.status ?? 'loading' }}</small>
        </div>
        <dl class="status-list">
          <div v-for="row in systemRows" :key="row.label">
            <dt>{{ row.label }}</dt>
            <dd>{{ row.value }}</dd>
          </div>
        </dl>
        <p v-if="reindexResult" class="mini-note">
          已索引 {{ reindexResult.document_count }} 个片段，向量维度
          {{ reindexResult.embedding_shape?.join(' x ') }}
        </p>
      </section>
    </aside>

    <section class="workspace">
      <header class="hero-header">
        <div>
          <p class="eyebrow">Molecular Safety Console</p>
          <h2>把三通道预测、MICE 风险、候选优化和证据检索放进同一个工作流。</h2>
          <p>
            输入分子后，系统会依次完成原始毒性判断、候选结构设计、复测与合成可行性评估，
            并把 RAG 命中的结构案例和文献片段一起展示出来。
          </p>
        </div>
        <div class="hero-status">
          <div class="status-pill" :class="{ running: loading }">{{ statusText }}</div>
          <div class="progress-number">{{ progress }}%</div>
          <small>第 {{ currentRound }} / {{ maxRounds }} 轮</small>
        </div>
      </header>

      <section class="metric-grid">
        <article v-for="metric in metrics" :key="metric.label" class="metric" :class="metric.tone">
          <span>{{ metric.label }}</span>
          <strong>{{ metric.value }}</strong>
          <small>{{ metric.hint }}</small>
        </article>
      </section>

      <section class="workflow-panel">
        <div class="section-title">
          <div>
            <span>智能体流程</span>
            <p>四个节点按顺序协作，当前节点会实时高亮。</p>
          </div>
          <small>{{ loading ? 'running' : 'standby' }}</small>
        </div>
        <div class="progress-track">
          <div class="progress-fill" :style="{ width: `${progress}%` }"></div>
        </div>
        <div class="agent-board">
          <article v-for="agent in workflowSteps" :key="agent.key" class="agent-node" :class="agent.state">
            <div class="agent-index">{{ agent.index }}</div>
            <div>
              <span>{{ agent.short }}</span>
              <strong>{{ agent.title }}</strong>
              <p>{{ agent.desc }}</p>
            </div>
          </article>
        </div>
      </section>

      <section v-if="!hasResult && !loading" class="empty-state">
        <div>
          <p class="eyebrow">Ready</p>
          <h3>等待第一个分子任务</h3>
          <p>
            左侧输入 SMILES 后，页面会自动展开指标卡、协作轨迹、候选历史、相似结构案例、
            文献检索证据和最终报告。
          </p>
        </div>
        <div class="empty-steps">
          <span>1. 预测三通道与 TdP 风险</span>
          <span>2. 生成候选分子</span>
          <span>3. 复测并判断是否收敛</span>
        </div>
      </section>

      <template v-if="activeResult">
        <section class="molecule-strip">
          <div>
            <span>原始分子</span>
            <code>{{ activeResult.original_smiles }}</code>
          </div>
          <div>
            <span>当前候选分子</span>
            <code>{{ activeResult.modified_smiles || activeResult.original_smiles }}</code>
          </div>
        </section>

        <section class="content-grid lower">
          <article class="panel">
            <div class="section-title">
              <div>
                <span>协作轨迹</span>
                <p>每个智能体的执行记录与中间状态。</p>
              </div>
              <small>{{ agentTrace.length }} events</small>
            </div>
            <div class="timeline">
              <div
                v-for="(item, index) in agentTrace"
                :key="`${item.agent}-${index}`"
                class="timeline-item"
                :class="item.status"
              >
                <span></span>
                <div>
                  <strong>第 {{ item.round }} 轮 · {{ item.agent }}</strong>
                  <p>{{ item.message }}</p>
                  <code v-if="item.smiles">{{ item.smiles }}</code>
                </div>
              </div>
              <p v-if="!agentTrace.length" class="muted">任务刚开始时，轨迹会在这里实时出现。</p>
            </div>
          </article>

          <article class="panel">
            <div class="section-title">
              <div>
                <span>候选轮次</span>
                <p>每轮候选分子的毒性概率和合成可行性。</p>
              </div>
              <small>{{ candidateHistory.length }} candidates</small>
            </div>
            <div class="candidate-list">
              <div v-for="item in candidateHistory" :key="`${item.round}-${item.smiles}`" class="candidate-item">
                <div class="evidence-head">
                  <strong>第 {{ item.round }} 轮</strong>
                  <span>{{ item.synthesis_feasible ? '可合成' : '需继续' }}</span>
                </div>
                <p>TdP：{{ formatPercent(item.toxicity_prob) }} · SA：{{ formatValue(item.sa_score) }}</p>
                <p>
                  相似度：{{ formatValue(item.molecular_review?.structure_similarity?.score) }} ·
                  QED：{{ formatValue(item.molecular_review?.drug_likeness?.qed) }}
                </p>
                <small>
                  hERG IC50 {{ item.constraints?.herg_ic50_improved ? '通过' : '未通过' }} ·
                  Nav1.5 IC50 {{ item.constraints?.['nav1.5_ic50_preserved'] ? '通过' : '未通过' }}
                </small>
                <code>{{ item.smiles }}</code>
              </div>
              <p v-if="!candidateHistory.length" class="muted">尚未产生候选分子。</p>
            </div>
          </article>
        </section>

        <section class="content-grid">
          <article class="panel">
            <div class="section-title">
              <div>
                <span>结构相似案例</span>
                <p>从知识库中召回的相似结构和风险标签。</p>
              </div>
              <small>{{ structureCases.length }} hits</small>
            </div>
            <div class="evidence-list">
              <div v-for="item in structureCases" :key="item.smiles" class="evidence">
                <div class="evidence-head">
                  <strong>{{ item.risk }}</strong>
                  <span>{{ item.score }}</span>
                </div>
                <code>{{ item.smiles }}</code>
              </div>
              <p v-if="!structureCases.length" class="muted">暂无结构案例命中。</p>
            </div>
          </article>

          <article class="panel">
            <div class="section-title">
              <div>
                <span>文献混合召回</span>
                <p>融合向量相似度与全文检索的证据片段。</p>
              </div>
              <small>{{ knowledgeChunks.length }} chunks</small>
            </div>
            <div class="evidence-list">
              <div v-for="item in knowledgeChunks" :key="`${item.source}-${item.chunk_index}`" class="evidence">
                <div class="evidence-head">
                  <strong>{{ shortPath(item.source) }} #{{ item.chunk_index }}</strong>
                  <span>{{ item.score }}</span>
                </div>
                <p>{{ item.text }}</p>
                <small v-if="item.dense_score !== undefined">
                  dense {{ item.dense_score }} · sparse {{ item.sparse_score }}
                </small>
              </div>
              <p v-if="!knowledgeChunks.length" class="muted">暂无文献片段命中。</p>
            </div>
          </article>
        </section>

        <section class="content-grid lower">
          <article class="panel">
            <div class="section-title">
              <div>
                <span>优化记录</span>
                <p>结构改造理由和每轮决策说明。</p>
              </div>
              <small>{{ activeResult.optimization_history?.length ?? 0 }} notes</small>
            </div>
            <div class="note-list">
              <p v-if="!activeResult.optimization_history?.length">当前分子未触发结构优化。</p>
              <p v-for="item in activeResult.optimization_history" :key="item">{{ item }}</p>
            </div>
          </article>

          <article class="panel">
            <div class="section-title">
              <div>
                <span>候选分子 RAG</span>
                <p>候选结构对应的证据召回。</p>
              </div>
              <small>{{ candidateRagItems.length }} items</small>
            </div>
            <div class="evidence-list compact">
              <div v-for="item in candidateRagItems" :key="`${item.source}-${item.chunk_index}-${item.score}`" class="evidence">
                <div class="evidence-head">
                  <strong>{{ item.kind === 'structure_case' ? item.risk : shortPath(item.source) }}</strong>
                  <span>{{ item.score }}</span>
                </div>
                <p>{{ item.text || item.smiles }}</p>
              </div>
              <p v-if="!candidateRagItems.length" class="muted">暂无候选分子证据。</p>
            </div>
          </article>
        </section>

        <section class="panel report">
          <div class="section-title">
            <div>
              <span>最终报告</span>
              <p>综合毒性、候选结构、合成可行性与证据来源。</p>
            </div>
            <small>{{ activeResult.final_report ? 'ready' : 'generating' }}</small>
          </div>
          <pre>{{ activeResult.final_report || '报告正在生成。' }}</pre>
        </section>
      </template>
    </section>
  </main>
</template>
