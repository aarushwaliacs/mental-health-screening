import { StrictMode, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Link, NavLink, Outlet, Route, Routes } from 'react-router-dom'
import { useForm } from 'react-hook-form'
import { z } from 'zod'
import { zodResolver } from '@hookform/resolvers/zod'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import {
  AlertTriangle, ArrowRight, CheckCircle2, Eye, FlaskConical, GitBranch, HeartHandshake, Info, Layers,
  ListChecks, Loader2, Lock, Menu, ShieldAlert, X,
} from 'lucide-react'
import './index.css'

/* TreeScreen frontend (single-file edition): types, API client, hooks, UI, pages, router and mount point. */

// -------- types/api.ts --------
export type Level = 'Low' | 'Moderate' | 'High'

export interface ScaleOption { value: number; label: string }
export interface Scale { id: string; options: ScaleOption[] }
export interface QuestionItem {
  id: string
  text: string
  short: string
  scale: string
  domain: string
  safety_critical: boolean
}
export interface QuestionSection {
  id: string
  title: string
  standardized: boolean
  instrument: string
  stem: string
  time_period: string
  provenance: string
  items: QuestionItem[]
}
export interface Questionnaire {
  title: string
  scales: Scale[]
  sections: QuestionSection[]
  item_count: number
  validation_notice: string
}

export interface ResourceEntry { name: string; contact: string; note?: string }
export interface CrisisResources {
  region: string
  verification_notice: string
  emergency: ResourceEntry[]
  helplines: ResourceEntry[]
  directory: ResourceEntry[]
  human_support: string[]
}
export interface Safety {
  triggered: boolean
  level: 'none' | 'support' | 'urgent'
  trigger: string | null
  message: string | null
  resources: CrisisResources | null
  note: string
}

export interface PathStep {
  node_id: number
  feature: string
  feature_label: string
  value: number
  threshold: number
  direction: 'left' | 'right'
  satisfied_condition: string
  text: string
}
export interface LeafEvidence {
  n_train_samples: number
  class_counts: Record<string, number>
  class_share: Record<string, number>
  description: string
}
export interface DecisionPath {
  node_ids: number[]
  steps: PathStep[]
  leaf_id: number
  prediction: string
  leaf_evidence: LeafEvidence
}
export interface KeyFactor { feature: string; label: string; description: string; value: number }
export interface ConditionResult {
  target: string
  display_name: string
  risk_level: Level
  meaning: string
  classes: Level[]
  decision_path: DecisionPath
  key_factors: KeyFactor[]
}
export interface ScreeningResult {
  status: 'ok' | 'model_unavailable'
  model: { available: boolean; run_id: string | null; is_synthetic: boolean | null; reason: string | null }
  safety: Safety
  reference_scores: {
    note: string
    phq9: { total: number; max: number; band: string }
    gad7: { total: number; max: number; band: string }
    supplementary: { total: number; max: number; note: string }
    stress: { available: boolean; note: string }
  }
  conditions: ConditionResult[]
  summary: { highest_level: Level; derived: boolean; note: string } | null
  features_used: Record<string, number> | null
  recommendations: string[]
  limitations: string[]
  disclaimer: string
}

export interface FeatureInfo { id: string; label: string; short: string; description: string; kind: string }
export interface TreeNode {
  id: number
  depth: number
  n_samples: number
  class_counts: Record<string, number>
  prediction: Level
  is_leaf: boolean
  feature?: string
  feature_label?: string
  threshold?: number
  left?: TreeNode
  right?: TreeNode
}
export interface Rule {
  leaf_id: number
  text: string
  prediction: Level
  n_samples: number
  class_counts: Record<string, number>
  class_share: Record<string, number>
  conditions: { feature: string; operator: string; threshold: number; text: string }[]
}
export interface Importance { feature: string; label: string; impurity: number; permutation_mean: number; permutation_std: number }
export interface TreeResponse {
  run_id: string
  is_synthetic: boolean
  target: string
  available_targets: { id: string; display_name: string }[]
  feature_info: Record<string, FeatureInfo>
  display_name: string
  classes: Level[]
  n_leaves: number
  depth: number
  selected_params: Record<string, unknown>
  tree: TreeNode
  rules: Rule[]
  feature_importance: Importance[]
  notes: string
}

export interface ModelStatus {
  trained: boolean
  message: string
  run_id?: string
  trained_at?: string
  is_synthetic?: boolean
  dataset?: { id: string; name: string; n_rows: number; is_synthetic: boolean; conditions_covered: string[] }
  targets: { id: string; display_name: string; classes: Level[] }[]
}

export interface PerClass { precision: number; recall: number; f1: number; support: number }
export interface Metrics {
  accuracy: number
  balanced_accuracy: number
  precision_macro: number
  recall_macro: number
  f1_macro: number
  roc_auc_ovr_macro: number | null
  per_class: Record<string, PerClass>
  confusion_matrix: number[][]
  n_test: number
}
export interface ModelEntry {
  display_name: string
  status: 'ok' | 'skipped' | 'failed'
  reason?: string
  selected_params?: Record<string, unknown>
  n_candidates?: number
  cv?: Record<string, { mean: number; std: number }>
  metrics?: Metrics
  fit_seconds?: number
  tree?: { n_leaves: number; depth: number; n_nodes: number }
}
export interface TargetReport {
  display_name: string
  target_definition: string
  classes: Level[]
  class_distribution: { all: Record<string, number>; train: Record<string, number>; test: Record<string, number> }
  cv_folds: number
  primary_model: string
  models: Record<string, ModelEntry>
}
export interface EvaluationReport {
  run_id: string
  created_at: string
  duration_seconds?: number
  dataset: {
    id: string | null; name: string | null; provenance: string | null; is_synthetic: boolean; n_rows: number
    conditions_covered: string[]; duplicate_rows?: number | null
  }
  options: Record<string, unknown>
  feature_schema: { columns: string[]; derived_totals_included: boolean }
  preprocessing: string[]
  split: { strategy: string; test_size: number; seed: number; n_train: number; n_test: number }
  tuning: { method: string; selection_metric: string; cv_folds_requested: number }
  metric_definitions: Record<string, string>
  software_versions: Record<string, string | null>
  targets: Record<string, TargetReport>
  skipped_targets: Record<string, string>
  warnings: string[]
}
export type MetricsResponse = { available: false; message: string } | { available: true; report: EvaluationReport }

export interface ProjectInfo {
  title: string
  report_source_status: string
  research_problem: string
  objectives: { id: string; text: string; implemented_by: string }[]
  scope: string[]
  methodology: string[]
  research_gaps: string[]
  technology_stack: Record<string, string[]>
  limitations: string[]
  ethics: string[]
  references: string[]
  references_status: string
  implementation_references: string[]
}

export interface ValidationReport {
  ok: boolean
  n_rows: number
  n_columns: number
  errors: string[]
  warnings: string[]
  targets_present?: string[]
  trainable_targets?: string[]
  blocked_targets?: Record<string, string>
  class_distribution?: Record<string, Record<string, number>>
  missing_values?: Record<string, number>
  duplicate_rows?: number
  leakage_warnings?: string[]
  is_synthetic?: boolean | null
}
export interface DatasetMeta {
  id: string; name: string; provenance: string; is_synthetic: boolean; uploaded_at: string
  n_rows: number; trainable_targets: string[]; targets_present: string[]
}
export interface RunSummary {
  run_id: string; created_at: string; active: boolean
  dataset: { id: string; name: string; is_synthetic: boolean; n_rows: number }
  targets: string[]; selection_metric: string
  cart_metrics: Record<string, { accuracy: number; recall_macro: number; f1_macro: number; balanced_accuracy: number }>
  duration_seconds?: number
}
export interface Job {
  id: string; kind: string; status: 'queued' | 'running' | 'succeeded' | 'failed'
  progress: number; message: string; result: { run_id: string; activated: boolean } | null; error: string | null
}

// -------- services/api.ts --------
const API_BASE: string = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? '/api'

class ApiError extends Error {
  status: number
  detail: unknown
  constructor(status: number, message: string, detail?: unknown) {
    super(message)
    this.status = status
    this.detail = detail
  }
}

function describe(status: number, body: unknown): string {
  const d = (body as { detail?: unknown } | null)?.detail
  if (typeof d === 'string') return d
  if (Array.isArray(d)) return 'Some answers were not accepted. Please check the form and try again.'
  if (d && typeof d === 'object' && 'message' in d) return String((d as { message: unknown }).message)
  if (status === 429) return 'Too many requests. Please wait a moment and try again.'
  if (status >= 500) return 'The service had a problem. Please try again shortly.'
  return `Request failed (${status}).`
}

async function request<T>(path: string, init: RequestInit & { token?: string } = {}): Promise<T> {
  const { token, ...rest } = init
  const headers = new Headers(rest.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (rest.body && typeof rest.body === 'string') headers.set('Content-Type', 'application/json')
  let res: Response
  try {
    res = await fetch(`${API_BASE}${path}`, { ...rest, headers })
  } catch {
    throw new ApiError(0, 'Cannot reach the server. Check your connection and that the backend is running.')
  }
  if (!res.ok) {
    let body: unknown = null
    try { body = await res.json() } catch { /* non-JSON error */ }
    throw new ApiError(res.status, describe(res.status, body), (body as { detail?: unknown } | null)?.detail)
  }
  return (await res.json()) as T
}

const api = {
  questionnaire: () => request<Questionnaire>('/questionnaire'),
  projectInfo: () => request<ProjectInfo>('/project-info'),
  modelStatus: () => request<ModelStatus>('/model/status'),
  metrics: () => request<MetricsResponse>('/model/metrics'),
  tree: (target?: string) => request<TreeResponse>(`/model/tree${target ? `?target=${encodeURIComponent(target)}` : ''}`),
  predict: (answers: Record<string, number>) =>
    request<ScreeningResult>('/screening/predict', { method: 'POST', body: JSON.stringify({ answers }) }),
}

const admin = {
  datasets: (token: string) => request<DatasetMeta[]>('/admin/datasets', { token }),
  evaluations: (token: string) => request<RunSummary[]>('/admin/evaluations', { token }),
  evaluation: (token: string, id: string) => request<EvaluationReport>(`/admin/evaluations/${id}`, { token }),
  validate: (token: string, file: File) => {
    const fd = new FormData(); fd.append('file', file)
    return request<ValidationReport>('/admin/dataset/validate', { method: 'POST', body: fd, token })
  },
  upload: (token: string, file: File, name: string, provenance: string, isSynthetic: boolean) => {
    const fd = new FormData()
    fd.append('file', file); fd.append('name', name); fd.append('provenance', provenance)
    fd.append('is_synthetic', String(isSynthetic))
    return request<{ dataset: DatasetMeta; validation: ValidationReport }>('/admin/dataset/upload', { method: 'POST', body: fd, token })
  },
  synthetic: (token: string, n_rows: number) =>
    request<{ dataset: DatasetMeta; validation: ValidationReport }>('/admin/dataset/synthetic', {
      method: 'POST', body: JSON.stringify({ n_rows }), token,
    }),
  train: (token: string, body: Record<string, unknown>) =>
    request<Job>('/admin/model/train', { method: 'POST', body: JSON.stringify(body), token }),
  job: (token: string, id: string) => request<Job>(`/admin/jobs/${id}`, { token }),
  activate: (token: string, id: string) =>
    request<{ active_run_id: string }>(`/admin/model/activate/${id}`, { method: 'POST', token }),
  exportRun: async (token: string, id: string, format: 'json' | 'csv'): Promise<Blob> => {
    const res = await fetch(`${API_BASE}/admin/evaluations/${id}/export?format=${format}`, {
      headers: { Authorization: `Bearer ${token}` },
    })
    if (!res.ok) throw new ApiError(res.status, 'Export failed.')
    return res.blob()
  },
}

// -------- utils/format.ts --------
const pct = (x: number | null | undefined, digits = 1): string =>
  x == null || Number.isNaN(x) ? '—' : `${(x * 100).toFixed(digits)}%`
const num3 = (x: number | null | undefined): string => (x == null ? '—' : x.toFixed(3))
const fmtDate = (iso?: string): string => (iso ? new Date(iso).toLocaleString() : '—')

const MODEL_KEYS = ['cart', 'random_forest', 'xgboost', 'lightgbm', 'logistic_regression'] as const
const MODEL_COLOR: Record<string, string> = {
  cart: 'var(--series-1)', random_forest: 'var(--series-2)', xgboost: 'var(--series-3)',
  lightgbm: 'var(--series-4)', logistic_regression: 'var(--series-5)',
}
const LEVEL_COLOR: Record<string, string> = {
  Low: 'var(--level-low)', Moderate: 'var(--level-moderate)', High: 'var(--level-high)',
}

// -------- utils/download.ts --------
function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

// -------- hooks/useApi.ts --------
export interface AsyncState<T> {
  data: T | null
  error: ApiError | null
  loading: boolean
  reload: () => void
}

/** Load data on mount / when deps change. Keeps stale data visible while reloading. */
function useApi<T>(fn: () => Promise<T>, deps: unknown[] = []): AsyncState<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [loading, setLoading] = useState(true)
  const [tick, setTick] = useState(0)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    fn()
      .then((d) => { if (!cancelled) { setData(d); setError(null) } })
      .catch((e) => { if (!cancelled) setError(e instanceof ApiError ? e : new ApiError(0, String(e))) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick])

  const reload = useCallback(() => setTick((t) => t + 1), [])
  return { data, error, loading, reload }
}

// -------- components/ui.tsx --------
function Card({ children, className = '', as: Tag = 'section' }: { children: ReactNode; className?: string; as?: 'section' | 'div' | 'article' }) {
  return <Tag className={`rounded-xl border p-5 sm:p-6 ${className}`} style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}>{children}</Tag>
}

function PageHeader({ title, lead, children }: { title: string; lead?: string; children?: ReactNode }) {
  return (
    <header className="mb-6">
      <h1 className="text-2xl sm:text-3xl font-semibold">{title}</h1>
      {lead && <p className="mt-2 max-w-3xl" style={{ color: 'var(--ink-2)' }}>{lead}</p>}
      {children}
    </header>
  )
}

type Tone = 'info' | 'warn' | 'danger' | 'ok'
const TONES: Record<Tone, { bg: string; ink: string; Icon: typeof Info }> = {
  info: { bg: 'var(--info-bg)', ink: 'var(--info-ink)', Icon: Info },
  warn: { bg: 'var(--warn-bg)', ink: 'var(--warn-ink)', Icon: AlertTriangle },
  danger: { bg: 'var(--danger-bg)', ink: 'var(--danger-ink)', Icon: ShieldAlert },
  ok: { bg: 'var(--ok-bg)', ink: 'var(--ok-ink)', Icon: CheckCircle2 },
}
function Notice({ tone = 'info', title, children, role }: { tone?: Tone; title?: string; children: ReactNode; role?: 'alert' | 'status' }) {
  const t = TONES[tone]
  return (
    <div role={role} className="flex gap-3 rounded-lg p-4 text-sm" style={{ background: t.bg, color: t.ink }}>
      <t.Icon aria-hidden className="mt-0.5 h-5 w-5 shrink-0" />
      <div>
        {title && <p className="font-semibold">{title}</p>}
        <div className={title ? 'mt-1' : ''}>{children}</div>
      </div>
    </div>
  )
}

function SyntheticBanner({ show = true }: { show?: boolean }) {
  if (!show) return null
  return (
    <div role="status" className="mb-5 flex gap-3 rounded-lg border-2 border-dashed p-3 text-sm" style={{ borderColor: 'var(--warn-ink)', background: 'var(--warn-bg)', color: 'var(--warn-ink)' }}>
      <FlaskConical aria-hidden className="mt-0.5 h-5 w-5 shrink-0" />
      <p><strong>Synthetic demonstration data.</strong> The model behind this page was trained on generated data to demonstrate the software. Nothing here describes real students or real predictive performance.</p>
    </div>
  )
}

function Spinner({ label = 'Loading…' }: { label?: string }) {
  return <div role="status" className="flex items-center gap-2 py-6" style={{ color: 'var(--ink-2)' }}><Loader2 className="h-5 w-5 animate-spin" aria-hidden />{label}</div>
}
function Skeleton({ h = 120 }: { h?: number }) {
  return <div className="skeleton w-full" style={{ height: h }} aria-hidden />
}

function ErrorState({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  return (
    <Notice tone="danger" title="Something went wrong" role="alert">
      <p>{error.message}</p>
      {onRetry && <button onClick={onRetry} className="mt-2 rounded-md border px-3 py-1.5 font-medium" style={{ borderColor: 'currentColor' }}>Try again</button>}
    </Notice>
  )
}

function EmptyState({ title, children, action }: { title: string; children: ReactNode; action?: ReactNode }) {
  return (
    <Card className="text-center">
      <h2 className="text-lg font-semibold">{title}</h2>
      <div className="mx-auto mt-2 max-w-xl" style={{ color: 'var(--ink-2)' }}>{children}</div>
      {action && <div className="mt-4">{action}</div>}
    </Card>
  )
}

const btn = 'inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 font-medium transition disabled:opacity-50 disabled:cursor-not-allowed min-h-[44px]'
function Button({ variant = 'primary', className = '', ...p }: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary' | 'secondary' | 'danger' }) {
  const style =
    variant === 'primary' ? { background: 'var(--brand)', color: 'var(--brand-ink)' }
    : variant === 'danger' ? { background: 'var(--danger-ink)', color: 'var(--danger-bg)' }
    : { background: 'var(--surface)', color: 'var(--ink)', border: '1px solid var(--border)' }
  return <button type="button" {...p} className={`${btn} ${className}`} style={style} />
}

function LevelBadge({ level }: { level: string }) {
  const tone: Record<string, string> = { Low: 'var(--ok-bg)', Moderate: 'var(--warn-bg)', High: 'var(--danger-bg)' }
  const ink: Record<string, string> = { Low: 'var(--ok-ink)', Moderate: 'var(--warn-ink)', High: 'var(--danger-ink)' }
  const mark: Record<string, string> = { Low: '●', Moderate: '◆', High: '▲' }
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-semibold" style={{ background: tone[level], color: ink[level] }}>
      <span aria-hidden>{mark[level]}</span>{level}
    </span>
  )
}

function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wide" style={{ color: 'var(--ink-3)' }}>{label}</div>
      <div className="text-xl font-semibold tabular-nums">{value}</div>
      {hint && <div className="text-xs" style={{ color: 'var(--ink-3)' }}>{hint}</div>}
    </div>
  )
}

// -------- components/SupportPanel.tsx --------
function SupportPanel({ safety }: { safety: Safety }) {
  if (!safety.triggered || !safety.resources) return null
  const r = safety.resources
  const urgent = safety.level === 'urgent'
  return (
    <section aria-labelledby="support-h" role="alert" className="mb-6 rounded-xl border-2 p-5 sm:p-6" style={{ borderColor: 'var(--danger-ink)', background: 'var(--danger-bg)', color: 'var(--danger-ink)' }}>
      <h2 id="support-h" className="flex items-center gap-2 text-xl font-semibold"><HeartHandshake aria-hidden className="h-6 w-6" />{urgent ? 'Please reach out for support today' : 'Support is available'}</h2>
      <p className="mt-2">{safety.message}</p>
      {safety.trigger && <p className="mt-2 text-sm opacity-90">{safety.trigger} triggered this message. It does not depend on the model result.</p>}
      <div className="mt-4 grid gap-4 md:grid-cols-2">
        <div>
          <h3 className="font-semibold">Emergency</h3>
          <ul className="mt-1 space-y-1">{r.emergency.map((e) => <li key={e.name}><strong>{e.contact}</strong> — {e.name}. {e.note}</li>)}</ul>
        </div>
        <div>
          <h3 className="font-semibold">Helplines</h3>
          <ul className="mt-1 space-y-1">{[...r.helplines, ...r.directory].map((e) => <li key={e.name}><strong>{e.contact}</strong> — {e.name}. {e.note}</li>)}</ul>
        </div>
      </div>
      <div className="mt-4">
        <h3 className="font-semibold">People who can help</h3>
        <ul className="mt-1 list-disc pl-5">{r.human_support.map((h) => <li key={h}>{h}</li>)}</ul>
      </div>
      <p className="mt-4 text-xs opacity-90">{r.verification_notice}</p>
    </section>
  )
}

// -------- components/DecisionPathView.tsx --------
function DecisionPathView({ path }: { path: DecisionPath }) {
  const ev = path.leaf_evidence
  return (
    <div>
      <ol className="space-y-2" aria-label="Decision path">
        {path.steps.map((s, i) => (
          <li key={s.node_id} className="flex gap-3 rounded-lg border p-3 text-sm" style={{ borderColor: 'var(--border)', background: 'var(--surface-2)' }}>
            <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full text-xs font-semibold" style={{ background: 'var(--brand)', color: 'var(--brand-ink)' }}>{i + 1}</span>
            <span>
              <strong>{s.satisfied_condition}</strong>
              <span style={{ color: 'var(--ink-2)' }}> — your value was {s.value}; went {s.direction === 'left' ? 'left (≤' : 'right (>'} {Number(s.threshold.toFixed(2))})</span>
            </span>
          </li>
        ))}
        {path.steps.length === 0 && <li className="text-sm">The tree is a single leaf; no conditions were evaluated.</li>}
      </ol>
      <p className="mt-3 text-sm"><strong>Outcome:</strong> {path.prediction}. Among {ev.n_train_samples} training respondents who ended in this same leaf, labels were{' '}
        {Object.entries(ev.class_counts).map(([c, n]) => `${c} ${n} (${pct(ev.class_share[c], 0)})`).join(', ')}.</p>
      <p className="mt-1 text-xs" style={{ color: 'var(--ink-3)' }}>{ev.description}</p>
    </div>
  )
}

// -------- components/TreeView.tsx --------
function counts(n: TreeNode) {
  return Object.entries(n.class_counts).map(([c, v]) => `${c} ${v}`).join(' · ')
}

function Node({ node, label, defaultOpen }: { node: TreeNode; label: string; defaultOpen: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  const tag = (
    <span className="ml-2 inline-flex items-center gap-1.5 rounded px-2 py-0.5 text-xs font-semibold" style={{ background: 'var(--surface-2)' }}>
      <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: LEVEL_COLOR[node.prediction] }} />
      {node.prediction}
    </span>
  )
  if (node.is_leaf) {
    return (
      <li className="py-1 text-sm" role="treeitem" aria-selected="false">
        <span style={{ color: 'var(--ink-3)' }}>{label}</span> <strong>Leaf</strong>{tag}
        <span className="ml-2" style={{ color: 'var(--ink-2)' }}>n={node.n_samples} ({counts(node)})</span>
      </li>
    )
  }
  const thr = node.threshold!
  const isHalf = Math.abs(thr - Math.floor(thr) - 0.5) < 1e-9
  const q = isHalf ? `${node.feature_label} ≤ ${Math.floor(thr)} ?` : `${node.feature_label} ≤ ${thr.toFixed(2)} ?`
  return (
    <li role="treeitem" aria-expanded={open} className="py-1 text-sm">
      <button onClick={() => setOpen(!open)} className="rounded px-1 text-left font-medium hover:underline" aria-label={`${open ? 'Collapse' : 'Expand'} ${q}`}>
        {open ? '▾' : '▸'} <span style={{ color: 'var(--ink-3)' }}>{label}</span> {q}
      </button>
      {tag}
      <span className="ml-2" style={{ color: 'var(--ink-2)' }}>n={node.n_samples}</span>
      {open && (
        <ul role="group" className="ml-4 border-l pl-3" style={{ borderColor: 'var(--border)' }}>
          <Node node={node.left!} label="yes →" defaultOpen={node.depth < 1} />
          <Node node={node.right!} label="no →" defaultOpen={node.depth < 1} />
        </ul>
      )}
    </li>
  )
}

function TreeView({ root }: { root: TreeNode }) {
  return (
    <ul role="tree" aria-label="Decision tree" className="overflow-x-auto">
      <Node node={root} label="start" defaultOpen />
    </ul>
  )
}

// -------- components/charts.tsx --------
const axis = { fill: 'var(--ink-2)', fontSize: 12 }
const tooltipStyle = { background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--ink)' }

export interface BarDatum { name: string; [series: string]: string | number }
export interface BarSeries { key: string; label: string; color: string }

function SimpleBars({ data, series, horizontal = false, height = 280, format, domain }: {
  data: BarDatum[]; series: BarSeries[]; horizontal?: boolean; height?: number; format?: (v: number) => string; domain?: [number, number]
}) {
  const fmt = format ?? ((v: number) => String(v))
  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer>
        <BarChart data={data} layout={horizontal ? 'vertical' : 'horizontal'} margin={{ top: 8, right: 16, bottom: 8, left: horizontal ? 24 : 0 }} barCategoryGap="22%">
          <CartesianGrid stroke="var(--grid)" strokeDasharray="3 3" vertical={horizontal} horizontal={!horizontal} />
          {horizontal ? (
            <>
              <XAxis type="number" tick={axis} tickFormatter={fmt} domain={domain} stroke="var(--grid)" />
              <YAxis type="category" dataKey="name" tick={axis} width={110} stroke="var(--grid)" />
            </>
          ) : (
            <>
              <XAxis dataKey="name" tick={axis} stroke="var(--grid)" interval={0} />
              <YAxis tick={axis} tickFormatter={fmt} domain={domain} stroke="var(--grid)" />
            </>
          )}
          <Tooltip contentStyle={tooltipStyle} formatter={(v) => fmt(Number(v))} cursor={{ fill: 'var(--surface-2)' }} />
          {series.length > 1 && <Legend wrapperStyle={{ color: 'var(--ink-2)', fontSize: 12 }} />}
          {series.map((s) => <Bar key={s.key} dataKey={s.key} name={s.label} fill={s.color} radius={[4, 4, 0, 0]} stroke="var(--surface)" strokeWidth={2} />)}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function ConfusionMatrix({ matrix, classes }: { matrix: number[][]; classes: string[] }) {
  const max = Math.max(1, ...matrix.flat())
  return (
    <table className="text-sm tabular-nums" aria-label="Confusion matrix: rows are true class, columns are predicted class">
      <thead>
        <tr><th className="p-1" /><th colSpan={classes.length} className="p-1 text-center font-medium" style={{ color: 'var(--ink-2)' }}>Predicted</th></tr>
        <tr><th className="p-1 text-left font-medium" style={{ color: 'var(--ink-2)' }}>True</th>{classes.map((c) => <th key={c} scope="col" className="p-1 text-center font-medium">{c}</th>)}</tr>
      </thead>
      <tbody>
        {matrix.map((row, i) => (
          <tr key={classes[i]}>
            <th scope="row" className="p-1 text-left font-medium">{classes[i]}</th>
            {row.map((v, j) => {
              const a = v / max
              return <td key={j} className="h-12 w-16 text-center font-semibold" style={{ background: `color-mix(in srgb, var(--series-1) ${Math.round(a * 85)}%, var(--surface))`, color: a > 0.5 ? '#fff' : 'var(--ink)', outline: i === j ? '2px solid var(--ink)' : 'none', outlineOffset: -2 }} title={`True ${classes[i]}, predicted ${classes[j]}: ${v}`}>{v}</td>
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

// -------- components/Results.tsx --------
function Results({ result, onRestart, onEdit }: { result: ScreeningResult; onRestart: () => void; onEdit: () => void }) {
  const [open, setOpen] = useState<string | null>(null)
  const ref = result.reference_scores
  return (
    <div>
      <h1 className="mb-4 text-2xl font-semibold sm:text-3xl">Your screening results</h1>
      <SupportPanel safety={result.safety} />
      <SyntheticBanner show={result.model.is_synthetic === true} />
      <Notice tone="warn" title="Screening estimate, not a diagnosis">{result.disclaimer}</Notice>

      {result.status === 'model_unavailable' ? (
        <Card className="mt-6">
          <h2 className="text-lg font-semibold">Model result unavailable</h2>
          <p className="mt-1" style={{ color: 'var(--ink-2)' }}>{result.model.reason} Your standard questionnaire scores are shown below; they do not need the model.</p>
        </Card>
      ) : (
        <>
          {result.summary && (
            <Card className="mt-6">
              <div className="flex flex-wrap items-center gap-3"><h2 className="text-lg font-semibold">Highest level across conditions</h2><LevelBadge level={result.summary.highest_level} /></div>
              <p className="mt-1 text-sm" style={{ color: 'var(--ink-2)' }}>{result.summary.note}</p>
            </Card>
          )}
          <div className="mt-6 grid gap-4 lg:grid-cols-3">
            {result.conditions.map((c) => (
              <Card key={c.target} as="article">
                <h2 className="text-lg font-semibold">{c.display_name}</h2>
                <div className="mt-2"><LevelBadge level={c.risk_level} /></div>
                <p className="mt-3 text-sm" style={{ color: 'var(--ink-2)' }}>{c.meaning}</p>
                <h3 className="mt-4 text-sm font-semibold">Answers that shaped this result</h3>
                <ul className="mt-1 list-disc pl-5 text-sm">
                  {c.key_factors.map((f) => <li key={f.feature}>{f.label}: {f.value}</li>)}
                </ul>
                <button className="mt-4 text-sm font-medium underline" aria-expanded={open === c.target} onClick={() => setOpen(open === c.target ? null : c.target)}>
                  {open === c.target ? 'Hide decision path' : 'Show decision path'}
                </button>
                {open === c.target && <div className="mt-3"><DecisionPathView path={c.decision_path} /></div>}
              </Card>
            ))}
          </div>
          {result.conditions.length < 3 && (
            <p className="mt-3 text-sm" style={{ color: 'var(--ink-3)' }}>Only conditions with labels in the training dataset are predicted; the others are not estimated.</p>
          )}
        </>
      )}

      <Card className="mt-6">
        <h2 className="text-lg font-semibold">Standard questionnaire scores</h2>
        <p className="mt-1 text-sm" style={{ color: 'var(--ink-2)' }}>{ref.note}</p>
        <dl className="mt-3 grid gap-4 sm:grid-cols-3">
          <div><dt className="text-xs uppercase" style={{ color: 'var(--ink-3)' }}>PHQ-9 (mood)</dt><dd className="text-lg font-semibold">{ref.phq9.total} / {ref.phq9.max} · {ref.phq9.band}</dd></div>
          <div><dt className="text-xs uppercase" style={{ color: 'var(--ink-3)' }}>GAD-7 (anxiety)</dt><dd className="text-lg font-semibold">{ref.gad7.total} / {ref.gad7.max} · {ref.gad7.band}</dd></div>
          <div><dt className="text-xs uppercase" style={{ color: 'var(--ink-3)' }}>Supplementary items</dt><dd className="text-lg font-semibold">{ref.supplementary.total} / {ref.supplementary.max}</dd><dd className="text-xs" style={{ color: 'var(--ink-3)' }}>{ref.supplementary.note}</dd></div>
        </dl>
      </Card>

      <Card className="mt-6">
        <h2 className="text-lg font-semibold">What you can do next</h2>
        <ul className="mt-2 list-disc space-y-1 pl-5">{result.recommendations.map((r) => <li key={r}>{r}</li>)}</ul>
      </Card>

      <Card className="mt-6">
        <h2 className="text-lg font-semibold">Limitations and uncertainty</h2>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm">{result.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
        <p className="mt-2 text-sm">Learn more on the <Link to="/explainability" className="underline">Explainable AI</Link> and <Link to="/performance" className="underline">Model performance</Link> pages.</p>
      </Card>

      <div className="mt-6 flex flex-wrap gap-3">
        <Button variant="secondary" onClick={onEdit}>Review my answers</Button>
        <Button onClick={onRestart}>Start a new screening</Button>
      </div>
    </div>
  )
}

// -------- layouts/AppLayout.tsx --------
const NAV = [
  { to: '/', label: 'Home', end: true },
  { to: '/screening', label: 'Screening' },
  { to: '/explainability', label: 'Explainable AI' },
  { to: '/performance', label: 'Model performance' },
  { to: '/about', label: 'About' },
  { to: '/admin', label: 'Admin' },
]

function AppLayout() {
  const [open, setOpen] = useState(false)
  const link = ({ isActive }: { isActive: boolean }) =>
    `rounded-md px-3 py-2 text-sm font-medium ${isActive ? 'underline underline-offset-8 decoration-2' : ''}`
  return (
    <div className="min-h-screen flex flex-col">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 rounded bg-white px-3 py-2 text-black">Skip to content</a>
      <header className="sticky top-0 z-40 border-b backdrop-blur" style={{ background: 'color-mix(in srgb, var(--surface) 92%, transparent)', borderColor: 'var(--border)' }}>
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
          <Link to="/" className="flex items-center gap-2 font-semibold" onClick={() => setOpen(false)}>
            <span className="grid h-8 w-8 place-items-center rounded-lg" style={{ background: 'var(--brand)', color: 'var(--brand-ink)' }}><GitBranch className="h-4 w-4" aria-hidden /></span>
            TreeScreen
          </Link>
          <nav aria-label="Main" className="hidden md:flex items-center gap-1">
            {NAV.map((n) => <NavLink key={n.to} to={n.to} end={n.end} className={link}>{n.label}</NavLink>)}
          </nav>
          <button className="md:hidden grid h-11 w-11 place-items-center rounded-md border" style={{ borderColor: 'var(--border)' }} aria-expanded={open} aria-controls="mobile-nav" aria-label={open ? 'Close menu' : 'Open menu'} onClick={() => setOpen(!open)}>
            {open ? <X aria-hidden /> : <Menu aria-hidden />}
          </button>
        </div>
        {open && (
          <nav id="mobile-nav" aria-label="Main mobile" className="md:hidden flex flex-col border-t px-4 py-2" style={{ borderColor: 'var(--border)' }}>
            {NAV.map((n) => <NavLink key={n.to} to={n.to} end={n.end} className={link} onClick={() => setOpen(false)}>{n.label}</NavLink>)}
          </nav>
        )}
      </header>
      <main id="main" className="mx-auto w-full max-w-6xl flex-1 px-4 py-8"><Outlet /></main>
      <footer className="border-t px-4 py-6 text-sm" style={{ borderColor: 'var(--border)', color: 'var(--ink-2)' }}>
        <div className="mx-auto max-w-6xl">
          <p><strong>Screening and decision support only — not a medical diagnosis.</strong> Results can be wrong in both directions. If you are struggling, please talk to a counsellor, doctor or someone you trust. Answers are processed in memory and are not stored.</p>
          <p className="mt-1">Not for admissions, employment, discipline or other consequential decisions about students.</p>
        </div>
      </footer>
    </div>
  )
}

// -------- pages/Home.tsx --------
const STEPS = [
  { icon: ListChecks, title: 'Answer 24 questions', text: 'PHQ-9 and GAD-7 items plus eight supplementary questions on sleep, study pressure and daily life. About 5 minutes.' },
  { icon: GitBranch, title: 'A decision tree reads them', text: 'A pruned CART tree per condition applies a short chain of if-then rules to your answers.' },
  { icon: Eye, title: 'See exactly why', text: 'You get Low, Moderate or High for depression, anxiety and stress, with the rules that produced each one.' },
]

function Home() {
  return (
    <div className="space-y-10">
      <section className="py-4 sm:py-10">
        <p className="text-sm font-medium" style={{ color: 'var(--brand)' }}>Major project · Phase II</p>
        <h1 className="mt-2 max-w-4xl text-3xl font-semibold sm:text-5xl">An interpretable decision tree for early screening of depression, anxiety and stress</h1>
        <p className="mt-4 max-w-2xl text-lg" style={{ color: 'var(--ink-2)' }}>
          Early signs are easy to miss. This tool gives students a quick, private screening whose reasoning can be read, questioned and checked, instead of a black-box score.
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <Link to="/screening" className="inline-flex min-h-[44px] items-center gap-2 rounded-lg px-5 py-2.5 font-medium" style={{ background: 'var(--brand)', color: 'var(--brand-ink)' }}>
            Begin screening <ArrowRight className="h-4 w-4" aria-hidden />
          </Link>
          <Link to="/explainability" className="inline-flex min-h-[44px] items-center rounded-lg border px-5 py-2.5 font-medium" style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}>See how the tree works</Link>
        </div>
      </section>

      <Notice tone="warn" title="This is not a diagnosis">
        Results are preliminary screening estimates from a decision-support tool. They can be wrong in both directions, and a Low result does not guarantee good mental health. If you are worried about how you feel, talk to a counsellor, a doctor or someone you trust.
      </Notice>

      <section aria-labelledby="how">
        <h2 id="how" className="mb-4 text-xl font-semibold">How the screening works</h2>
        <ol className="grid gap-4 md:grid-cols-3">
          {STEPS.map((s, i) => (
            <li key={s.title}>
              <Card as="div" className="h-full">
                <s.icon className="h-6 w-6" style={{ color: 'var(--brand)' }} aria-hidden />
                <h3 className="mt-3 font-semibold">{i + 1}. {s.title}</h3>
                <p className="mt-1 text-sm" style={{ color: 'var(--ink-2)' }}>{s.text}</p>
              </Card>
            </li>
          ))}
        </ol>
      </section>

      <section className="grid gap-4 md:grid-cols-3" aria-label="Key ideas">
        <Card as="div"><Layers className="h-5 w-5" style={{ color: 'var(--brand)' }} aria-hidden /><h3 className="mt-2 font-semibold">Risk stratification</h3><p className="mt-1 text-sm" style={{ color: 'var(--ink-2)' }}>Each condition is placed in Low, Moderate or High, the labels the model was trained on. A single tier is more actionable than a raw score, but it is coarser too.</p></Card>
        <Card as="div"><Eye className="h-5 w-5" style={{ color: 'var(--brand)' }} aria-hidden /><h3 className="mt-2 font-semibold">Interpretability</h3><p className="mt-1 text-sm" style={{ color: 'var(--ink-2)' }}>A shallow tree can be read in full. Every result lists the exact conditions on your answers that led to it. These show association in the training data, not causes.</p></Card>
        <Card as="div"><Lock className="h-5 w-5" style={{ color: 'var(--brand)' }} aria-hidden /><h3 className="mt-2 font-semibold">Private by design</h3><p className="mt-1 text-sm" style={{ color: 'var(--ink-2)' }}>No name, email or student ID is requested. Answers are processed in memory, are not stored or logged, and are never sent to third-party services.</p></Card>
      </section>
    </div>
  )
}

// -------- pages/Screening.tsx --------
type Answers = Record<string, string>
type Stage = 'form' | 'review' | 'result'

function buildSchema(q: Questionnaire) {
  const shape: Record<string, z.ZodType> = {}
  for (const s of q.sections) for (const i of s.items) {
    shape[i.id] = z.string({ error: 'Please choose an answer' }).regex(/^[0-3]$/, 'Please choose an answer')
  }
  return z.object(shape)
}

function Screening() {
  const { data: q, error, loading, reload } = useApi(() => api.questionnaire(), [])
  if (loading && !q) return <div className="space-y-4"><Skeleton h={80} /><Skeleton h={300} /></div>
  if (error && !q) return <ErrorState error={error} onRetry={reload} />
  if (!q) return null
  return <Flow q={q} />
}

function Flow({ q }: { q: Questionnaire }) {
  const schema = useMemo(() => buildSchema(q), [q])
  const form = useForm<Answers>({ resolver: zodResolver(schema) as never, defaultValues: {}, mode: 'onSubmit' })
  const { register, handleSubmit, watch, formState: { errors }, reset } = form
  const [stage, setStage] = useState<Stage>('form')
  const [step, setStep] = useState(0)
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<ApiError | null>(null)
  const [result, setResult] = useState<ScreeningResult | null>(null)

  const values = watch()
  const total = q.item_count
  const answered = q.sections.flatMap((s) => s.items).filter((i) => values[i.id] !== undefined && values[i.id] !== null && values[i.id] !== '').length
  const options = q.scales[0].options
  const section = q.sections[step]
  const unansweredIn = (idx: number) => q.sections[idx].items.filter((i) => !values[i.id] && values[i.id] !== '0').length

  async function submit(vals: Answers) {
    setSubmitting(true); setSubmitError(null)
    try {
      const numeric = Object.fromEntries(Object.entries(vals).map(([k, v]) => [k, Number(v)]))
      setResult(await api.predict(numeric)); setStage('result'); window.scrollTo(0, 0)
    } catch (e) {
      setSubmitError(e instanceof ApiError ? e : new ApiError(0, 'Unexpected error.'))
    } finally { setSubmitting(false) }
  }

  if (stage === 'result' && result) {
    return <Results result={result} onEdit={() => { setStage('review'); setResult(null) }} onRestart={() => { reset({}); setResult(null); setStep(0); setStage('form'); window.scrollTo(0, 0) }} />
  }

  const firstMissingSection = q.sections.findIndex((_, idx) => unansweredIn(idx) > 0)

  return (
    <div>
      <h1 className="text-2xl font-semibold sm:text-3xl">Screening questionnaire</h1>
      <p className="mt-2 max-w-3xl" style={{ color: 'var(--ink-2)' }}>Answer based on the last 2 weeks. There are no right or wrong answers. Nothing you enter is stored.</p>

      <div className="mt-4" role="group" aria-label="Progress">
        <div className="flex justify-between text-sm"><span>{answered} of {total} answered</span><span>{total - answered} left</span></div>
        <div className="mt-1 h-2 overflow-hidden rounded-full" style={{ background: 'var(--surface-2)' }} role="progressbar" aria-valuemin={0} aria-valuemax={total} aria-valuenow={answered} aria-label="Questions answered">
          <div className="h-full rounded-full" style={{ width: `${(answered / total) * 100}%`, background: 'var(--brand)' }} />
        </div>
      </div>

      {stage === 'form' && (
        <>
          <ol className="mt-4 flex flex-wrap gap-2" aria-label="Sections">
            {q.sections.map((s, idx) => (
              <li key={s.id}>
                <button type="button" onClick={() => setStep(idx)} aria-current={idx === step ? 'step' : undefined}
                  className="min-h-[44px] rounded-lg border px-3 py-2 text-sm font-medium" style={{ borderColor: idx === step ? 'var(--brand)' : 'var(--border)', background: idx === step ? 'var(--brand-soft)' : 'var(--surface)' }}>
                  {idx + 1}. {s.title}{unansweredIn(idx) === 0 ? ' ✓' : ''}
                </button>
              </li>
            ))}
          </ol>

          <form noValidate onSubmit={(e) => { e.preventDefault(); setStage('review'); window.scrollTo(0, 0) }} className="mt-4">
            <Card>
              <h2 className="text-xl font-semibold">{section.title}</h2>
              <p className="mt-1 font-medium">{section.stem}</p>
              <p className="text-sm" style={{ color: 'var(--ink-3)' }}>Time period: {section.time_period} · {section.standardized ? `Standard ${section.instrument} wording` : 'Proposed items, not a validated instrument'}</p>
              {!section.standardized && <div className="mt-3"><Notice tone="info">{section.provenance}</Notice></div>}
              <div className="mt-5 space-y-6">
                {section.items.map((item, n) => {
                  const err = errors[item.id]
                  return (
                    <fieldset key={item.id} aria-describedby={err ? `${item.id}-err` : undefined} className="rounded-lg">
                      <legend className="font-medium"><span style={{ color: 'var(--ink-3)' }}>{n + 1}.</span> {item.text}</legend>
                      <div className="mt-2 grid gap-2 sm:grid-cols-4">
                        {options.map((o) => (
                          <label key={o.value} className="flex min-h-[44px] cursor-pointer items-center gap-2 rounded-lg border px-3 py-2 text-sm" style={{ borderColor: values[item.id] === String(o.value) ? 'var(--brand)' : 'var(--border)', background: values[item.id] === String(o.value) ? 'var(--brand-soft)' : 'var(--surface)' }}>
                            <input type="radio" value={String(o.value)} {...register(item.id)} className="h-4 w-4" style={{ accentColor: 'var(--brand)' }} />
                            {o.label}
                          </label>
                        ))}
                      </div>
                      {err && <p id={`${item.id}-err`} role="alert" className="mt-1 text-sm font-medium" style={{ color: 'var(--danger-ink)' }}>Please choose an answer for this question.</p>}
                    </fieldset>
                  )
                })}
              </div>
            </Card>

            <div className="mt-4 flex flex-wrap justify-between gap-3">
              <Button variant="secondary" disabled={step === 0} onClick={() => { setStep(step - 1); window.scrollTo(0, 0) }}>Back</Button>
              {step < q.sections.length - 1
                ? <Button onClick={() => { setStep(step + 1); window.scrollTo(0, 0) }}>Next section</Button>
                : <button type="submit" className="inline-flex min-h-[44px] items-center rounded-lg px-4 py-2.5 font-medium" style={{ background: 'var(--brand)', color: 'var(--brand-ink)' }}>Review answers</button>}
            </div>
          </form>
        </>
      )}

      {stage === 'review' && (
        <div className="mt-6">
          <h2 className="text-xl font-semibold">Review your answers</h2>
          {answered < total && (
            <div className="mt-3"><Notice tone="warn" role="alert" title={`${total - answered} question${total - answered === 1 ? '' : 's'} still unanswered`}>
              All questions are needed for a result. <button className="underline" onClick={() => { setStep(Math.max(0, firstMissingSection)); setStage('form'); void handleSubmit(() => {})() }}>Go to the first unanswered section</button>.
            </Notice></div>
          )}
          <div className="mt-4 space-y-4">
            {q.sections.map((s, idx) => (
              <Card key={s.id}>
                <div className="flex items-center justify-between"><h3 className="font-semibold">{s.title}</h3><button className="text-sm underline" onClick={() => { setStep(idx); setStage('form') }}>Edit</button></div>
                <ul className="mt-2 divide-y text-sm" style={{ borderColor: 'var(--border)' }}>
                  {s.items.map((i) => {
                    const v = values[i.id]
                    return (
                      <li key={i.id} className="flex justify-between gap-4 py-1.5" style={{ borderColor: 'var(--border)' }}>
                        <span>{i.short}</span>
                        <span className="shrink-0 font-medium" style={{ color: v === undefined || v === '' ? 'var(--danger-ink)' : undefined }}>{v === undefined || v === '' ? 'Not answered' : options[Number(v)].label}</span>
                      </li>
                    )
                  })}
                </ul>
              </Card>
            ))}
          </div>
          {submitError && <div className="mt-4"><ErrorState error={submitError} /></div>}
          <div className="mt-4 flex flex-wrap gap-3">
            <Button variant="secondary" onClick={() => setStage('form')}>Back to questions</Button>
            <Button disabled={answered < total || submitting} onClick={handleSubmit(submit, () => setSubmitError(new ApiError(0, 'Please answer every question.')))}>
              {submitting ? 'Analysing…' : 'Get my results'}
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

// -------- pages/Explainability.tsx --------
function Explainability() {
  const [target, setTarget] = useState<string | undefined>(undefined)
  const { data, error, loading, reload } = useApi(() => api.tree(target), [target])

  if (loading && !data) return <div className="space-y-4"><Skeleton h={60} /><Skeleton h={320} /></div>
  if (error?.status === 503 && !data) {
    return (
      <>
        <PageHeader title="Explainable AI" />
        <EmptyState title="No trained tree yet" action={<Link to="/admin" className="underline">Open administration</Link>}>
          There is nothing to explain until an administrator trains a model on a dataset. No example tree is shown, because it would not be the real model.
        </EmptyState>
      </>
    )
  }
  if (error && !data) return <ErrorState error={error} onRetry={reload} />
  if (!data) return null

  const imp = data.feature_importance.filter((f) => f.impurity > 0 || f.permutation_mean > 0.001).slice(0, 10)
  return (
    <div>
      <PageHeader title="Explainable AI" lead="This page shows the fitted CART tree itself: its structure, the if-then rules read from its leaves, and which inputs it relies on. Nothing here is hand-written." />
      <SyntheticBanner show={data.is_synthetic} />

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <label htmlFor="target" className="font-medium">Condition</label>
        <select id="target" value={data.target} onChange={(e) => setTarget(e.target.value)} className="min-h-[44px] rounded-lg border px-3" style={{ borderColor: 'var(--border)', background: 'var(--surface)', color: 'var(--ink)' }}>
          {data.available_targets.map((t) => <option key={t.id} value={t.id}>{t.display_name}</option>)}
        </select>
        <span className="text-sm" style={{ color: 'var(--ink-2)' }}>{data.n_leaves} leaves · depth {data.depth}</span>
      </div>

      <Card className="mb-6">
        <h2 className="text-lg font-semibold">The tree</h2>
        <p className="mb-3 text-sm" style={{ color: 'var(--ink-2)' }}>Each box asks a yes/no question about one input. “n” is the number of training respondents reaching that point. Click to expand or collapse. Colour squares are paired with the class name.</p>
        <TreeView root={data.tree} />
      </Card>

      <Card className="mb-6">
        <h2 className="text-lg font-semibold">If-then rules ({data.rules.length})</h2>
        <p className="mb-3 text-sm" style={{ color: 'var(--ink-2)' }}>Every respondent matches exactly one rule. Counts are raw training respondents by dataset label, not probabilities.</p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead><tr className="text-left" style={{ color: 'var(--ink-2)' }}><th className="p-2">Rule</th><th className="p-2">Predicts</th><th className="p-2 text-right">n</th><th className="p-2">Training labels in this leaf</th></tr></thead>
            <tbody>
              {data.rules.map((r) => (
                <tr key={r.leaf_id} className="border-t align-top" style={{ borderColor: 'var(--border)' }}>
                  <td className="p-2">{r.text}</td>
                  <td className="p-2"><LevelBadge level={r.prediction} /></td>
                  <td className="p-2 text-right tabular-nums">{r.n_samples}</td>
                  <td className="p-2">{Object.entries(r.class_counts).map(([c, n]) => `${c} ${n} (${pct(r.class_share[c], 0)})`).join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card className="mb-6">
        <h2 className="text-lg font-semibold">Feature importance</h2>
        <p className="mb-2 text-sm" style={{ color: 'var(--ink-2)' }}>{data.notes}</p>
        <SimpleBars horizontal height={Math.max(220, imp.length * 38)} format={(v) => v.toFixed(2)}
          data={imp.map((f) => ({ name: f.label, impurity: Number(f.impurity.toFixed(4)), permutation: Number(Math.max(0, f.permutation_mean).toFixed(4)) }))}
          series={[{ key: 'impurity', label: 'Impurity importance (train)', color: 'var(--series-1)' }, { key: 'permutation', label: 'Permutation importance (test, Δ macro-F1)', color: 'var(--series-2)' }]} />
        <details className="mt-3 text-sm"><summary className="cursor-pointer font-medium">Show as table</summary>
          <table className="mt-2 w-full"><thead><tr className="text-left"><th>Feature</th><th>Impurity</th><th>Permutation (mean ± sd)</th></tr></thead>
            <tbody>{data.feature_importance.map((f) => <tr key={f.feature}><td>{f.label}</td><td>{f.impurity.toFixed(4)}</td><td>{f.permutation_mean.toFixed(4)} ± {f.permutation_std.toFixed(4)}</td></tr>)}</tbody></table>
        </details>
      </Card>

      <Card className="mb-6">
        <h2 className="text-lg font-semibold">What the inputs mean</h2>
        <dl className="mt-2 grid gap-x-6 gap-y-2 text-sm md:grid-cols-2">
          {imp.slice(0, 8).map((f) => { const info = data.feature_info[f.feature]; return info ? <div key={f.feature}><dt className="font-medium">{info.short}</dt><dd style={{ color: 'var(--ink-2)' }}>{info.description}</dd></div> : null })}
        </dl>
      </Card>

      <Notice tone="info" title="Association is not causation">
        A rule such as “sleep disturbance ≥ 2” describes a pattern in the training data. It does not mean that sleep causes the outcome, or that changing one answer would change a person’s mental health. Selected hyper-parameters: {Object.entries(data.selected_params).map(([k, v]) => `${k}=${String(v)}`).join(', ')}.
      </Notice>
    </div>
  )
}

// -------- pages/Performance.tsx --------
function Performance() {
  const { data, error, loading, reload } = useApi(() => api.metrics(), [])
  const [target, setTarget] = useState<string | null>(null)
  const [model, setModel] = useState('cart')

  if (loading && !data) return <div className="space-y-4"><Skeleton h={60} /><Skeleton h={300} /></div>
  if (error && !data) return <ErrorState error={error} onRetry={reload} />
  if (!data) return null
  if (!data.available) {
    return (
      <>
        <PageHeader title="Model performance" />
        <EmptyState title="Evaluation pending" action={<Link to="/admin" className="underline">Open administration</Link>}>
          {data.message} No metrics are shown because none have been computed, and placeholder numbers would be misleading.
        </EmptyState>
      </>
    )
  }
  const r = data.report
  const tKey = target && r.targets[target] ? target : Object.keys(r.targets)[0]
  const t = r.targets[tKey]
  const ok = MODEL_KEYS.filter((k) => t.models[k]?.status === 'ok')
  const notOk = MODEL_KEYS.filter((k) => t.models[k] && t.models[k].status !== 'ok')
  const sel = t.models[model]?.status === 'ok' ? model : 'cart'
  const m = t.models[sel].metrics!
  const compare = ['accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_macro'].map((k) => ({
    name: ({ accuracy: 'Accuracy', balanced_accuracy: 'Balanced acc.', precision_macro: 'Precision (macro)', recall_macro: 'Recall (macro)', f1_macro: 'F1 (macro)' } as Record<string, string>)[k],
    ...Object.fromEntries(ok.map((mk) => [mk, Number((t.models[mk].metrics as unknown as Record<string, number>)[k].toFixed(4))])),
  }))
  const dist = t.classes.map((c) => ({ name: c, train: t.class_distribution.train[c] ?? 0, test: t.class_distribution.test[c] ?? 0 }))
  const perClass = t.classes.map((c) => ({ name: c, precision: m.per_class[c].precision, recall: m.per_class[c].recall, f1: m.per_class[c].f1 }))

  return (
    <div>
      <PageHeader title="Model performance" lead="Held-out test results from the evaluation pipeline. Every model uses the same split, the same inputs and the same cross-validation protocol; tuning never touches the test split." />
      <SyntheticBanner show={r.dataset.is_synthetic} />

      <div className="mb-4 flex flex-wrap items-center gap-4">
        <div><label htmlFor="t" className="mr-2 font-medium">Condition</label>
          <select id="t" value={tKey} onChange={(e) => setTarget(e.target.value)} className="min-h-[44px] rounded-lg border px-3" style={{ borderColor: 'var(--border)', background: 'var(--surface)', color: 'var(--ink)' }}>
            {Object.entries(r.targets).map(([k, v]) => <option key={k} value={k}>{v.display_name}</option>)}</select></div>
        <div><label htmlFor="m" className="mr-2 font-medium">Model detail</label>
          <select id="m" value={sel} onChange={(e) => setModel(e.target.value)} className="min-h-[44px] rounded-lg border px-3" style={{ borderColor: 'var(--border)', background: 'var(--surface)', color: 'var(--ink)' }}>
            {ok.map((k) => <option key={k} value={k}>{t.models[k].display_name}</option>)}</select></div>
      </div>

      <Card className="mb-6">
        <h2 className="mb-3 text-lg font-semibold">{t.models[sel].display_name} — test set (n={m.n_test})</h2>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
          <Stat label="Accuracy" value={pct(m.accuracy)} />
          <Stat label="Balanced acc." value={pct(m.balanced_accuracy)} />
          <Stat label="Precision (macro)" value={pct(m.precision_macro)} />
          <Stat label="Recall (macro)" value={pct(m.recall_macro)} />
          <Stat label="F1 (macro)" value={pct(m.f1_macro)} />
          <Stat label="ROC-AUC (OvR)" value={m.roc_auc_ovr_macro == null ? 'n/a' : num3(m.roc_auc_ovr_macro)} />
        </div>
        <p className="mt-3 text-xs" style={{ color: 'var(--ink-3)' }}>Macro = unweighted mean over classes. ROC-AUC uses predicted probabilities, which are not calibrated for trees and ensembles.</p>
      </Card>

      <Card className="mb-6">
        <h2 className="text-lg font-semibold">Model comparison</h2>
        <p className="mb-2 text-sm" style={{ color: 'var(--ink-2)' }}>Differences between models can be within noise on a single split; see the cross-validation spread below. The accuracy-versus-interpretability trade-off is read from this chart, not assumed.</p>
        <SimpleBars horizontal data={compare} format={(v) => `${(v * 100).toFixed(1)}%`} domain={[0, 1]} height={520}
          series={ok.map((k) => ({ key: k, label: t.models[k].display_name, color: MODEL_COLOR[k] }))} />
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-sm tabular-nums">
            <thead><tr className="text-left" style={{ color: 'var(--ink-2)' }}><th className="p-2">Model</th><th className="p-2">Accuracy</th><th className="p-2">Recall (macro)</th><th className="p-2">F1 (macro)</th><th className="p-2">ROC-AUC</th><th className="p-2">CV F1 mean ± sd</th><th className="p-2">Selected parameters</th></tr></thead>
            <tbody>
              {MODEL_KEYS.filter((k) => t.models[k]).map((k) => { const e = t.models[k]; const x = e.metrics; return (
                <tr key={k} className="border-t align-top" style={{ borderColor: 'var(--border)' }}>
                  <td className="p-2 font-medium">{e.display_name}</td>
                  {e.status === 'ok' && x ? (<>
                    <td className="p-2">{pct(x.accuracy)}</td><td className="p-2">{pct(x.recall_macro)}</td><td className="p-2">{pct(x.f1_macro)}</td>
                    <td className="p-2">{x.roc_auc_ovr_macro == null ? 'n/a' : num3(x.roc_auc_ovr_macro)}</td>
                    <td className="p-2">{e.cv ? `${num3(e.cv.f1_macro.mean)} ± ${num3(e.cv.f1_macro.std)}` : '—'}</td>
                    <td className="p-2 text-xs">{Object.entries(e.selected_params ?? {}).map(([a, b]) => `${a}=${String(b)}`).join(', ')}</td></>)
                    : <td className="p-2" colSpan={6}>Not run ({e.status}): {e.reason}</td>}
                </tr>) })}
            </tbody>
          </table>
        </div>
        {notOk.length > 0 && <div className="mt-3"><Notice tone="warn">Some models were skipped or failed; reasons are listed in the table.</Notice></div>}
      </Card>

      <div className="mb-6 grid gap-6 lg:grid-cols-2">
        <Card>
          <h2 className="mb-2 text-lg font-semibold">Confusion matrix</h2>
          <ConfusionMatrix matrix={m.confusion_matrix} classes={t.classes} />
          <p className="mt-2 text-xs" style={{ color: 'var(--ink-3)' }}>Rows: true label. Columns: predicted label. The outlined diagonal is correct predictions.</p>
        </Card>
        <Card>
          <h2 className="mb-2 text-lg font-semibold">Per-class performance</h2>
          <SimpleBars data={perClass} format={(v) => `${(v * 100).toFixed(0)}%`} domain={[0, 1]} height={240}
            series={[{ key: 'precision', label: 'Precision', color: 'var(--series-1)' }, { key: 'recall', label: 'Recall', color: 'var(--series-2)' }, { key: 'f1', label: 'F1', color: 'var(--series-3)' }]} />
          <table className="mt-2 w-full text-sm tabular-nums"><thead><tr className="text-left"><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
            <tbody>{t.classes.map((c) => <tr key={c}><td>{c}</td><td>{pct(m.per_class[c].precision)}</td><td>{pct(m.per_class[c].recall)}</td><td>{pct(m.per_class[c].f1)}</td><td>{m.per_class[c].support}</td></tr>)}</tbody></table>
        </Card>
      </div>

      <div className="mb-6 grid gap-6 lg:grid-cols-2">
        <Card>
          <h2 className="mb-2 text-lg font-semibold">Class distribution</h2>
          <SimpleBars data={dist} height={220} series={[{ key: 'train', label: 'Train', color: 'var(--series-1)' }, { key: 'test', label: 'Test', color: 'var(--series-2)' }]} />
          <p className="mt-1 text-sm" style={{ color: 'var(--ink-2)' }}>All: {t.classes.map((c) => <span key={c} className="mr-3"><span aria-hidden style={{ color: LEVEL_COLOR[c] }}>■</span> {c} {t.class_distribution.all[c]}</span>)}</p>
        </Card>
        <Card>
          <h2 className="mb-2 text-lg font-semibold">Training information</h2>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
            <dt>Run</dt><dd>{r.run_id}</dd><dt>Trained</dt><dd>{fmtDate(r.created_at)}</dd>
            <dt>Dataset</dt><dd>{r.dataset.name} {r.dataset.is_synthetic && '(synthetic)'}</dd>
            <dt>Rows</dt><dd>{r.dataset.n_rows} (train {r.split.n_train} / test {r.split.n_test})</dd>
            <dt>Split</dt><dd>{r.split.strategy}</dd>
            <dt>Tuning</dt><dd>{r.tuning.method}, {t.cv_folds}-fold CV, selection by {r.tuning.selection_metric}</dd>
            <dt>Features</dt><dd>{r.feature_schema.columns.length} ({r.feature_schema.derived_totals_included ? 'with' : 'without'} derived totals)</dd>
            <dt>Software</dt><dd>{Object.entries(r.software_versions).map(([k, v]) => `${k} ${v ?? 'n/a'}`).join(' · ')}</dd>
          </dl>
          <h3 className="mt-3 text-sm font-semibold">Target definition</h3>
          <p className="text-sm" style={{ color: 'var(--ink-2)' }}>{t.target_definition}</p>
        </Card>
      </div>

      <Card>
        <h2 className="mb-2 text-lg font-semibold">Preprocessing and warnings</h2>
        <ul className="list-disc space-y-1 pl-5 text-sm">{r.preprocessing.map((p) => <li key={p}>{p}</li>)}</ul>
        {r.warnings.length > 0 && <div className="mt-3 space-y-2">{r.warnings.map((w) => <Notice key={w} tone="warn">{w}</Notice>)}</div>}
        {Object.keys(r.skipped_targets).length > 0 && <p className="mt-3 text-sm">Targets not trained: {Object.entries(r.skipped_targets).map(([k, v]) => `${k} (${v})`).join('; ')}</p>}
      </Card>
    </div>
  )
}

// -------- pages/About.tsx --------
const List = ({ items }: { items: string[] }) => <ul className="list-disc space-y-1 pl-5">{items.map((i) => <li key={i}>{i}</li>)}</ul>

function About() {
  const { data, error, loading, reload } = useApi(() => api.projectInfo(), [])
  if (loading && !data) return <Skeleton h={400} />
  if (error && !data) return <ErrorState error={error} onRetry={reload} />
  if (!data) return null
  return (
    <div className="space-y-6">
      <PageHeader title="About the project" lead={data.title} />
      <Notice tone="warn" title="Source status">{data.report_source_status}</Notice>
      <Card><h2 className="mb-2 text-lg font-semibold">Research problem</h2><p>{data.research_problem}</p></Card>
      <Card>
        <h2 className="mb-2 text-lg font-semibold">Objectives and where they are implemented</h2>
        <ul className="divide-y" style={{ borderColor: 'var(--border)' }}>
          {data.objectives.map((o) => (
            <li key={o.id} className="py-2 text-sm"><strong>{o.id}.</strong> {o.text}<div style={{ color: 'var(--ink-2)' }}>Implemented by: {o.implemented_by}</div></li>
          ))}
        </ul>
      </Card>
      <div className="grid gap-6 md:grid-cols-2">
        <Card><h2 className="mb-2 text-lg font-semibold">Scope</h2><List items={data.scope} /></Card>
        <Card><h2 className="mb-2 text-lg font-semibold">Research gaps</h2><List items={data.research_gaps} /></Card>
        <Card><h2 className="mb-2 text-lg font-semibold">Methodology</h2><List items={data.methodology} /></Card>
        <Card><h2 className="mb-2 text-lg font-semibold">Technology stack</h2>
          <dl className="space-y-1 text-sm">{Object.entries(data.technology_stack).map(([k, v]) => <div key={k}><dt className="inline font-medium capitalize">{k}: </dt><dd className="inline">{v.join(', ')}</dd></div>)}</dl></Card>
        <Card><h2 className="mb-2 text-lg font-semibold">Limitations</h2><List items={data.limitations} /></Card>
        <Card><h2 className="mb-2 text-lg font-semibold">Ethical considerations</h2><List items={data.ethics} /></Card>
      </div>
      <Card>
        <h2 className="mb-2 text-lg font-semibold">References</h2>
        <p className="text-sm" style={{ color: 'var(--ink-2)' }}>{data.references_status}</p>
        <h3 className="mt-3 text-sm font-semibold">Used for implementation</h3>
        <List items={data.implementation_references} />
      </Card>
    </div>
  )
}

// -------- pages/Admin.tsx --------
function ValidationView({ v }: { v: ValidationReport }) {
  return (
    <div className="mt-3 space-y-2 text-sm">
      <Notice tone={v.ok ? 'ok' : 'danger'} title={v.ok ? 'Dataset is valid' : 'Dataset has errors'}>
        {v.n_rows} rows · {v.n_columns} columns · trainable targets: {v.trainable_targets?.join(', ') || 'none'}
      </Notice>
      {v.errors.map((e) => <Notice key={e} tone="danger">{e}</Notice>)}
      {v.warnings.map((w) => <Notice key={w} tone="warn">{w}</Notice>)}
      {v.class_distribution && (
        <table className="tabular-nums"><thead><tr className="text-left"><th className="pr-4">Target</th><th className="pr-4">Low</th><th className="pr-4">Moderate</th><th>High</th></tr></thead>
          <tbody>{Object.entries(v.class_distribution).map(([t, d]) => <tr key={t}><td className="pr-4">{t}</td><td className="pr-4">{d.Low}</td><td className="pr-4">{d.Moderate}</td><td>{d.High}</td></tr>)}</tbody></table>
      )}
      {v.missing_values && Object.keys(v.missing_values).length > 0 && <p>Missing values: {Object.entries(v.missing_values).map(([k, n]) => `${k}: ${n}`).join(', ')}</p>}
    </div>
  )
}

function Admin() {
  const [token, setToken] = useState('')
  const [draft, setDraft] = useState('')
  const [authError, setAuthError] = useState<ApiError | null>(null)
  const [datasets, setDatasets] = useState<DatasetMeta[]>([])
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [msg, setMsg] = useState<{ tone: 'ok' | 'danger'; text: string } | null>(null)
  const [busy, setBusy] = useState(false)

  const [file, setFile] = useState<File | null>(null)
  const [name, setName] = useState('')
  const [prov, setProv] = useState('')
  const [synthFlag, setSynthFlag] = useState(false)
  const [validation, setValidation] = useState<ValidationReport | null>(null)

  const [dsId, setDsId] = useState('')
  const [metric, setMetric] = useState('f1_macro')
  const [derived, setDerived] = useState(true)
  const [activate, setActivate] = useState(false)
  const [job, setJob] = useState<Job | null>(null)
  const poll = useRef<number | null>(null)

  const fail = (e: unknown) => setMsg({ tone: 'danger', text: e instanceof ApiError ? e.message : 'Unexpected error.' })

  async function refresh(tk = token) {
    const [d, r] = await Promise.all([admin.datasets(tk), admin.evaluations(tk)])
    setDatasets(d); setRuns(r)
    setDsId((cur) => cur || d[0]?.id || '')
  }

  async function signIn(e: React.FormEvent) {
    e.preventDefault(); setAuthError(null)
    try { await refresh(draft); setToken(draft) } catch (err) { setAuthError(err instanceof ApiError ? err : new ApiError(0, 'Failed')) }
  }

  useEffect(() => () => { if (poll.current) window.clearInterval(poll.current) }, [])

  async function run<T>(fn: () => Promise<T>, ok: string) {
    setBusy(true); setMsg(null)
    try { await fn(); setMsg({ tone: 'ok', text: ok }); await refresh() } catch (e) { fail(e) } finally { setBusy(false) }
  }

  function startPolling(id: string) {
    if (poll.current) window.clearInterval(poll.current)
    poll.current = window.setInterval(async () => {
      try {
        const j = await admin.job(token, id); setJob(j)
        if (j.status === 'succeeded' || j.status === 'failed') {
          window.clearInterval(poll.current!); poll.current = null
          if (j.status === 'succeeded') { setMsg({ tone: 'ok', text: `Training finished: run ${j.result?.run_id}${j.result?.activated ? ' (activated)' : ' (not activated)'}` }); await refresh() }
        }
      } catch (e) { fail(e); window.clearInterval(poll.current!); poll.current = null }
    }, 1500)
  }

  if (!token) {
    return (
      <div className="mx-auto max-w-md">
        <PageHeader title="Administration" lead="Dataset upload, validation, training and evaluation history. Restricted to authorised users." />
        <Card>
          <form onSubmit={signIn}>
            <label htmlFor="key" className="font-medium">Admin API key</label>
            <input id="key" type="password" autoComplete="off" value={draft} onChange={(e) => setDraft(e.target.value)} className="mt-1 min-h-[44px] w-full rounded-lg border px-3" style={{ borderColor: 'var(--border)', background: 'var(--surface)', color: 'var(--ink)' }} />
            <p className="mt-1 text-xs" style={{ color: 'var(--ink-3)' }}>The key is set by the server operator via the ADMIN_API_KEY environment variable. It is kept in memory for this tab only.</p>
            {authError && <div className="mt-3"><ErrorState error={authError} /></div>}
            <Button className="mt-4 w-full" onClick={() => {}} type="submit" disabled={draft.length === 0}>Sign in</Button>
          </form>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <PageHeader title="Administration"><Button variant="secondary" className="mt-3" onClick={() => { setToken(''); setDraft(''); setDatasets([]); setRuns([]) }}>Sign out</Button></PageHeader>
      {msg && <Notice tone={msg.tone} role="status">{msg.text}</Notice>}

      <Card>
        <h2 className="text-lg font-semibold">1. Dataset</h2>
        <p className="text-sm" style={{ color: 'var(--ink-2)' }}>CSV with the 24 item columns (0–3), at least one of depression_risk / anxiety_risk / stress_risk / overall_risk (Low, Moderate, High) and optional participant_id. Columns outside the schema are discarded on upload.</p>
        <div className="mt-3 grid gap-3 md:grid-cols-2">
          <div><label htmlFor="f" className="text-sm font-medium">CSV file</label><input id="f" type="file" accept=".csv,text/csv" onChange={(e) => { setFile(e.target.files?.[0] ?? null); setValidation(null) }} className="mt-1 block w-full text-sm" /></div>
          <div><label htmlFor="n" className="text-sm font-medium">Dataset name</label><input id="n" value={name} onChange={(e) => setName(e.target.value)} className="mt-1 min-h-[44px] w-full rounded-lg border px-3" style={{ borderColor: 'var(--border)', background: 'var(--surface)', color: 'var(--ink)' }} /></div>
          <div className="md:col-span-2"><label htmlFor="p" className="text-sm font-medium">Provenance (source, collection date, how labels were defined)</label><input id="p" value={prov} onChange={(e) => setProv(e.target.value)} className="mt-1 min-h-[44px] w-full rounded-lg border px-3" style={{ borderColor: 'var(--border)', background: 'var(--surface)', color: 'var(--ink)' }} /></div>
          <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={synthFlag} onChange={(e) => setSynthFlag(e.target.checked)} /> This dataset is synthetic</label>
        </div>
        <div className="mt-3 flex flex-wrap gap-3">
          <Button variant="secondary" disabled={!file || busy} onClick={async () => { setBusy(true); setMsg(null); try { setValidation(await admin.validate(token, file!)) } catch (e) { fail(e) } finally { setBusy(false) } }}>Validate only</Button>
          <Button disabled={!file || !name.trim() || prov.trim().length < 3 || busy} onClick={() => run(async () => { const r = await admin.upload(token, file!, name, prov, synthFlag); setValidation(r.validation); setDsId(r.dataset.id) }, 'Dataset uploaded.')}>Validate and upload</Button>
          <Button variant="secondary" disabled={busy} onClick={() => run(async () => { const r = await admin.synthetic(token, 1500); setDsId(r.dataset.id); setValidation(r.validation) }, 'Synthetic demonstration dataset created.')}>Generate synthetic demo data</Button>
        </div>
        {validation && <ValidationView v={validation} />}
        <h3 className="mt-5 font-semibold">Stored datasets</h3>
        {datasets.length === 0 ? <p className="text-sm" style={{ color: 'var(--ink-2)' }}>None yet.</p> : (
          <ul className="mt-1 divide-y text-sm" style={{ borderColor: 'var(--border)' }}>{datasets.map((d) => <li key={d.id} className="py-1.5"><strong>{d.name}</strong> {d.is_synthetic && <span style={{ color: 'var(--warn-ink)' }}>[synthetic]</span>} — {d.n_rows} rows · targets: {d.trainable_targets.join(', ')} · {fmtDate(d.uploaded_at)}</li>)}</ul>
        )}
      </Card>

      <Card>
        <h2 className="text-lg font-semibold">2. Train and evaluate</h2>
        <div className="mt-3 grid gap-3 md:grid-cols-3">
          <div><label htmlFor="ds" className="text-sm font-medium">Dataset</label>
            <select id="ds" value={dsId} onChange={(e) => setDsId(e.target.value)} className="mt-1 min-h-[44px] w-full rounded-lg border px-3" style={{ borderColor: 'var(--border)', background: 'var(--surface)', color: 'var(--ink)' }}>
              <option value="">Select…</option>{datasets.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}</select></div>
          <div><label htmlFor="sm" className="text-sm font-medium">Selection metric</label>
            <select id="sm" value={metric} onChange={(e) => setMetric(e.target.value)} className="mt-1 min-h-[44px] w-full rounded-lg border px-3" style={{ borderColor: 'var(--border)', background: 'var(--surface)', color: 'var(--ink)' }}>
              <option value="f1_macro">Macro F1</option><option value="recall_macro">Macro recall</option><option value="balanced_accuracy">Balanced accuracy</option></select></div>
          <div className="space-y-1 pt-6 text-sm">
            <label className="flex items-center gap-2"><input type="checkbox" checked={derived} onChange={(e) => setDerived(e.target.checked)} /> Include derived score totals</label>
            <label className="flex items-center gap-2"><input type="checkbox" checked={activate} onChange={(e) => setActivate(e.target.checked)} /> Activate this run when finished</label>
          </div>
        </div>
        <p className="mt-2 text-xs" style={{ color: 'var(--ink-3)' }}>Turn derived totals off if the dataset’s labels were defined from PHQ-9/GAD-7 totals (label leakage). The first run is always activated; later runs never overwrite earlier ones.</p>
        <Button className="mt-3" disabled={!dsId || busy || job?.status === 'running' || job?.status === 'queued'} onClick={async () => {
          setMsg(null)
          if (!window.confirm('Start training? This can take a few minutes and creates a new saved run.')) return
          try { const j = await admin.train(token, { dataset_id: dsId, selection_metric: metric, include_derived_totals: derived, activate: activate || null }); setJob(j); startPolling(j.id) } catch (e) { fail(e) }
        }}>Train all models</Button>
        {job && (
          <div className="mt-3" role="status">
            {(job.status === 'running' || job.status === 'queued') && <Spinner label={`${job.message} (${Math.round(job.progress * 100)}%)`} />}
            {job.status === 'failed' && <Notice tone="danger" title="Training failed">{job.error}</Notice>}
          </div>
        )}
      </Card>

      <Card>
        <h2 className="text-lg font-semibold">3. Saved evaluation runs</h2>
        {runs.length === 0 ? <p className="text-sm" style={{ color: 'var(--ink-2)' }}>No runs yet.</p> : (
          <div className="overflow-x-auto"><table className="mt-2 w-full text-sm tabular-nums">
            <thead><tr className="text-left" style={{ color: 'var(--ink-2)' }}><th className="p-2">Run</th><th className="p-2">Dataset</th><th className="p-2">CART macro-F1 by condition</th><th className="p-2">Status</th><th className="p-2">Actions</th></tr></thead>
            <tbody>{runs.map((r) => (
              <tr key={r.run_id} className="border-t align-top" style={{ borderColor: 'var(--border)' }}>
                <td className="p-2">{r.run_id}<div className="text-xs" style={{ color: 'var(--ink-3)' }}>{fmtDate(r.created_at)}</div></td>
                <td className="p-2">{r.dataset.name} {r.dataset.is_synthetic && '[synthetic]'}</td>
                <td className="p-2">{Object.entries(r.cart_metrics).map(([t, m]) => `${t.replace('_risk', '')} ${pct(m.f1_macro)}`).join(' · ')}</td>
                <td className="p-2">{r.active ? <strong>Active</strong> : 'Inactive'}</td>
                <td className="p-2"><div className="flex flex-wrap gap-2">
                  {!r.active && <Button variant="secondary" className="!min-h-[36px] !px-3 !py-1" disabled={busy} onClick={() => { if (window.confirm('Make this run serve predictions for all users?')) void run(() => admin.activate(token, r.run_id), 'Run activated.') }}>Activate</Button>}
                  <Button variant="secondary" className="!min-h-[36px] !px-3 !py-1" onClick={async () => { try { downloadBlob(await admin.exportRun(token, r.run_id, 'json'), `evaluation-${r.run_id}.json`) } catch (e) { fail(e) } }}>JSON</Button>
                  <Button variant="secondary" className="!min-h-[36px] !px-3 !py-1" onClick={async () => { try { downloadBlob(await admin.exportRun(token, r.run_id, 'csv'), `evaluation-${r.run_id}.csv`) } catch (e) { fail(e) } }}>CSV</Button>
                </div></td>
              </tr>))}</tbody></table></div>
        )}
      </Card>
    </div>
  )
}

// -------- App.tsx --------
function NotFound() {
  return (
    <div className="py-16 text-center">
      <h1 className="text-2xl font-semibold">Page not found</h1>
      <p className="mt-2" style={{ color: 'var(--ink-2)' }}>That page does not exist.</p>
      <Link to="/" className="mt-4 inline-block underline">Back to home</Link>
    </div>
  )
}

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<Home />} />
        <Route path="screening" element={<Screening />} />
        <Route path="explainability" element={<Explainability />} />
        <Route path="performance" element={<Performance />} />
        <Route path="about" element={<About />} />
        <Route path="admin" element={<Admin />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  )
}

// Mount (skipped when there is no #root element, e.g. under test)
const rootEl = typeof document !== 'undefined' ? document.getElementById('root') : null
if (rootEl) {
  createRoot(rootEl).render(
    <StrictMode>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </StrictMode>,
  )
}
