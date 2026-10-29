import '@testing-library/jest-dom/vitest'
import { afterEach, describe, it, expect, vi, beforeEach } from 'vitest'
import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import App from './main'
import type { Questionnaire, ScreeningResult } from './main'

afterEach(() => cleanup())
// jsdom lacks these browser APIs used by Recharts / scrolling.
class RO { observe() {} unobserve() {} disconnect() {} }
;(globalThis as unknown as { ResizeObserver: typeof RO }).ResizeObserver = RO
window.scrollTo = () => {}
Element.prototype.scrollIntoView = () => {}

const opts = [0, 1, 2, 3].map((v) => ({ value: v, label: ['Not at all', 'Several days', 'More than half the days', 'Nearly every day'][v] }))
const mk = (id: string, text: string, crit = false) => ({ id, text, short: text, scale: 's', domain: 'd', safety_critical: crit })
const Q: Questionnaire = {
  title: 'q', item_count: 3, validation_notice: '', scales: [{ id: 's', options: opts }],
  sections: [{ id: 'a', title: 'Section A', standardized: true, instrument: 'PHQ-9', stem: 'Stem', time_period: 'Last 2 weeks', provenance: '', items: [mk('i1', 'Item one'), mk('i2', 'Item two'), mk('i3', 'Item three', true)] }],
}
const RESULT: ScreeningResult = {
  status: 'ok', disclaimer: 'Not a medical diagnosis disclaimer.', limitations: ['Limit one'], recommendations: ['Rec one'],
  model: { available: true, run_id: 'r', is_synthetic: true, reason: null },
  safety: { triggered: false, level: 'none', trigger: null, message: null, resources: null, note: '' },
  reference_scores: { note: 'n', phq9: { total: 3, max: 27, band: 'Minimal' }, gad7: { total: 0, max: 21, band: 'Minimal' }, supplementary: { total: 0, max: 24, note: 'x' }, stress: { available: false, note: '' } },
  summary: { highest_level: 'Moderate', derived: true, note: 'Derived note' }, features_used: {},
  conditions: [{ target: 'depression_risk', display_name: 'Depression risk', risk_level: 'Moderate', meaning: 'Means moderate', classes: ['Low', 'Moderate', 'High'], key_factors: [{ feature: 'i1', label: 'Item one', description: '', value: 2 }],
    decision_path: { node_ids: [0, 1], leaf_id: 1, prediction: 'Moderate', steps: [{ node_id: 0, feature: 'i1', feature_label: 'Item one', value: 2, threshold: 1.5, direction: 'right', satisfied_condition: 'Item one ≥ 2', text: '' }], leaf_evidence: { n_train_samples: 10, class_counts: { Low: 1, Moderate: 8, High: 1 }, class_share: { Low: 0.1, Moderate: 0.8, High: 0.1 }, description: 'desc' } } }],
}

function json(body: unknown, status = 200) { return Promise.resolve(new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })) }
let routes: Record<string, () => Promise<Response>>
beforeEach(() => {
  routes = { '/api/questionnaire': () => json(Q) }
  vi.stubGlobal('fetch', vi.fn((url: string) => { const k = Object.keys(routes).find((r) => String(url).startsWith(r)); return k ? routes[k]() : json({ detail: 'nf' }, 404) }))
})
const go = (path: string) => render(<MemoryRouter initialEntries={[path]}><App /></MemoryRouter>)

async function answerAll(user: ReturnType<typeof userEvent.setup>) {
  for (const item of ['Item one', 'Item two', 'Item three']) {
    const fs = (await screen.findByText(new RegExp(item))).closest('fieldset') as HTMLElement
    await user.click(within(fs).getByLabelText('Several days'))
  }
}

describe('navigation and static pages', () => {
  it('renders home with disclaimer and CTA', () => {
    go('/')
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(/interpretable decision tree/i)
    expect(screen.getAllByText(/not a (medical )?diagnosis/i).length).toBeGreaterThan(0)
    expect(screen.getByRole('link', { name: /begin screening/i })).toHaveAttribute('href', '/screening')
  })
  it('navigates via the header', async () => {
    go('/'); const user = userEvent.setup()
    routes['/api/project-info'] = () => json({ title: 'T', report_source_status: 'src', research_problem: 'rp', objectives: [], scope: [], methodology: [], research_gaps: [], technology_stack: {}, limitations: [], ethics: [], references: [], references_status: 'Pending refs', implementation_references: [] })
    await user.click(screen.getAllByRole('link', { name: 'About' })[0])
    expect(await screen.findByRole('heading', { name: /about the project/i })).toBeInTheDocument()
  })
  it('shows 404 page', () => { go('/nope'); expect(screen.getByText(/page not found/i)).toBeInTheDocument() })
})

describe('questionnaire', () => {
  it('requires every answer, shows progress, and blocks submit', async () => {
    go('/screening'); const user = userEvent.setup()
    expect(await screen.findByText('0 of 3 answered')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /review answers/i }))
    expect(await screen.findByText(/3 questions still unanswered/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /get my results/i })).toBeDisabled()
  })
  it('submits integer answers and renders returned results + decision path', async () => {
    const calls: unknown[] = []
    routes['/api/screening/predict'] = () => { return json(RESULT) }
    ;(fetch as unknown as ReturnType<typeof vi.fn>).mockImplementation((url: string, init?: RequestInit) => {
      if (String(url).endsWith('/screening/predict')) calls.push(JSON.parse(String(init?.body)))
      const k = Object.keys(routes).find((r) => String(url).startsWith(r)); return k ? routes[k]() : json({}, 404)
    })
    go('/screening'); const user = userEvent.setup()
    await answerAll(user)
    expect(screen.getByText('3 of 3 answered')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /review answers/i }))
    await user.click(await screen.findByRole('button', { name: /get my results/i }))
    expect(await screen.findByRole('heading', { name: /your screening results/i })).toBeInTheDocument()
    expect(calls[0]).toEqual({ answers: { i1: 1, i2: 1, i3: 1 } })
    expect(screen.getByText('Depression risk')).toBeInTheDocument()
    expect(screen.getByText(/synthetic demonstration data/i)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /show decision path/i }))
    expect(screen.getByText('Item one ≥ 2')).toBeInTheDocument()
    expect(screen.queryByRole('alert', { name: /support/i })).not.toBeInTheDocument()
  })
  it('shows the support panel when the safety check triggers', async () => {
    const r = { ...RESULT, safety: { triggered: true, level: 'urgent' as const, trigger: 'Your answer', message: 'Please reach out today.', note: '', resources: { region: 'IN', verification_notice: 'verify me', emergency: [{ name: 'Emergency', contact: '112' }], helplines: [{ name: 'Helpline', contact: '14416' }], directory: [], human_support: ['A counsellor'] } } }
    routes['/api/screening/predict'] = () => json(r)
    go('/screening'); const user = userEvent.setup()
    await answerAll(user); await user.click(screen.getByRole('button', { name: /review answers/i }))
    await user.click(await screen.findByRole('button', { name: /get my results/i }))
    expect(await screen.findByText(/please reach out for support today/i)).toBeInTheDocument()
    expect(screen.getByText('112')).toBeInTheDocument(); expect(screen.getByText(/verify me/)).toBeInTheDocument()
  })
  it('renders the model-unavailable state without fabricated output', async () => {
    routes['/api/screening/predict'] = () => json({ ...RESULT, status: 'model_unavailable', conditions: [], summary: null, model: { available: false, run_id: null, is_synthetic: null, reason: 'No trained model is active.' } })
    go('/screening'); const user = userEvent.setup()
    await answerAll(user); await user.click(screen.getByRole('button', { name: /review answers/i }))
    await user.click(await screen.findByRole('button', { name: /get my results/i }))
    expect(await screen.findByText(/model result unavailable/i)).toBeInTheDocument()
    expect(screen.queryByText(/highest level/i)).not.toBeInTheDocument()
    expect(screen.getByText(/PHQ-9 \(mood\)/)).toBeInTheDocument()
  })
  it('shows an error when the API fails and allows retry', async () => {
    routes['/api/screening/predict'] = () => json({ detail: 'Too many requests.' }, 429)
    go('/screening'); const user = userEvent.setup()
    await answerAll(user); await user.click(screen.getByRole('button', { name: /review answers/i }))
    await user.click(await screen.findByRole('button', { name: /get my results/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/too many requests/i)
    expect(screen.getByRole('button', { name: /get my results/i })).toBeEnabled()
  })
  it('shows an error state if the questionnaire cannot load', async () => {
    routes['/api/questionnaire'] = () => Promise.reject(new TypeError('net'))
    go('/screening')
    expect(await screen.findByRole('alert')).toHaveTextContent(/cannot reach the server/i)
  })
})

describe('empty states (no trained model)', () => {
  it('performance page shows pending state, no metrics', async () => {
    routes['/api/model/metrics'] = () => json({ available: false, message: 'Evaluation has not been run.' })
    go('/performance')
    expect(await screen.findByText(/evaluation pending/i)).toBeInTheDocument()
    expect(screen.queryByText(/accuracy/i)).not.toBeInTheDocument()
  })
  it('explainability page shows no tree when the model is missing', async () => {
    routes['/api/model/tree'] = () => json({ detail: 'No trained model' }, 503)
    go('/explainability')
    expect(await screen.findByText(/no trained tree yet/i)).toBeInTheDocument()
  })
})

describe('admin', () => {
  it('rejects a wrong key and shows the dashboard for a valid one', async () => {
    routes['/api/admin/datasets'] = () => json({ detail: 'Invalid or missing admin credentials.' }, 401)
    go('/admin'); const user = userEvent.setup()
    await user.type(screen.getByLabelText(/admin api key/i), 'wrong'); await user.click(screen.getByRole('button', { name: /sign in/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/invalid or missing/i)
    routes['/api/admin/datasets'] = () => json([]); routes['/api/admin/evaluations'] = () => json([])
    await user.clear(screen.getByLabelText(/admin api key/i)); await user.type(screen.getByLabelText(/admin api key/i), 'right'); await user.click(screen.getByRole('button', { name: /sign in/i }))
    await waitFor(() => expect(screen.getByRole('heading', { name: /1\. dataset/i })).toBeInTheDocument())
    expect(screen.getByRole('button', { name: /train all models/i })).toBeDisabled()
  })
})
