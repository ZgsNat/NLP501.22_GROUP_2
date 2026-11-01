import { useState } from 'react'
import type { Config, Mode, RunRequest } from './types'

interface Props { config: Config | null; running: boolean; onStart: (req: RunRequest) => void }

export default function RunPanel({ config, running, onStart }: Props) {
  const [jd, setJd] = useState('')
  const [profile, setProfile] = useState('default')
  const [reasoning, setReasoning] = useState('')
  const [fast, setFast] = useState('')
  const [mode, setMode] = useState<Mode>('agent')
  const [goal, setGoal] = useState('')

  if (!config) return <section className="card"><h2>New run</h2><p className="muted">Loading config…</p></section>

  const pickSample = (id: string) => {
    const s = config.sample_jds.find((j) => j.id === id)
    if (s) { setJd(s.text); setProfile(s.company_profile) }
  }
  const submit = () => {
    const models = { ...(reasoning.trim() && { reasoning: reasoning.trim() }), ...(fast.trim() && { fast: fast.trim() }) }
    onStart({ jd_text: jd, company_profile: profile, mode, ...(mode === 'agent' && { goal: goal.trim() }),
      ...(Object.keys(models).length && { models }) })
  }

  return (
    <section className="card">
      <h2>New run</h2>
      <label>Sample JD
        <select defaultValue="" onChange={(e) => pickSample(e.target.value)}>
          <option value="" disabled>Choose a sample…</option>
          {config.sample_jds.map((j) => <option key={j.id} value={j.id}>{j.id} · {j.title}</option>)}
        </select>
      </label>
      <label>Job description
        <textarea rows={8} value={jd} onChange={(e) => setJd(e.target.value)} placeholder="Paste or pick a JD" />
      </label>
      <label>Mode
        <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
          <option value="agent">Agent (P7): the model plans its own steps</option>
          <option value="pipeline">Pipeline (P6): fixed graph, rule routers</option>
        </select>
      </label>
      {mode === 'agent' && (
        <label>Your goal (to the agent)
          <textarea rows={2} value={goal} onChange={(e) => setGoal(e.target.value)}
                    placeholder="e.g. Shortlist 5 senior Playwright testers; ISTQB is a plus" />
        </label>
      )}
      <label>Company profile
        <select value={profile} onChange={(e) => setProfile(e.target.value)}>
          {config.company_profiles.map((p) => <option key={p.key} value={p.key}>{p.label}</option>)}
        </select>
      </label>
      <label>Reasoning model
        <input list="providers" value={reasoning} onChange={(e) => setReasoning(e.target.value)} placeholder={config.defaults.reasoning} />
      </label>
      <label>Fast model
        <input list="providers" value={fast} onChange={(e) => setFast(e.target.value)} placeholder={config.defaults.fast} />
      </label>
      <datalist id="providers">{config.providers.map((p) => <option key={p.name} value={`${p.name}:`} />)}</datalist>
      <div className="chips">
        {config.providers.map((p) => (
          <span key={p.name} className={`badge ${p.configured ? 'green' : 'gray'}`}>
            {p.name} · {p.configured ? 'configured' : 'not configured'}
          </span>
        ))}
      </div>
      <p className="muted small">Format <code>provider:model</code>; leave blank for defaults.</p>
      <button className="primary" disabled={running || !jd.trim()} onClick={submit}>
        {running ? 'Running…' : 'Start'}
      </button>
    </section>
  )
}
