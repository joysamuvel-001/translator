const BASE = '/api'

async function asJson(res) {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed (${res.status})`)
  }
  return res.json()
}

export const api = {
  health: () => fetch(`${BASE}/health`).then(asJson),

  listSpeakers: () => fetch(`${BASE}/speakers`).then(asJson),

  enrollSpeaker: (name, blob) => {
    const form = new FormData()
    form.append('name', name)
    form.append('sample', blob, 'sample.webm')
    return fetch(`${BASE}/speakers/enroll`, { method: 'POST', body: form }).then(asJson)
  },

  createSession: (title = 'Untitled session') =>
    fetch(`${BASE}/sessions?title=${encodeURIComponent(title)}`, { method: 'POST' }).then(asJson),

  listSessions: () => fetch(`${BASE}/sessions`).then(asJson),

  getSession: (id) => fetch(`${BASE}/sessions/${id}`).then(asJson),

  transcribeTurn: (sessionId, blob, languageHint) => {
    const form = new FormData()
    form.append('session_id', sessionId)
    form.append('audio', blob, 'turn.webm')
    if (languageHint) form.append('language_hint', languageHint)
    return fetch(`${BASE}/transcribe`, { method: 'POST', body: form }).then(asJson)
  },
}
