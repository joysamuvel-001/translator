import { useEffect, useRef, useState } from 'react'
import Sidebar from './components/Sidebar'
import EnrollmentPanel from './components/EnrollmentPanel'
import SpeakerList from './components/SpeakerList'
import RecordingPanel from './components/RecordingPanel'
import TranscriptFeed from './components/TranscriptFeed'
import { useRecorder } from './hooks/useRecorder'
import { api } from './lib/api'

export default function App() {
  const [speakers, setSpeakers] = useState([])
  const [session, setSession] = useState(null)
  const [turns, setTurns] = useState([])
  const [busy, setBusy] = useState(false)
  const [language, setLanguage] = useState('hi')
  const lastBlob = useRef(null)
  const { isRecording, elapsed, start, stop } = useRecorder()

  useEffect(() => {
    api.listSpeakers().then(setSpeakers).catch(() => {})
    api.createSession('Consult — ' + new Date().toLocaleDateString()).then(setSession)
  }, [])

  const runTurn = async (blob) => {
    if (!session) return
    setBusy(true)
    lastBlob.current = blob
    try {
      const { turns: newTurns } = await api.transcribeTurn(session.id, blob, language)
      setTurns((prev) => [...prev, ...(newTurns || [])])
    } catch (err) {
      console.error(err)
    } finally {
      setBusy(false)
    }
  }

  const handleToggleRecording = async () => {
    if (!isRecording) {
      await start()
      return
    }
    const blob = await stop()
    if (blob) await runTurn(blob)
  }

  const handleRetryLast = () => {
    if (lastBlob.current) runTurn(lastBlob.current)
  }

  const handleNewSession = async () => {
    const fresh = await api.createSession('Consult — ' + new Date().toLocaleTimeString())
    setSession(fresh)
    setTurns([])
    lastBlob.current = null
  }

  return (
    <div className="h-screen w-screen flex bg-ink overflow-hidden">
      <Sidebar>
        <EnrollmentPanel onEnrolled={(s) => setSpeakers((prev) => [...prev, s])} />
        <SpeakerList speakers={speakers} />
        <RecordingPanel
          isRecording={isRecording}
          elapsed={elapsed}
          busy={busy}
          onToggle={handleToggleRecording}
          onRetryLast={handleRetryLast}
          onNewSession={handleNewSession}
          canRetry={!!lastBlob.current}
          language={language}
          onLanguageChange={setLanguage}
        />
        <SessionMeta turns={turns} speakers={speakers} />
      </Sidebar>

      <TranscriptFeed
        turns={turns}
        sessionTitle={session?.title ?? 'Loading session…'}
        turnCount={turns.length}
        busy={busy}
      />
    </div>
  )
}

function SessionMeta({ turns }) {
  const counts = turns.reduce((acc, t) => {
    acc[t.speaker_name] = (acc[t.speaker_name] || 0) + 1
    return acc
  }, {})

  return (
    <section className="px-6 py-5">
      <p className="text-[0.7rem] font-mono uppercase tracking-[0.14em] text-text-faint mb-3">
        This session
      </p>
      <p className="text-[0.84rem] text-text-primary mb-2">
        <span className="font-mono text-vital-teal">{turns.length}</span> turns total
      </p>
      <ul className="space-y-1.5">
        {Object.entries(counts).map(([name, count]) => (
          <li key={name} className="flex items-center justify-between text-[0.8rem]">
            <span className="text-text-muted">{name}</span>
            <span className="font-mono text-text-faint">{count} turns</span>
          </li>
        ))}
      </ul>
    </section>
  )
}