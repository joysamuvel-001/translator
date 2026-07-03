import { Activity, FileAudio } from 'lucide-react'
import TranscriptCard from './TranscriptCard'

export default function TranscriptFeed({ turns, sessionTitle, turnCount }) {
  return (
    <main className="flex-1 h-full overflow-y-auto">
      <div className="sticky top-0 z-10 bg-ink/85 backdrop-blur border-b border-ink-border px-10 py-5 flex items-center justify-between">
        <div>
          <h2 className="font-display text-[1.15rem] font-semibold text-text-primary">
            {sessionTitle}
          </h2>
          <p className="text-[0.76rem] text-text-faint mt-0.5">
            {turnCount} turn{turnCount === 1 ? '' : 's'} transcribed
          </p>
        </div>
        <div className="flex items-center gap-1.5 text-[0.74rem] font-mono text-vital-teal/80">
          <Activity className="w-3.5 h-3.5" />
          IndicConformer · IndicTrans2 · TitaNet
        </div>
      </div>

      <div className="px-10 py-7 max-w-[860px] mx-auto space-y-4">
        {turns.length === 0 ? (
          <EmptyState />
        ) : (
          turns.map((t) => <TranscriptCard key={t.id} turn={t} />)
        )}
      </div>
    </main>
  )
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center text-center py-28 border border-dashed border-ink-border rounded-xl2">
      <div className="w-12 h-12 rounded-full bg-ink-raised border border-ink-border flex items-center justify-center mb-4">
        <FileAudio className="w-5 h-5 text-text-faint" />
      </div>
      <p className="text-[0.92rem] text-text-muted">Nothing transcribed yet</p>
      <p className="text-[0.8rem] text-text-faint mt-1 max-w-[320px]">
        Enroll your speakers on the left, then tap the mic to start capturing
        the consult.
      </p>
    </div>
  )
}
