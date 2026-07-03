import { Activity } from 'lucide-react'

export default function Sidebar({ children }) {
  return (
    <aside className="w-[340px] shrink-0 h-full bg-ink-surface border-r border-ink-border flex flex-col">
      <div className="px-6 pt-7 pb-6 border-b border-ink-border">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-vital-teal/15 border border-vital-teal/30 flex items-center justify-center">
            <Activity className="w-4 h-4 text-vital-teal" strokeWidth={2.25} />
          </div>
          <h1 className="font-display text-[1.35rem] font-semibold tracking-tight text-text-primary">
            MedTranscribe
          </h1>
        </div>
        <p className="mt-1.5 text-[0.8rem] text-text-faint pl-[42px]">
          Multilingual consult transcription
        </p>
      </div>
      <div className="flex-1 overflow-y-auto">{children}</div>
    </aside>
  )
}
