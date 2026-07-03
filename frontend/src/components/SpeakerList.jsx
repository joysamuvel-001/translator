export default function SpeakerList({ speakers }) {
  if (!speakers.length) {
    return (
      <section className="px-6 py-5 border-b border-ink-border">
        <p className="text-[0.7rem] font-mono uppercase tracking-[0.14em] text-text-faint mb-2">
          Enrolled speakers
        </p>
        <p className="text-[0.78rem] text-text-faint">No one enrolled yet.</p>
      </section>
    )
  }

  return (
    <section className="px-6 py-5 border-b border-ink-border">
      <p className="text-[0.7rem] font-mono uppercase tracking-[0.14em] text-text-faint mb-3">
        Enrolled speakers · {speakers.length}
      </p>
      <ul className="space-y-2">
        {speakers.map((s) => (
          <li key={s.id} className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-full bg-vital-indigo/15 border border-vital-indigo/30 flex items-center justify-center text-[0.72rem] font-medium text-vital-indigo">
              {s.name.trim().charAt(0).toUpperCase()}
            </div>
            <span className="text-[0.84rem] text-text-primary">{s.name}</span>
            <span className="ml-auto text-[0.68rem] font-mono text-text-faint">
              {s.sample_duration_sec}s
            </span>
          </li>
        ))}
      </ul>
    </section>
  )
}
