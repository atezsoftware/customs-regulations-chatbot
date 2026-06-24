export function EmptyState({hasDirectories}: {hasDirectories: boolean}) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center rounded-xl border border-dashed border-slate-200 bg-white/70 px-6 py-12 text-center">
      <p className="text-base font-semibold text-slate-700">
        {hasDirectories ? 'Ask your first question' : 'Link a directory to start'}
      </p>
      <p className="mt-2 max-w-md text-sm leading-6 text-slate-400">
        {hasDirectories
          ? 'The assistant will research only the files visible to this chat.'
          : 'This chat has no visible files yet. Choose one or more directories from the panel.'}
      </p>
    </div>
  );
}
