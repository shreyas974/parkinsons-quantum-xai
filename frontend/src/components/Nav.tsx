const LINKS = [
  ["#signals", "Signals"],
  ["#quantum", "Qubits"],
  ["#screen", "Try it"],
  ["#research", "Research"],
  ["#about", "About"],
];

export function Nav() {
  return (
    <header className="nav">
      <a className="brand" href="#top" aria-label="QPD, back to top">
        <svg viewBox="0 0 64 64" width="26" height="26" aria-hidden="true">
          <path d="M32 32c1.5 0 2.6 1.3 2.4 2.8-.3 2.5-3.3 3.6-5.5 2.5-3.4-1.7-3.6-6.4-1.2-9 3.3-3.6 9-3.2 12 .4 3.6 4.4 2.7 11-1.5 14.5-5 4.3-12.7 3.4-16.7-1.6-4.6-5.7-3.4-14.2 2.3-18.6 6.3-4.9 15.6-3.6 20.4 2.7" fill="none" stroke="currentColor" strokeWidth="3.4" strokeLinecap="round" />
        </svg>
        <span>QPD</span>
      </a>
      <nav aria-label="Sections">
        {LINKS.map(([href, label]) => (
          <a key={href} href={href} className={href === "#screen" ? "nav-cta" : undefined}>
            {label}
          </a>
        ))}
      </nav>
    </header>
  );
}
