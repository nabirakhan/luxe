import { motion } from 'motion/react'

const TEAM = ['Nabira', 'Rameen', 'Aisha']
const COURSE = 'Deep Learning Practice  ·  2026'

function GithubIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
      <path
        d="M12 2C6.477 2 2 6.477 2 12c0 4.418 2.865 8.167 6.839 9.49.5.09.682-.217.682-.482 0-.237-.009-.868-.013-1.703-2.782.605-3.369-1.342-3.369-1.342-.454-1.154-1.11-1.461-1.11-1.461-.908-.62.069-.608.069-.608 1.004.07 1.532 1.032 1.532 1.032.892 1.53 2.341 1.088 2.91.832.091-.647.35-1.088.636-1.338-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.03-2.682-.103-.253-.447-1.27.097-2.646 0 0 .84-.269 2.75 1.026A9.578 9.578 0 0 1 12 6.836a9.58 9.58 0 0 1 2.504.337c1.909-1.295 2.747-1.026 2.747-1.026.546 1.376.202 2.394.1 2.646.64.698 1.026 1.591 1.026 2.682 0 3.841-2.337 4.687-4.565 4.935.359.309.678.919.678 1.852 0 1.337-.012 2.416-.012 2.743 0 .267.18.578.688.48C19.138 20.163 22 16.417 22 12c0-5.523-4.477-10-10-10z"
        fill="currentColor"
      />
    </svg>
  )
}

export default function Footer() {
  return (
    <footer style={{
      background: 'rgba(2,8,5,0.85)',
      backdropFilter: 'blur(20px) saturate(160%)',
      WebkitBackdropFilter: 'blur(20px) saturate(160%)',
      borderTop: '1px solid rgba(227,220,210,0.1)',
      boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.05)',
      padding: 'clamp(40px,5vw,72px) clamp(28px,6vw,100px) clamp(36px,4vw,56px)',
    }}>
      <div style={{
        maxWidth: '1080px', margin: '0 auto',
        display: 'grid',
        gridTemplateColumns: '1fr auto 1fr',
        alignItems: 'center', gap: '32px',
      }}>
        <div style={{
          fontFamily: 'var(--font-serif)',
          fontSize: '20px', fontWeight: '600',
          color: 'var(--copper)', letterSpacing: '4px', userSelect: 'none',
        }}>
          LUXE
        </div>

        <motion.a
          href="https://github.com"
          target="_blank"
          rel="noopener noreferrer"
          whileHover={{ color: 'var(--cream)', y: -2 }}
          transition={{ type: 'spring', stiffness: 400, damping: 20 }}
          style={{
            display: 'flex', alignItems: 'center', gap: '8px',
            color: 'rgba(227,220,210,0.65)',
            textDecoration: 'none', fontSize: '13px', fontWeight: '400',
          }}
        >
          <GithubIcon />
          View on GitHub
        </motion.a>

        <div style={{ textAlign: 'right' }}>
          <div style={{
            fontSize: '12px', color: 'rgba(227,220,210,0.7)',
            marginBottom: '4px', letterSpacing: '0.3px',
          }}>
            {TEAM.join(' · ')}
          </div>
          <div style={{
            fontSize: '11px', color: 'rgba(227,220,210,0.45)',
            letterSpacing: '0.5px',
          }}>
            {COURSE}
          </div>
        </div>
      </div>

      <div style={{
        maxWidth: '1080px', margin: '40px auto 0',
        borderTop: '1px solid rgba(227,220,210,0.05)',
        paddingTop: '20px',
        textAlign: 'center',
        fontSize: '11px', color: 'rgba(227,220,210,0.4)',
        letterSpacing: '0.5px',
      }}>
        © 2026 Luxe. All rights reserved.
      </div>
    </footer>
  )
}
