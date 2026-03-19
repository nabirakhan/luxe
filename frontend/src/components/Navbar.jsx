import { motion, useMotionValue, useSpring, useTransform } from 'motion/react'
import { useState, useEffect, useRef } from 'react'

const NAV_LINKS = [
  { label: 'Problem',      href: '#problem'      },
  { label: 'Solution',     href: '#solution'     },
  { label: 'How it Works', href: '#how-it-works' },
  { label: 'Try It',       href: '#try-it'       },
  { label: 'Privacy',      href: '#privacy'      },
]

const smoothScroll = (e, href) => {
  e.preventDefault()
  document.querySelector(href)?.scrollIntoView({ behavior: 'smooth' })
}

const SPRING   = { mass: 0.1, stiffness: 150, damping: 12 }
const DISTANCE = 120
const MAX_SCALE = 1.22

function NavItem({ label, href, mouseX }) {
  const ref = useRef(null)

  const mouseDistance = useTransform(mouseX, val => {
    const rect = ref.current?.getBoundingClientRect() ?? { x: 0, width: 60 }
    return val - rect.x - rect.width / 2
  })

  const targetScale = useTransform(
    mouseDistance,
    [-DISTANCE, 0, DISTANCE],
    [1, MAX_SCALE, 1]
  )
  const scale = useSpring(targetScale, SPRING)

  return (
    <motion.div
      ref={ref}
      style={{ scale, position: 'relative', display: 'inline-flex' }}
    >
      <a
        href={href}
        onClick={(e) => smoothScroll(e, href)}
        style={{
          color: 'rgba(237,232,224,0.55)',
          textDecoration: 'none',
          fontSize: '11.5px',
          fontWeight: '400',
          letterSpacing: '0.3px',
          padding: '6px 12px',
          borderRadius: '100px',
          transition: 'color 0.25s ease, background 0.25s ease',
          whiteSpace: 'nowrap',
          display: 'block',
        }}
        onMouseEnter={e => {
          e.currentTarget.style.color = 'var(--cream)'
          e.currentTarget.style.background = 'rgba(212,149,107,0.08)'
        }}
        onMouseLeave={e => {
          e.currentTarget.style.color = 'rgba(237,232,224,0.55)'
          e.currentTarget.style.background = 'transparent'
        }}
      >
        {label}
      </a>
    </motion.div>
  )
}

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false)
  const navRef  = useRef(null)
  const mouseX  = useMotionValue(Infinity)
  const isHover = useMotionValue(0)

  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 24)
    window.addEventListener('scroll', handler, { passive: true })
    return () => window.removeEventListener('scroll', handler)
  }, [])

  return (
    <motion.nav
      ref={navRef}
      initial={{ y: -80, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ type: 'spring', stiffness: 200, damping: 22, delay: 0.2 }}
      onMouseMove={({ pageX }) => { isHover.set(1); mouseX.set(pageX) }}
      onMouseLeave={() => { isHover.set(0); mouseX.set(Infinity) }}
      style={{
        position: 'fixed',
        top: '20px',
        left: '50%',
        translateX: '-50%',
        zIndex: 1000,
        display: 'flex',
        alignItems: 'center',
        gap: '4px',
        padding: '5px 6px 5px 18px',
        borderRadius: '100px',
        backgroundColor: scrolled
          ? 'rgba(2,18,12,0.7)'
          : 'rgba(2,14,9,0.45)',
        backdropFilter: 'blur(28px) saturate(180%)',
        WebkitBackdropFilter: 'blur(28px) saturate(180%)',
        border: scrolled
          ? '1px solid rgba(212,149,107,0.2)'
          : '1px solid rgba(212,149,107,0.1)',
        boxShadow: scrolled
          ? '0 8px 40px rgba(0,0,0,0.45), 0 0 0 1px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.07), inset 0 -1px 0 rgba(0,0,0,0.18)'
          : '0 4px 20px rgba(0,0,0,0.22), inset 0 1px 0 rgba(255,255,255,0.05)',
        transition: 'background-color 0.4s ease, box-shadow 0.4s ease, border-color 0.4s ease',
      }}
    >
      <span style={{
        fontFamily: 'var(--font-serif)',
        fontSize: '14px',
        fontWeight: '600',
        color: 'var(--copper)',
        letterSpacing: '3.5px',
        marginRight: '8px',
        userSelect: 'none',
        flexShrink: 0,
        textShadow: '0 0 20px rgba(212,149,107,0.4)',
      }}>
        LUXE
      </span>

      <div style={{
        width: '1px', height: '14px',
        background: 'rgba(212,149,107,0.18)',
        marginRight: '4px', flexShrink: 0,
      }} />

      <div style={{ display: 'flex', alignItems: 'center', gap: '0px' }}>
        {NAV_LINKS.map(({ label, href }) => (
          <NavItem key={label} label={label} href={href} mouseX={mouseX} />
        ))}
      </div>

      <motion.button
        whileHover={{ scale: 1.06, y: -1, boxShadow: '0 6px 24px rgba(212,149,107,0.45)' }}
        whileTap={{ scale: 0.96 }}
        transition={{ type: 'spring', stiffness: 400, damping: 17 }}
        onClick={() => document.querySelector('#try-it')?.scrollIntoView({ behavior: 'smooth' })}
        style={{
          background: 'linear-gradient(135deg, rgba(212,149,107,0.95) 0%, rgba(175,112,65,0.95) 100%)',
          color: '#060C09',
          border: 'none',
          borderRadius: '100px',
          padding: '8px 18px',
          fontSize: '11.5px',
          fontWeight: '600',
          cursor: 'pointer',
          letterSpacing: '0.4px',
          marginLeft: '6px',
          flexShrink: 0,
          boxShadow: '0 4px 16px rgba(212,149,107,0.28), inset 0 1px 0 rgba(255,255,255,0.25)',
          textShadow: '0 1px 2px rgba(0,0,0,0.15)',
        }}
      >
        Get Early Access
      </motion.button>
    </motion.nav>
  )
}
