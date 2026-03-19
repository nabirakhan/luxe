import { useRef, useEffect, useState } from 'react'
import { motion } from 'motion/react'

function useInView(threshold = 0.15) {
  const ref = useRef(null)
  const [inView, setInView] = useState(false)
  useEffect(() => {
    const obs = new IntersectionObserver(
      ([e]) => { if (e.isIntersecting) { setInView(true); obs.disconnect() } },
      { threshold }
    )
    if (ref.current) obs.observe(ref.current)
    return () => obs.disconnect()
  }, [threshold])
  return [ref, inView]
}

function FragmentedPhoto() {
  const strips = [
    { yOffset: -6, xOffset: 3,  clip: 'polygon(0 0, 100% 0, 100% 22%, 0 22%)' },
    { yOffset: 2,  xOffset: -5, clip: 'polygon(0 22%, 100% 22%, 100% 44%, 0 44%)' },
    { yOffset: -3, xOffset: 7,  clip: 'polygon(0 44%, 100% 44%, 100% 66%, 0 66%)' },
    { yOffset: 5,  xOffset: -3, clip: 'polygon(0 66%, 100% 66%, 100% 84%, 0 84%)' },
    { yOffset: -2, xOffset: 4,  clip: 'polygon(0 84%, 100% 84%, 100% 100%, 0 100%)' },
  ]

  return (
    <div style={{ position: 'relative', width: '260px', height: '320px', flexShrink: 0 }}>
      {strips.map(({ yOffset, xOffset, clip }, i) => (
        <div key={i} style={{
          position: 'absolute', inset: 0,
          backgroundColor: i % 2 === 0 ? 'rgba(212,149,107,0.12)' : 'rgba(237,232,224,0.06)',
          border: '1px solid rgba(212,149,107,0.2)',
          clipPath: clip,
          transform: `translateX(${xOffset}px) translateY(${yOffset}px)`,
          transition: 'transform 4s ease infinite',
        }} />
      ))}
      {[18, 44, 64].map((top, i) => (
        <div key={i} style={{
          position: 'absolute', left: 0, right: 0, top: `${top}%`,
          height: '1px',
          background: `rgba(212,149,107,${0.15 + i * 0.08})`,
          transform: `translateX(${i % 2 === 0 ? -6 : 6}px)`,
        }} />
      ))}
      {[
        { top: 0, left: 0, borderTop: '1px solid', borderLeft: '1px solid' },
        { top: 0, right: 0, borderTop: '1px solid', borderRight: '1px solid' },
        { bottom: 0, left: 0, borderBottom: '1px solid', borderLeft: '1px solid' },
        { bottom: 0, right: 0, borderBottom: '1px solid', borderRight: '1px solid' },
      ].map((style, i) => (
        <div key={i} style={{
          position: 'absolute', width: 20, height: 20,
          borderColor: 'rgba(212,149,107,0.5)',
          ...style,
        }} />
      ))}
      <div style={{
        position: 'absolute', top: '50%', left: '50%',
        transform: 'translate(-50%, -50%) rotate(-12deg)',
        fontSize: '10px', fontWeight: '600', letterSpacing: '3px',
        textTransform: 'uppercase', color: 'rgba(212,149,107,0.45)',
        whiteSpace: 'nowrap',
        border: '1px solid rgba(212,149,107,0.25)',
        padding: '6px 14px',
      }}>
        Manipulated
      </div>
    </div>
  )
}

export default function Problem() {
  const [ref, inView] = useInView(0.12)

  return (
    <section
      ref={ref}
      id="problem"
      style={{
        background: 'transparent',
        padding: 'clamp(60px,8vw,120px) clamp(28px,6vw,100px)',
        position: 'relative', overflow: 'hidden',
      }}
    >
      <div style={{
        position: 'absolute', top: 0,
        left: 'clamp(28px,6vw,100px)', right: 'clamp(28px,6vw,100px)',
        height: '1px',
        background: 'linear-gradient(to right, transparent, rgba(212,149,107,0.25), transparent)',
      }} />

      <div style={{
        position: 'absolute', top: '30%', left: '-10vw',
        width: '40vw', height: '40vw', borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(212,149,107,0.07) 0%, transparent 65%)',
        filter: 'blur(40px)', pointerEvents: 'none',
      }} />

      <div style={{
        maxWidth: '1100px', margin: '0 auto',
        display: 'flex', alignItems: 'center', gap: 'clamp(40px, 6vw, 96px)',
      }}>
        <motion.div
          animate={{ opacity: inView ? 1 : 0, x: inView ? 0 : -32 }}
          transition={{ duration: 0.85 }}
          style={{ flex: 1 }}
        >
          <div style={{
            fontSize: '10.5px', letterSpacing: '3.5px',
            textTransform: 'uppercase', color: 'var(--copper)',
            marginBottom: '22px', fontWeight: '500',
            display: 'flex', alignItems: 'center', gap: '10px',
          }}>
            <span style={{ width: 22, height: 1, background: 'var(--copper)', opacity: 0.6, display: 'inline-block' }} />
            The Problem
          </div>

          <h2 style={{
            fontFamily: 'var(--font-serif)',
            fontSize: 'clamp(36px, 4vw, 60px)', fontWeight: '600',
            color: 'var(--cream)', lineHeight: '1.08',
            marginBottom: '32px', letterSpacing: '-0.3px',
          }}>
            One Photo Can
            <br />
            <em style={{ color: 'var(--copper)', fontStyle: 'italic', textShadow: '0 0 30px rgba(212,149,107,0.18)' }}>
              Destroy a Life.
            </em>
          </h2>

          <p style={{
            fontSize: '17px', lineHeight: '1.85',
            color: 'rgba(237,232,224,0.56)', fontWeight: '300',
            maxWidth: '460px', marginBottom: '22px', letterSpacing: '0.1px',
          }}>
            Diffusion-based tools can strip clothing or alter appearance using nothing more
            than a single uploaded image and a text prompt. This affects women disproportionately
            and causes real-world harassment, coercion, and reputational harm.
          </p>

          <p style={{
            fontSize: '17px', lineHeight: '1.85',
            color: 'rgba(237,232,224,0.56)', fontWeight: '300',
            maxWidth: '460px',
          }}>
            Tools like PhotoGuard and Glaze address artistic style theft — not body privacy.
            Neither handles nudification and outfit-modification simultaneously.
            Luxe is the first tool designed for both threat classes at once.
          </p>
        </motion.div>

        <motion.div
          animate={{ opacity: inView ? 1 : 0, x: inView ? 0 : 32 }}
          transition={{ duration: 0.85, delay: 0.15 }}
          style={{ flexShrink: 0, position: 'relative' }}
        >
          <div style={{
            position: 'absolute', inset: '-20px',
            borderRadius: '28px',
            background: 'radial-gradient(ellipse, rgba(0,60,44,0.2) 0%, transparent 70%)',
            filter: 'blur(20px)',
            pointerEvents: 'none',
          }} />
          <video
            src="/scroll.mp4"
            autoPlay loop muted playsInline
            style={{
              width: 'min(580px, 48vw)',
              borderRadius: '18px',
              display: 'block',
              objectFit: 'cover',
              border: '1px solid rgba(212,149,107,0.1)',
              boxShadow: '0 24px 72px rgba(0,0,0,0.5), 0 0 0 1px rgba(0,0,0,0.2)',
              position: 'relative',
            }}
          />
        </motion.div>
      </div>
    </section>
  )
}
