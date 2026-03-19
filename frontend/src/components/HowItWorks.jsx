import { Fragment, useRef, useEffect, useState } from 'react'
import { motion } from 'motion/react'
import CountUp from './CountUp'

function useInView(threshold = 0.1) {
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

const STEPS = [
  {
    id: '01',
    title: 'Upload your photo',
    desc: 'Held only in RAM for the duration of the session. Never written to disk, never sent to a third-party service.',
  },
  {
    id: '02',
    title: 'Mask & target',
    desc: 'SegFormer-B2 generates a pixel-level mask of clothing and skin. The perturbation budget is concentrated here for maximum effect.',
  },
  {
    id: '03',
    title: 'U-Net fast pass',
    desc: 'A trained encoder-decoder predicts adversarial noise in one forward pass — under 10 s on CPU — within an ε = 8/255 bound that keeps the image visually identical.',
  },
  {
    id: '04',
    title: 'Download & share',
    desc: 'Full-resolution, no watermarks, no quality loss. EOT training ensures protection survives JPEG compression, resizing, and screenshot re-upload.',
  },
]

function StepCircle({ id, index, inView }) {
  const r = 52
  const circumference = 2 * Math.PI * r
  const delay = index * 0.18

  return (
    <motion.div
      initial={{ opacity: 0, y: 24, scale: 0.88 }}
      animate={inView ? { opacity: 1, y: 0, scale: 1 } : {}}
      transition={{ duration: 0.6, delay, ease: [0.22, 1, 0.36, 1] }}
      whileHover={{ scale: 1.07, transition: { type: 'spring', stiffness: 340, damping: 22 } }}
      style={{
        position: 'relative',
        width: '120px',
        height: '120px',
        flexShrink: 0,
        marginBottom: '28px',
        cursor: 'default',
      }}
    >
      <svg
        width="120" height="120"
        viewBox="0 0 120 120"
        style={{ position: 'absolute', inset: 0 }}
      >
        <circle
          cx="60" cy="60" r={r}
          fill="none"
          stroke="rgba(212,149,107,0.12)"
          strokeWidth="1.5"
        />
        <motion.circle
          cx="60" cy="60" r={r}
          fill="none"
          stroke="rgba(212,149,107,0.85)"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference, rotate: -90 }}
          animate={inView ? { strokeDashoffset: 0 } : {}}
          transition={{ duration: 1.1, delay: delay + 0.1, ease: [0.4, 0, 0.2, 1] }}
          style={{ transformOrigin: '60px 60px', rotate: '-90deg' }}
        />
        <motion.circle
          cx="60" cy="8"
          r="3"
          fill="rgba(212,149,107,0.9)"
          initial={{ opacity: 0, scale: 0 }}
          animate={inView ? { opacity: [0, 1, 0.6], scale: [0, 1.4, 1] } : {}}
          transition={{ duration: 0.5, delay: delay + 1.1 }}
        />
      </svg>

      <motion.div
        initial={{ opacity: 0, scale: 0.7 }}
        animate={inView ? { opacity: 1, scale: 1 } : {}}
        transition={{ duration: 0.5, delay: delay + 0.3, ease: 'backOut' }}
        style={{
          position: 'absolute',
          inset: '10px',
          borderRadius: '50%',
          backgroundColor: 'rgba(212,149,107,0.06)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <motion.div
          animate={{ scale: [1, 1.18, 1], opacity: [0.5, 0, 0.5] }}
          transition={{ duration: 2.8, delay: delay + 1.4, repeat: Infinity, ease: 'easeInOut' }}
          style={{
            position: 'absolute', inset: 0,
            borderRadius: '50%',
            border: '1px solid rgba(212,149,107,0.35)',
          }}
        />

        <motion.span
          initial={{ opacity: 0, scale: 0.5 }}
          animate={inView ? { opacity: 1, scale: 1 } : {}}
          transition={{ duration: 0.4, delay: delay + 0.5, ease: 'backOut' }}
        >
          <CountUp
            to={parseInt(id, 10)}
            duration={0.9}
            startWhen={inView}
            style={{
              fontSize: '24px',
              fontWeight: '700',
              color: 'var(--copper)',
              letterSpacing: '1px',
            }}
          />
        </motion.span>
      </motion.div>
    </motion.div>
  )
}

function Arrow({ index, inView }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={inView ? { opacity: 1, x: 0 } : {}}
      transition={{ duration: 0.45, delay: index * 0.18 + 0.55 }}
      style={{
        flexShrink: 0,
        display: 'flex',
        alignItems: 'center',
        height: '120px',
        marginBottom: '28px',
        paddingLeft: '4px',
        paddingRight: '4px',
      }}
    >
      <motion.svg
        width="48" height="48" viewBox="0 0 48 48" fill="none"
        animate={inView ? { x: [0, 5, 0] } : {}}
        transition={{ duration: 1.8, delay: index * 0.18 + 1.3, repeat: Infinity, ease: 'easeInOut' }}
      >
        <motion.path
          d="M6 24 H38 M28 13 L40 24 L28 35"
          stroke="rgba(212,149,107,0.75)"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={inView ? { pathLength: 1, opacity: 1 } : {}}
          transition={{ duration: 0.7, delay: index * 0.18 + 0.65 }}
        />
      </motion.svg>
    </motion.div>
  )
}

export default function HowItWorks() {
  const [sectionRef, inView] = useInView(0.1)

  return (
    <section
      id="how-it-works"
      ref={sectionRef}
      style={{
        background: 'transparent',
        padding: 'clamp(60px,8vw,120px) clamp(28px,6vw,100px) clamp(80px,10vw,140px)',
        position: 'relative',
      }}
    >
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: '1px',
        background: 'linear-gradient(to right, transparent, rgba(212,149,107,0.22), transparent)',
      }} />
      <div style={{
        position: 'absolute', top: '50%', left: '50%',
        transform: 'translate(-50%, -50%)',
        width: '60vw', height: '40vw', borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(0,60,44,0.07) 0%, transparent 70%)',
        filter: 'blur(60px)', pointerEvents: 'none',
      }} />

      <motion.div
        initial={{ opacity: 0, y: 28 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.8 }}
        style={{ textAlign: 'center', marginBottom: '80px' }}
      >
        <div style={{
          fontSize: '10.5px', letterSpacing: '3.5px',
          textTransform: 'uppercase', color: 'var(--copper)',
          marginBottom: '18px', fontWeight: '500',
          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px',
        }}>
          The Process
        </div>
        <h2 style={{
          fontFamily: 'var(--font-serif)',
          fontSize: 'clamp(32px, 4vw, 52px)', fontWeight: '600',
          color: 'var(--cream)', lineHeight: '1.1',
        }}>
          How Luxe{' '}
          <em style={{ color: 'var(--copper)', fontStyle: 'italic' }}>
            Works
          </em>
        </h2>
      </motion.div>

      <div style={{ overflowX: 'auto', width: '100%', paddingBottom: '4px' }}>
      <div style={{
        maxWidth: '1080px',
        minWidth: '680px',
        margin: '0 auto',
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'center',
        gap: '0',
      }}>
        {STEPS.map(({ id, title, desc }, i) => (
          <Fragment key={id}>
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              flex: 1,
              maxWidth: '220px',
            }}>
              <StepCircle id={id} index={i} inView={inView} />

              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={inView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.55, delay: i * 0.18 + 0.55 }}
                style={{ textAlign: 'center' }}
              >
                <div style={{
                  fontSize: '14px', fontWeight: '600',
                  color: 'var(--cream)', marginBottom: '10px',
                  letterSpacing: '0.2px', lineHeight: '1.3',
                }}>
                  {title}
                </div>
                <div style={{
                  fontSize: '13px', lineHeight: '1.7',
                  color: 'rgba(227,220,210,0.45)',
                  fontWeight: '300',
                }}>
                  {desc}
                </div>
              </motion.div>
            </div>

            {i < STEPS.length - 1 && (
              <Arrow key={`arrow-${i}`} index={i} inView={inView} />
            )}
          </Fragment>
        ))}
      </div>
      </div>
    </section>
  )
}
