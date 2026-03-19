import { useRef, useEffect, useState } from 'react'
import { motion } from 'framer-motion'

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

const THEME = {
  copper: '#D4956B',
  cream: '#E3DCD2',
  cardBg: 'rgba(20, 25, 24, 0.6)',
  border: 'rgba(212, 149, 107, 0.15)',
}

const CARDS = [
  {
    title: 'RAM-Only Logic',
    body: 'Processing occurs strictly within volatile memory. When the session ends, the data ceases to exist at a hardware level. No disk, no trace.',
    align: 'left',
    number: '01'
  },
  {
    title: 'Zero Tracking',
    body: 'No accounts, no cookies, no analytics. Our architecture is designed so that the tool has no knowledge of who you are.',
    align: 'right',
    number: '02'
  },
  {
    title: 'ε-Bounded Fidelity',
    body: 'Indistinguishable to the eye, invisible to AI. Perturbations are capped so strictly that visual quality remains flawless.',
    align: 'left',
    number: '03'
  },
  {
    title: 'Platform Robust',
    body: 'Engineered to survive. Protection remains active even after aggressive JPEG compression or social media re-uploads.',
    align: 'right',
    number: '04'
  }
]

export default function Privacy() {
  const [ref, inView] = useInView(0.08)

  return (
    <section
      id="privacy"
      ref={ref}
      style={{
        background: 'transparent',
        padding: 'clamp(60px,8vw,120px) 24px',
        position: 'relative',
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center'
      }}
    >
      <div style={{ maxWidth: '1100px', margin: '0 auto', width: '100%' }}>
        <motion.div
          initial={{ opacity: 0, y: 28 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
          style={{ textAlign: 'center', marginBottom: '80px' }}
        >
          <div style={{
            fontSize: '10.5px', 
            letterSpacing: '3.5px', 
            textTransform: 'uppercase',
            color: THEME.copper, 
            marginBottom: '18px', 
            fontWeight: '500',
            display: 'flex', 
            alignItems: 'center', 
            gap: '10px', 
            justifyContent: 'center',
          }}>
            Privacy
          </div>
          <h2 style={{
            fontFamily: 'serif',
            fontSize: 'clamp(32px, 4vw, 52px)', 
            fontWeight: '600',
            color: THEME.cream, 
            lineHeight: '1.1',
          }}>
            Built around{' '}
            <em style={{ color: THEME.copper, fontStyle: 'italic' }}>your trust.</em>
          </h2>
          <p style={{
            marginTop: '20px', 
            fontSize: '16px', 
            lineHeight: '1.8',
            color: 'rgba(227,220,210,0.5)', 
            fontWeight: '300',
            maxWidth: '460px', 
            margin: '20px auto 0',
          }}>
            Every technical decision is built around one principle: your data is yours.
          </p>
        </motion.div>

        <div style={{ 
          display: 'flex', 
          flexDirection: 'column', 
          gap: '24px',
          position: 'relative'
        }}>
          {CARDS.map((card, i) => (
            <motion.div
              key={card.title}
              initial={{ opacity: 0, x: card.align === 'left' ? -40 : 40 }}
              animate={inView ? { opacity: 1, x: 0 } : {}}
              transition={{ duration: 0.7, delay: i * 0.15 }}
              whileHover={{ scale: 1.01, borderColor: THEME.copper }}
              style={{
                alignSelf: card.align === 'left' ? 'flex-start' : 'flex-end',
                width: '100%',
                maxWidth: '540px',
                background: THEME.cardBg,
                backdropFilter: 'blur(16px)',
                border: `1px solid ${THEME.border}`,
                borderRadius: '28px',
                padding: '40px',
                position: 'relative',
                boxShadow: '0 15px 35px rgba(0,0,0,0.2)',
                cursor: 'default'
              }}
            >
              <motion.div 
                animate={{ 
                  opacity: [0.4, 0.7, 0.4],
                  textShadow: [
                    `0 0 10px ${THEME.copper}66`, 
                    `0 0 20px ${THEME.copper}aa`, 
                    `0 0 10px ${THEME.copper}66`
                  ]
                }}
                transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                style={{
                  position: 'absolute', 
                  top: '30px', 
                  right: '40px',
                  fontFamily: 'serif', 
                  fontSize: '42px', 
                  fontWeight: '700',
                  color: THEME.copper,
                  pointerEvents: 'none',
                  zIndex: 0
                }}
              >
                {card.number}
              </motion.div>

              <div style={{ position: 'relative', zIndex: 1 }}>
                <h3 style={{ 
                  color: THEME.cream, 
                  fontSize: '22px', 
                  marginBottom: '14px',
                  fontWeight: '600',
                  letterSpacing: '-0.01em'
                }}>
                  {card.title}
                </h3>
                <p style={{ 
                  color: 'rgba(227, 220, 210, 0.5)', 
                  fontSize: '15px', 
                  lineHeight: '1.7',
                  margin: 0,
                  fontWeight: '300'
                }}>
                  {card.body}
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}