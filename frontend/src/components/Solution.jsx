import { useRef, useEffect, useState } from 'react'
import { motion } from 'motion/react'
import Cubes from './Cubes'

function useInView(threshold = 0.12) {
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

const POINTS = [
  {
    label: 'Semantic Masking',
    body: 'A SegFormer-B2 model identifies clothing and skin regions. The adversarial budget targets these areas — making the attack precise, efficient, and harder to defeat.',
  },
  {
    label: 'Dual PGD Attack',
    body: 'Two separate PGD loops attack the VAE encoder of a fine-tuned SD v1.5 inpainting model and the conditioning of InstructPix2Pix/IP-Adapter — covering nudification and outfit-swap threats simultaneously.',
  },
  {
    label: 'U-Net Fast Path',
    body: 'A trained encoder-decoder predicts the protection perturbation in a single forward pass — reducing inference from 5–10 minutes to under 10 seconds on CPU. 90× speedup. Zero quality loss.',
  },
]

export default function Solution() {
  const [ref, inView] = useInView(0.1)

  return (
    <section
      id="solution"
      ref={ref}
      style={{
        background: 'transparent',
        padding: 'clamp(60px,8vw,120px) clamp(28px,6vw,100px)',
        position: 'relative', overflow: 'hidden',
      }}
    >
      <div style={{
        position: 'absolute', inset: 0, pointerEvents: 'none',
        background: 'radial-gradient(ellipse 70% 50% at 20% 60%, rgba(0,80,58,0.1) 0%, transparent 65%)',
      }} />

      <div style={{
        position: 'absolute', left: '-120px', bottom: '-120px',
        width: '420px', height: '420px', borderRadius: '50%',
        border: '1px solid rgba(212,149,107,0.06)',
        pointerEvents: 'none',
      }} />
      <div style={{
        position: 'absolute', left: '-80px', bottom: '-80px',
        width: '280px', height: '280px', borderRadius: '50%',
        border: '1px solid rgba(212,149,107,0.04)',
        pointerEvents: 'none',
      }} />

      <div style={{
        maxWidth: '1100px', margin: '0 auto',
        display: 'flex', alignItems: 'center', gap: 'clamp(40px, 6vw, 96px)',
      }}>
        <motion.div
          animate={{ opacity: inView ? 1 : 0, scale: inView ? 1 : 0.94 }}
          transition={{ duration: 1, ease: 'easeOut' }}
          style={{
            width: 'clamp(280px, 30vw, 400px)',
            height: 'clamp(280px, 30vw, 400px)',
            flexShrink: 0, cursor: 'crosshair',
            position: 'relative',
          }}
        >
          <div style={{
            position: 'absolute', inset: '-30px', borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(212,149,107,0.1) 0%, transparent 70%)',
            filter: 'blur(30px)', pointerEvents: 'none',
          }} />
          <Cubes
            gridSize={9}
            imageUrl="/woman.png"
            faceColor="var(--green)"
            borderStyle="1px solid rgba(1,51,40,0.6)"
            rippleColor="#D4956B"
            maxAngle={40}
            radius={3.5}
            autoAnimate={true}
            rippleOnClick={true}
            duration={{ enter: 0.28, leave: 0.55 }}
          />
        </motion.div>

        <div style={{ flex: 1 }}>
          <motion.div
            animate={{ opacity: inView ? 1 : 0, y: inView ? 0 : 28 }}
            transition={{ duration: 0.8, delay: 0.1 }}
          >
            <div style={{
              fontSize: '10.5px', letterSpacing: '3.5px',
              textTransform: 'uppercase', color: 'var(--copper)',
              marginBottom: '22px', fontWeight: '500',
              display: 'flex', alignItems: 'center', gap: '10px',
            }}>
              <span style={{ width: 22, height: 1, background: 'var(--copper)', opacity: 0.6, display: 'inline-block' }} />
              The Solution
            </div>

            <h2 style={{
              fontFamily: 'var(--font-serif)',
              fontSize: 'clamp(34px, 4vw, 58px)', fontWeight: '600',
              color: 'var(--cream)', lineHeight: '1.08',
              marginBottom: '52px', letterSpacing: '-0.3px',
            }}>
              Protection that's
              <br />
              <em style={{ color: 'var(--copper)', fontStyle: 'italic', textShadow: '0 0 30px rgba(212,149,107,0.18)' }}>
                imperceptible.
              </em>
            </h2>
          </motion.div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '36px' }}>
            {POINTS.map(({ label, body }, i) => (
              <motion.div
                key={label}
                animate={{ opacity: inView ? 1 : 0, x: inView ? 0 : 24 }}
                transition={{ duration: 0.7, delay: inView ? 0.25 + i * 0.15 : 0 }}
                style={{ display: 'flex', gap: '22px', alignItems: 'flex-start' }}
              >
                <div style={{
                  fontFamily: 'var(--font-serif)', fontSize: '13px',
                  fontWeight: '600', color: 'rgba(212,149,107,0.5)',
                  marginTop: '2px', minWidth: '24px',
                  letterSpacing: '1px',
                }}>
                  {String(i + 1).padStart(2, '0')}
                </div>

                <div>
                  <div style={{
                    fontSize: '15px', fontWeight: '500',
                    color: 'var(--cream)', marginBottom: '7px',
                    letterSpacing: '0.2px',
                  }}>
                    {label}
                  </div>
                  <div style={{
                    fontSize: '14.5px', lineHeight: '1.78',
                    color: 'rgba(237,232,224,0.48)', fontWeight: '300',
                  }}>
                    {body}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          <motion.div
            animate={{ opacity: inView ? 1 : 0 }}
            transition={{ duration: 1, delay: 0.9 }}
            style={{
              marginTop: '48px',
              fontSize: '10.5px', letterSpacing: '2.5px',
              textTransform: 'uppercase', color: 'rgba(212,149,107,0.35)',
              display: 'flex', alignItems: 'center', gap: '8px',
            }}
          >
            <span style={{ width: 16, height: 1, background: 'rgba(212,149,107,0.3)', display: 'inline-block' }} />
            Move cursor over the grid to interact
          </motion.div>
        </div>
      </div>
    </section>
  )
}
