import { useRef, useEffect, useState, useCallback } from 'react'
import { motion } from 'motion/react'
import PhoneScene from './PhoneScene'
import CountUp from './CountUp'

export default function Hero() {
  const mouseRef        = useRef({ x: 0, y: 0 })
  const scrollRef       = useRef(0)
  const [spinTrigger, setSpinTrigger] = useState(0)
  const spinFiredRef    = useRef(false)

  const handleSpinComplete = useCallback(() => {
    document.body.style.overflow = ''
    document.getElementById('problem')?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    const onMouse  = (e) => {
      mouseRef.current = {
        x: (e.clientX / window.innerWidth  - 0.5) * 2,
        y: (e.clientY / window.innerHeight - 0.5) * 2,
      }
    }
    const onScroll = () => { scrollRef.current = window.scrollY }

    const onWheel = (e) => {
      if (window.scrollY > 80) return
      if (e.deltaY <= 0) return
      e.preventDefault()
      spinFiredRef.current = true
      window.removeEventListener('wheel', onWheel)
      document.body.style.overflow = 'hidden'
      setSpinTrigger(1)
    }

    window.addEventListener('mousemove', onMouse,  { passive: true })
    window.addEventListener('scroll',   onScroll, { passive: true })
    window.addEventListener('wheel',    onWheel,  { passive: false })
    return () => {
      window.removeEventListener('mousemove', onMouse)
      window.removeEventListener('scroll',   onScroll)
      window.removeEventListener('wheel',    onWheel)
    }
  }, [])

  return (
    <section id="hero" className="hero-section">

      <div style={{
        position: 'absolute',
        top: '50%', left: '27%',
        transform: 'translate(-50%, -50%)',
        width: '70vw', height: '70vw',
        borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(0,90,65,0.18) 0%, rgba(0,60,44,0.1) 35%, transparent 65%)',
        pointerEvents: 'none', zIndex: 0,
        filter: 'blur(40px)',
        animation: 'breathe 8s ease-in-out infinite',
      }} />

      <div style={{
        position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 0,
        background: 'radial-gradient(ellipse 55% 70% at 78% 55%, rgba(212,149,107,0.11) 0%, transparent 60%)',
      }} />

      <div style={{
        position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 1,
        backgroundImage: 'url("data:image/svg+xml,%3Csvg viewBox=\'0 0 256 256\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cfilter id=\'noise\'%3E%3CfeTurbulence type=\'fractalNoise\' baseFrequency=\'0.9\' numOctaves=\'4\' stitchTiles=\'stitch\'/%3E%3C/filter%3E%3Crect width=\'100%25\' height=\'100%25\' filter=\'url(%23noise)\' opacity=\'0.04\'/%3E%3C/svg%3E")',
        backgroundRepeat: 'repeat',
      }} />

      <div className="hero-content" style={{ position: 'relative', zIndex: 2 }}>

        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.25 }}
          style={{
            display: 'inline-flex', alignItems: 'center', gap: '10px',
            marginBottom: '36px',
          }}
        >
          <span style={{
            width: 5, height: 5, borderRadius: '50%',
            backgroundColor: 'var(--copper)',
            display: 'inline-block',
            animation: 'pulse 2.4s ease infinite',
            boxShadow: '0 0 8px rgba(212,149,107,0.6)',
          }} />
          <span style={{
            fontSize: '10.5px', fontWeight: '500',
            color: 'var(--copper)', letterSpacing: '3px',
            textTransform: 'uppercase',
          }}>
            Adversarial Image Protection
          </span>
          <span style={{
            fontSize: '10px', color: 'rgba(212,149,107,0.45)',
            border: '1px solid rgba(212,149,107,0.25)',
            borderRadius: '100px', padding: '2px 10px',
            letterSpacing: '1.5px', fontWeight: '400',
          }}>
            BETA
          </span>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 26 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.85, delay: 0.4 }}
          style={{
            fontFamily: 'var(--font-serif)',
            fontSize: 'clamp(46px, 5.5vw, 80px)',
            fontWeight: '600',
            lineHeight: '1.03',
            color: 'var(--cream)',
            marginBottom: '30px',
            letterSpacing: '-0.5px',
          }}
        >
          Your Photos.
          <br />
          <em style={{
            color: 'var(--copper)',
            fontStyle: 'italic',
            textShadow: '0 0 40px rgba(212,149,107,0.2)',
          }}>
            Your Rules.
          </em>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.55 }}
          style={{
            fontSize: '16px', lineHeight: '1.85',
            color: 'rgba(237,232,224,0.58)',
            maxWidth: '380px', marginBottom: '48px', fontWeight: '300',
            letterSpacing: '0.1px',
          }}
        >
          Luxe embeds imperceptible adversarial perturbations into your photos using a trained U-Net — 90× faster than standard PGD — rendering nudification and outfit-modification attacks incoherent, entirely on-device.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.7 }}
          style={{ display: 'flex', gap: '14px', alignItems: 'center' }}
        >
          <motion.button
            whileHover={{ scale: 1.05, y: -2, boxShadow: '0 12px 36px rgba(212,149,107,0.42)' }}
            whileTap={{ scale: 0.96 }}
            transition={{ type: 'spring', stiffness: 380, damping: 18 }}
            onClick={() => document.getElementById('try-it')?.scrollIntoView({ behavior: 'smooth' })}
            style={{
              background: 'linear-gradient(145deg, rgba(212,149,107,0.88) 0%, rgba(175,112,65,0.88) 100%)',
              color: '#060C09',
              border: '1px solid rgba(212,149,107,0.45)',
              borderRadius: '100px',
              padding: '15px 36px', fontSize: '13px', fontWeight: '600',
              cursor: 'pointer', letterSpacing: '0.5px',
              backdropFilter: 'blur(14px) saturate(160%)',
              WebkitBackdropFilter: 'blur(14px) saturate(160%)',
              boxShadow: '0 6px 28px rgba(212,149,107,0.28), inset 0 1px 0 rgba(255,255,255,0.25)',
              textShadow: '0 1px 2px rgba(0,0,0,0.12)',
            }}
          >
            Try It Now
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.03, y: -2, borderColor: 'rgba(0,90,65,0.8)' }}
            whileTap={{ scale: 0.97 }}
            transition={{ type: 'spring', stiffness: 380, damping: 18 }}
            onClick={() => document.getElementById('how-it-works')?.scrollIntoView({ behavior: 'smooth' })}
            style={{
              backgroundColor: 'rgba(1,40,28,0.5)',
              color: 'rgba(237,232,224,0.8)',
              border: '1px solid rgba(1,85,60,0.55)',
              borderRadius: '100px',
              padding: '15px 34px', fontSize: '13px', fontWeight: '400',
              cursor: 'pointer', letterSpacing: '0.4px',
              backdropFilter: 'blur(14px) saturate(160%)',
              WebkitBackdropFilter: 'blur(14px) saturate(160%)',
              boxShadow: '0 4px 16px rgba(0,0,0,0.22), inset 0 1px 0 rgba(255,255,255,0.07)',
              transition: 'border-color 0.25s ease, background-color 0.25s ease',
            }}
          >
            Learn More
          </motion.button>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1.2, delay: 1.1 }}
          style={{
            display: 'flex', gap: '32px', marginTop: '60px',
            paddingTop: '28px',
            borderTop: '1px solid rgba(212,149,107,0.1)',
            position: 'relative',
          }}
        >
          <div style={{
            position: 'absolute', top: 0, left: 0, right: 0, height: '1px',
            background: 'linear-gradient(90deg, transparent, rgba(212,149,107,0.5), transparent)',
            backgroundSize: '200% 100%',
            animation: 'shimmerLine 3s ease infinite',
            overflow: 'hidden',
          }} />

          {[
            { to: 90,  suffix: '×',        lbl: 'Speedup vs PGD'   },
            { to: 10,  suffix: 's', prefix: '<', lbl: 'CPU Inference'   },
            { to: 0,   suffix: 'ms',       lbl: 'Cloud Exposure'   },
          ].map(({ to, prefix, suffix, lbl }) => (
            <div key={lbl}>
              <div style={{ fontSize: '21px', fontWeight: '700', color: 'var(--copper)', textShadow: '0 0 20px rgba(212,149,107,0.3)' }}>
                <CountUp to={to} prefix={prefix} suffix={suffix} duration={1.8} />
              </div>
              <div style={{
                fontSize: '10px', color: 'rgba(237,232,224,0.38)',
                letterSpacing: '0.8px', marginTop: '4px', textTransform: 'uppercase',
              }}>
                {lbl}
              </div>
            </div>
          ))}
        </motion.div>
      </div>

      <div className="hero-canvas-wrap" style={{ zIndex: 2 }}>
        <PhoneScene
          mouseRef={mouseRef}
          spinTrigger={spinTrigger}
          onSpinComplete={handleSpinComplete}
        />
      </div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.9, duration: 1.2 }}
        style={{
          position: 'absolute', bottom: '32px', left: '50%',
          transform: 'translateX(-50%)',
          display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px',
          zIndex: 2,
        }}
      >
        <span style={{ fontSize: '9px', letterSpacing: '4px', textTransform: 'uppercase', color: 'rgba(237,232,224,0.25)' }}>
          scroll
        </span>
        <div style={{
          width: '1px', height: '44px',
          background: 'linear-gradient(to bottom, rgba(212,149,107,0.65), transparent)',
          animation: 'scrollBlink 2.2s ease infinite',
        }} />
      </motion.div>
    </section>
  )
}