import { useRef, useState, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'motion/react'
import PixelCard from './PixelCard'

const MODES = [
  { id: 'nudify', label: 'Nudification Shield',  desc: 'Attacks the VAE encoder of fine-tuned SD v1.5 inpainting models. Corrupts the latent representation nudification pipelines depend on.' },
  { id: 'modify', label: 'Outfit-Swap Guard',    desc: 'Attacks InstructPix2Pix and IP-Adapter conditioning. Disrupts appearance-modification and outfit-swap threat class.' },
]

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function BeforeAfterSlider({ original, protected: protectedSrc }) {
  const [pos, setPos] = useState(50)
  const containerRef  = useRef(null)
  const dragging      = useRef(false)

  const update = useCallback((clientX) => {
    if (!containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const pct  = Math.max(0, Math.min(((clientX - rect.left) / rect.width) * 100, 100))
    setPos(pct)
  }, [])

  const onMouseMove  = useCallback((e) => { if (dragging.current) update(e.clientX) }, [update])
  const onTouchMove  = useCallback((e) => { if (dragging.current) update(e.touches[0].clientX) }, [update])
  const stopDrag     = useCallback(() => { dragging.current = false }, [])

  useEffect(() => {
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup',   stopDrag)
    window.addEventListener('touchmove', onTouchMove, { passive: true })
    window.addEventListener('touchend',  stopDrag,    { passive: true })
    return () => {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup',   stopDrag)
      window.removeEventListener('touchmove', onTouchMove)
      window.removeEventListener('touchend',  stopDrag)
    }
  }, [onMouseMove, onTouchMove, stopDrag])

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative', width: '100%', height: '360px',
        borderRadius: '12px', overflow: 'hidden',
        border: '1px solid rgba(227,220,210,0.1)',
        cursor: 'ew-resize', userSelect: 'none',
      }}
    >
      {/* Protected image (right side, always underneath) */}
      <img
        src={protectedSrc}
        alt="Protected"
        style={{
          position: 'absolute', inset: 0,
          width: '100%', height: '100%', objectFit: 'cover',
          filter: 'hue-rotate(4deg) saturate(0.97) contrast(1.015)',
        }}
      />

      {/* Original image (left side, clipped) */}
      <div style={{
        position: 'absolute', inset: 0,
        clipPath: `inset(0 ${100 - pos}% 0 0)`,
        transition: dragging.current ? 'none' : 'clip-path 0.05s ease',
      }}>
        <img
          src={original}
          alt="Original"
          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
        />
        <div style={{
          position: 'absolute', top: '14px', left: '14px',
          fontSize: '10px', letterSpacing: '2px', textTransform: 'uppercase',
          color: 'rgba(227,220,210,0.6)',
          backgroundColor: 'rgba(16,12,13,0.55)',
          padding: '4px 12px', borderRadius: '100px',
          backdropFilter: 'blur(8px)',
        }}>
          Original
        </div>
      </div>

      <div style={{
        position: 'absolute', top: '14px', right: '14px',
        fontSize: '10px', letterSpacing: '2px', textTransform: 'uppercase',
        color: 'var(--copper)',
        backgroundColor: 'rgba(16,12,13,0.55)',
        padding: '4px 12px', borderRadius: '100px',
        backdropFilter: 'blur(8px)',
      }}>
        Protected
      </div>

      <div style={{
        position: 'absolute', top: 0, bottom: 0, left: `${pos}%`,
        width: '2px', backgroundColor: 'var(--copper)',
        transform: 'translateX(-50%)',
        boxShadow: '0 0 12px rgba(212,149,107,0.6)',
      }}>
        <div
          onMouseDown={() => { dragging.current = true }}
          onTouchStart={() => { dragging.current = true }}
          style={{
            position: 'absolute', top: '50%', left: '50%',
            transform: 'translate(-50%, -50%)',
            width: '38px', height: '38px', borderRadius: '50%',
            backgroundColor: 'var(--copper)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 2px 16px rgba(212,149,107,0.5)',
            cursor: 'ew-resize',
          }}
        >
          <svg width="16" height="10" viewBox="0 0 16 10" fill="none">
            <path d="M1 5h14M5 1L1 5l4 4M11 1l4 4-4 4" stroke="#100C0D" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
      </div>
    </div>
  )
}

const STATE = { IDLE: 'idle', PREVIEW: 'preview', PROCESSING: 'processing', DONE: 'done' }

export default function TryIt() {
  const [mode,            setMode]            = useState('nudify')
  const [stage,           setStage]           = useState(STATE.IDLE)
  const [imageSrc,        setImageSrc]        = useState(null)   // original data URL
  const [originalResized, setOriginalResized] = useState(null)   // 512×512 canvas data URL
  const [protectedSrc,    setProtectedSrc]    = useState(null)   // blob URL of protected PNG
  const [unetMissing,     setUnetMissing]     = useState(false)
  const [errorMsg,        setErrorMsg]        = useState(null)
  const [progress,        setProgress]        = useState(0)
  const [draggingOver,    setDraggingOver]    = useState(false)
  const inputRef    = useRef(null)
  const fileRef     = useRef(null)  // holds the raw File for FormData

  const loadFile = (file) => {
    if (!file || !file.type.startsWith('image/')) return
    fileRef.current = file
    const reader = new FileReader()
    reader.onload = (e) => { setImageSrc(e.target.result); setStage(STATE.PREVIEW) }
    reader.readAsDataURL(file)
  }

  const onDrop = (e) => {
    e.preventDefault(); setDraggingOver(false)
    loadFile(e.dataTransfer.files[0])
  }

  const startProtection = async () => {
    if (!fileRef.current || !imageSrc) return
    setStage(STATE.PROCESSING)
    setProgress(0)
    setErrorMsg(null)

    // Animate progress bar while waiting for the API
    let p = 0
    const iv = setInterval(() => {
      p += Math.random() * 2 + 0.5
      setProgress(Math.min(p, 92))
    }, 200)

    try {
      const form = new FormData()
      form.append('file', fileRef.current)
      form.append('mode', mode)
      form.append('texture', 'false')

      const resp = await fetch(`${API_URL}/protect`, { method: 'POST', body: form })

      if (resp.status === 429) {
        throw new Error('Server is busy — please try again in a moment.')
      }
      if (!resp.ok) {
        const text = await resp.text().catch(() => '')
        throw new Error(`Protection failed (${resp.status})${text ? ': ' + text : ''}`)
      }

      const checkpointStatus = resp.headers.get('X-Checkpoint-Status') || ''
      setUnetMissing(checkpointStatus.includes('MISSING'))

      const blob = await resp.blob()
      const blobUrl = URL.createObjectURL(blob)
      setProtectedSrc(blobUrl)

      // Resize original to 512×512 via canvas so slider sides match
      const img = new Image()
      img.onload = () => {
        const canvas = document.createElement('canvas')
        canvas.width = canvas.height = 512
        canvas.getContext('2d').drawImage(img, 0, 0, 512, 512)
        setOriginalResized(canvas.toDataURL())
        clearInterval(iv)
        setProgress(100)
        setStage(STATE.DONE)
      }
      img.src = imageSrc

    } catch (err) {
      clearInterval(iv)
      setErrorMsg(err.message || 'Something went wrong.')
      setStage(STATE.PREVIEW)
    }
  }

  const download = () => {
    if (!protectedSrc) return
    const a = document.createElement('a')
    a.href     = protectedSrc
    a.download = 'luxe-protected.png'
    a.click()
  }

  const reset = () => {
    if (protectedSrc) URL.revokeObjectURL(protectedSrc)
    setStage(STATE.IDLE)
    setImageSrc(null)
    setOriginalResized(null)
    setProtectedSrc(null)
    setUnetMissing(false)
    setErrorMsg(null)
    setProgress(0)
    fileRef.current = null
  }

  return (
    <section
      id="try-it"
      style={{
        background: 'transparent',
        padding: 'clamp(60px,8vw,120px) clamp(28px,6vw,100px)',
        position: 'relative', overflow: 'hidden',
      }}
    >
      <div style={{
        position: 'absolute', top: 0, left: '80px', right: '80px', height: '1px',
        background: 'linear-gradient(to right, transparent, rgba(212,149,107,0.18), transparent)',
      }} />

      <div style={{ maxWidth: '840px', margin: '0 auto' }}>
        <div style={{ textAlign: 'center', marginBottom: '64px' }}>
          <div style={{
            fontSize: '11px', letterSpacing: '3px',
            textTransform: 'uppercase', color: 'var(--copper)',
            marginBottom: '18px', fontWeight: '500',
          }}>
            Try It
          </div>
          <h2 style={{
            fontFamily: 'var(--font-serif)',
            fontSize: 'clamp(34px, 4vw, 52px)', fontWeight: '600',
            color: 'var(--cream)', lineHeight: '1.1',
          }}>
            Protect a photo{' '}
            <em style={{ color: 'var(--copper)', fontStyle: 'italic' }}>
              right now.
            </em>
          </h2>
        </div>

        <div style={{
          display: 'flex', gap: '12px', marginBottom: '40px', justifyContent: 'center',
        }}>
          {MODES.map(({ id, label, desc }) => (
            <button
              key={id}
              onClick={() => setMode(id)}
              style={{
                backgroundColor: mode === id ? 'rgba(212,149,107,0.12)' : 'transparent',
                border: `1px solid ${mode === id ? 'rgba(212,149,107,0.45)' : 'rgba(227,220,210,0.12)'}`,
                borderRadius: '16px', padding: '20px 26px',
                cursor: 'pointer', transition: 'all 0.25s ease',
                textAlign: 'left', flex: 1, maxWidth: '320px',
              }}
            >
              <div style={{
                fontSize: '13px', fontWeight: '600',
                color: mode === id ? 'var(--copper)' : 'rgba(227,220,210,0.6)',
                marginBottom: '6px', transition: 'color 0.25s ease',
              }}>
                {label}
              </div>
              <div style={{
                fontSize: '12px', lineHeight: '1.6',
                color: 'rgba(227,220,210,0.38)', fontWeight: '300',
              }}>
                {desc}
              </div>
            </button>
          ))}
        </div>

        <AnimatePresence mode="wait">
          {stage === STATE.IDLE && (
            <motion.div
              key="idle"
              initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -16 }}
              transition={{ duration: 0.4 }}
            >
              <PixelCard
                colors="#CC8B65,#a86840,#7a4220"
                gap={6}
                speed={55}
                style={{
                  width: '100%',
                  padding: '80px 40px',
                  cursor: 'pointer',
                  backgroundColor: draggingOver ? 'rgba(212,149,107,0.05)' : 'rgba(227,220,210,0.01)',
                  borderColor: draggingOver ? 'rgba(212,149,107,0.7)' : undefined,
                }}
                onDragOver={(e) => { e.preventDefault(); setDraggingOver(true) }}
                onDragLeave={() => setDraggingOver(false)}
                onDrop={onDrop}
                onClick={() => inputRef.current?.click()}
              >
                <div style={{ position: 'relative', zIndex: 1, textAlign: 'center', pointerEvents: 'none' }}>
                  <svg width="40" height="40" viewBox="0 0 40 40" fill="none" style={{ marginBottom: '20px', opacity: 0.45 }}>
                    <circle cx="20" cy="20" r="19" stroke="#E3DCD2" strokeWidth="1"/>
                    <path d="M20 26V14M14 20l6-6 6 6" stroke="#CC8B65" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  <div style={{
                    fontFamily: 'var(--font-serif)', fontSize: '20px',
                    color: 'var(--cream)', marginBottom: '10px',
                  }}>
                    Drag and drop your photo
                  </div>
                  <div style={{ fontSize: '13px', color: 'rgba(227,220,210,0.38)', fontWeight: '300' }}>
                    or click to browse — JPG, PNG, WEBP
                  </div>
                </div>
                <input
                  ref={inputRef} type="file" accept="image/*"
                  onChange={(e) => loadFile(e.target.files[0])}
                  style={{ display: 'none' }}
                />
              </PixelCard>
            </motion.div>
          )}

          {stage === STATE.PREVIEW && imageSrc && (
            <motion.div
              key="preview"
              initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -16 }}
              transition={{ duration: 0.4 }}
            >
              <div style={{
                borderRadius: '12px', overflow: 'hidden',
                border: '1px solid rgba(227,220,210,0.1)',
                marginBottom: '28px',
              }}>
                <img
                  src={imageSrc}
                  alt="Preview"
                  style={{ width: '100%', maxHeight: '400px', objectFit: 'contain', display: 'block', backgroundColor: 'rgba(16,12,13,0.6)' }}
                />
              </div>
              {errorMsg && (
                <div style={{
                  marginBottom: '16px', padding: '12px 20px',
                  backgroundColor: 'rgba(200,60,60,0.1)',
                  border: '1px solid rgba(200,60,60,0.3)',
                  borderRadius: '10px', color: 'rgba(255,150,150,0.9)',
                  fontSize: '13px', textAlign: 'center',
                }}>
                  {errorMsg}
                </div>
              )}
              <div style={{ display: 'flex', gap: '14px', justifyContent: 'center' }}>
                <motion.button
                  whileHover={{ scale: 1.04, boxShadow: '0 12px 32px rgba(212,149,107,0.38)' }}
                  whileTap={{ scale: 0.97 }}
                  transition={{ type: 'spring', stiffness: 380, damping: 18 }}
                  onClick={startProtection}
                  style={{
                    backgroundColor: 'var(--copper)', color: 'var(--black)',
                    border: 'none', borderRadius: '100px',
                    padding: '14px 36px', fontSize: '14px', fontWeight: '600',
                    cursor: 'pointer', letterSpacing: '0.5px',
                    boxShadow: '0 6px 24px rgba(212,149,107,0.28)',
                  }}
                >
                  Protect Now
                </motion.button>
                <button
                  onClick={reset}
                  style={{
                    backgroundColor: 'transparent', color: 'rgba(227,220,210,0.5)',
                    border: '1px solid rgba(227,220,210,0.14)',
                    borderRadius: '100px', padding: '14px 28px',
                    fontSize: '13px', cursor: 'pointer',
                  }}
                >
                  Choose another
                </button>
              </div>
            </motion.div>
          )}

          {stage === STATE.PROCESSING && (
            <motion.div
              key="processing"
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              transition={{ duration: 0.4 }}
              style={{ textAlign: 'center', padding: '80px 40px' }}
            >
              <div style={{
                width: '64px', height: '64px', borderRadius: '50%',
                border: '2px solid rgba(227,220,210,0.1)',
                borderTopColor: 'var(--copper)',
                animation: 'spin 0.9s linear infinite',
                margin: '0 auto 32px',
              }} />

              <div style={{
                fontFamily: 'var(--font-serif)', fontSize: '22px',
                color: 'var(--cream)', marginBottom: '20px',
              }}>
                Running U-Net protection...
              </div>

              <div style={{
                width: '100%', maxWidth: '400px', margin: '0 auto',
                height: '2px', backgroundColor: 'rgba(227,220,210,0.1)',
                borderRadius: '2px', overflow: 'hidden',
              }}>
                <motion.div
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.15, ease: 'easeOut' }}
                  style={{ height: '100%', backgroundColor: 'var(--copper)', borderRadius: '2px' }}
                />
              </div>
              <div style={{
                marginTop: '12px', fontSize: '12px',
                color: 'rgba(227,220,210,0.35)', letterSpacing: '1px',
              }}>
                {Math.round(progress)}% complete
              </div>

              <div style={{
                display: 'flex', justifyContent: 'center', gap: '32px',
                marginTop: '36px',
              }}>
                {['Masking', 'U-Net pass', 'Finalising'].map((step, i) => (
                  <div key={step} style={{
                    fontSize: '11px', letterSpacing: '1.5px',
                    textTransform: 'uppercase',
                    color: progress > (i + 1) * 30
                      ? 'var(--copper)'
                      : 'rgba(227,220,210,0.25)',
                    transition: 'color 0.4s ease',
                  }}>
                    {step}
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {stage === STATE.DONE && originalResized && protectedSrc && (
            <motion.div
              key="done"
              initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              transition={{ duration: 0.5 }}
            >
              <div style={{ marginBottom: '24px' }}>
                {/* Both sides are 512×512 so the slider aligns perfectly */}
                <BeforeAfterSlider original={originalResized} protected={protectedSrc} />
              </div>

              {unetMissing && (
                <div style={{
                  marginBottom: '16px', padding: '10px 18px',
                  backgroundColor: 'rgba(212,149,107,0.08)',
                  border: '1px solid rgba(212,149,107,0.25)',
                  borderRadius: '10px',
                  fontSize: '12px', color: 'rgba(212,149,107,0.8)',
                  textAlign: 'center',
                }}>
                  PGD fallback used — cloak_unet.pth is missing. Processing took longer than usual.
                </div>
              )}

              <div style={{
                display: 'flex', gap: '14px', justifyContent: 'center', flexWrap: 'wrap',
              }}>
                <motion.button
                  whileHover={{ scale: 1.04, boxShadow: '0 12px 32px rgba(212,149,107,0.38)' }}
                  whileTap={{ scale: 0.97 }}
                  transition={{ type: 'spring', stiffness: 380, damping: 18 }}
                  onClick={download}
                  style={{
                    backgroundColor: 'var(--copper)', color: 'var(--black)',
                    border: 'none', borderRadius: '100px',
                    padding: '14px 36px', fontSize: '14px', fontWeight: '600',
                    cursor: 'pointer', letterSpacing: '0.5px',
                    boxShadow: '0 6px 24px rgba(212,149,107,0.28)',
                  }}
                >
                  Download Protected Image
                </motion.button>
                <button
                  onClick={reset}
                  style={{
                    backgroundColor: 'transparent', color: 'rgba(227,220,210,0.5)',
                    border: '1px solid rgba(227,220,210,0.14)',
                    borderRadius: '100px', padding: '14px 28px',
                    fontSize: '13px', cursor: 'pointer',
                  }}
                >
                  Protect another
                </button>
              </div>

              <div style={{
                textAlign: 'center', marginTop: '20px',
                fontSize: '11px', color: 'rgba(227,220,210,0.28)', letterSpacing: '0.5px',
              }}>
                Drag the slider to compare. Output is 512×512 PNG.
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  )
}
