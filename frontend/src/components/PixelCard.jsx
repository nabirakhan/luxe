import { useEffect, useRef } from 'react'
import './PixelCard.css'

class Pixel {
  constructor(canvas, context, x, y, color, speed, delay) {
    this.width = canvas.width
    this.height = canvas.height
    this.ctx = context
    this.x = x
    this.y = y
    this.color = color
    this.speed = this.getRandomValue(0.1, 0.9) * speed
    this.size = 0
    this.sizeStep = Math.random() * 0.4
    this.minSize = 0.5
    this.maxSizeInteger = 2
    this.maxSize = this.getRandomValue(this.minSize, this.maxSizeInteger)
    this.delay = delay
    this.counter = 0
    this.counterStep = Math.random() * 4 + (this.width + this.height) * 0.01
    this.isIdle = false
    this.isReverse = false
    this.isShimmer = false
  }

  getRandomValue(min, max) { return Math.random() * (max - min) + min }

  draw() {
    const centerOffset = this.maxSizeInteger * 0.5 - this.size * 0.5
    this.ctx.fillStyle = this.color
    this.ctx.fillRect(this.x + centerOffset, this.y + centerOffset, this.size, this.size)
  }

  appear() {
    this.isIdle = false
    if (this.counter <= this.delay) { this.counter += this.counterStep; return }
    if (this.size >= this.maxSize) this.isShimmer = true
    if (this.isShimmer) this.shimmer()
    else this.size += this.sizeStep
    this.draw()
  }

  disappear() {
    this.isShimmer = false
    this.counter = 0
    if (this.size <= 0) { this.isIdle = true; return }
    else this.size -= 0.1
    this.draw()
  }

  shimmer() {
    if (this.size >= this.maxSize) this.isReverse = true
    else if (this.size <= this.minSize) this.isReverse = false
    if (this.isReverse) this.size -= this.speed
    else this.size += this.speed
  }
}

function getEffectiveSpeed(value, reducedMotion) {
  const parsed = parseInt(value, 10)
  if (parsed <= 0 || reducedMotion) return 0
  if (parsed >= 100) return 0.1
  return parsed * 0.001
}

export default function PixelCard({
  gap = 6,
  speed = 60,
  colors = '#CC8B65,#a86840,#E3DCD2',
  className = '',
  style = {},
  children,
  ...rest
}) {
  const containerRef    = useRef(null)
  const canvasRef       = useRef(null)
  const pixelsRef       = useRef([])
  const animationRef    = useRef(null)
  const timePreviousRef = useRef(performance.now())
  const reducedMotion   = useRef(window.matchMedia('(prefers-reduced-motion: reduce)').matches).current

  const initPixels = () => {
    if (!containerRef.current || !canvasRef.current) return
    const rect   = containerRef.current.getBoundingClientRect()
    const width  = Math.floor(rect.width)
    const height = Math.floor(rect.height)
    const ctx    = canvasRef.current.getContext('2d')
    canvasRef.current.width  = width
    canvasRef.current.height = height
    canvasRef.current.style.width  = `${width}px`
    canvasRef.current.style.height = `${height}px`

    const colorsArray = colors.split(',')
    const pxs = []
    for (let x = 0; x < width; x += parseInt(gap, 10)) {
      for (let y = 0; y < height; y += parseInt(gap, 10)) {
        const color    = colorsArray[Math.floor(Math.random() * colorsArray.length)]
        const dx       = x - width / 2
        const dy       = y - height / 2
        const distance = Math.sqrt(dx * dx + dy * dy)
        const delay    = reducedMotion ? 0 : distance
        pxs.push(new Pixel(canvasRef.current, ctx, x, y, color, getEffectiveSpeed(speed, reducedMotion), delay))
      }
    }
    pixelsRef.current = pxs
  }

  const doAnimate = fnName => {
    animationRef.current = requestAnimationFrame(() => doAnimate(fnName))
    const timeNow     = performance.now()
    const timePassed  = timeNow - timePreviousRef.current
    if (timePassed < 1000 / 60) return
    timePreviousRef.current = timeNow - (timePassed % (1000 / 60))
    const ctx = canvasRef.current?.getContext('2d')
    if (!ctx || !canvasRef.current) return
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    let allIdle = true
    for (const pixel of pixelsRef.current) {
      pixel[fnName]()
      if (!pixel.isIdle) allIdle = false
    }
    if (allIdle) cancelAnimationFrame(animationRef.current)
  }

  const handleAnimation = name => {
    cancelAnimationFrame(animationRef.current)
    animationRef.current = requestAnimationFrame(() => doAnimate(name))
  }

  useEffect(() => {
    initPixels()
    handleAnimation('appear')
    const observer = new ResizeObserver(() => { initPixels(); handleAnimation('appear') })
    if (containerRef.current) observer.observe(containerRef.current)
    return () => { observer.disconnect(); cancelAnimationFrame(animationRef.current) }
  }, [gap, speed, colors])

  return (
    <div
      ref={containerRef}
      className={`pixel-card ${className}`}
      style={style}
      onMouseEnter={() => handleAnimation('disappear')}
      onMouseLeave={() => handleAnimation('appear')}
      {...rest}
    >
      <canvas className="pixel-canvas" ref={canvasRef} />
      {children}
    </div>
  )
}
