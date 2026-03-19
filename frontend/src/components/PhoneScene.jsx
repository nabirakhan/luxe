import { Suspense, useRef, useEffect } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { useGLTF, Environment, ContactShadows, OrbitControls } from '@react-three/drei'
import * as THREE from 'three'

function IPhoneModel({ spinTrigger, onSpinComplete }) {
  const groupRef = useRef()
  // Load the GLB without any modifications
  const { scene } = useGLTF('/iphone_17_pro_max.glb')
  const spin = useRef({ active: false, startT: null })

  useEffect(() => {
    if (spinTrigger > 0) {
      spin.current = { active: true, startT: null }
    }
  }, [spinTrigger])

  useFrame(({ clock }) => {
    if (!groupRef.current) return
    const t = clock.getElapsedTime()
    
    // Floating animation (Keeping your exact sample pivots)
    groupRef.current.position.y = Math.sin(t * 0.6) * 0.05

    if (spin.current.active) {
      if (spin.current.startT === null) spin.current.startT = t
      const elapsed = t - spin.current.startT
      const DURATION = 1.4
      const p = Math.min(elapsed / DURATION, 1)
      const eased = p < 0.5 ? 2 * p * p : -1 + (4 - 2 * p) * p
      groupRef.current.rotation.y = eased * Math.PI * 2
      
      if (p >= 1) {
        spin.current.active = false
        groupRef.current.rotation.y = 0
        onSpinComplete?.()
      }
    }
  })

  return (
    <group ref={groupRef} position={[0.6, 0, 0]}>
      {/* Renders the original scene exactly as it is in the file */}
      <primitive object={scene} scale={1.7} rotation={[0.2, 0.35, -0.75]} />
    </group>
  )
}

export default function PhoneScene({ spinTrigger, onSpinComplete }) {
  return (
    <div style={{ width: '100%', height: '100vh' }}>
      <Canvas
        shadows
        dpr={[1, 2]}
        camera={{ position: [0, 0, 7], fov: 26 }}
        gl={{ antialias: true, alpha: true }}
      >
        <Suspense fallback={null}>
          {/* Environment provides the reflections for the original materials */}
          <Environment preset="city" />
          <ambientLight intensity={1} />
          <directionalLight position={[5, 10, 5]} intensity={1.5} />
          
          <IPhoneModel spinTrigger={spinTrigger} onSpinComplete={onSpinComplete} />

          <ContactShadows position={[0, -1.4, 0]} opacity={0.25} scale={8} blur={2.5} />
        </Suspense>

        <OrbitControls 
          target={[0.6, 0.4, 0]} 
          enableZoom={false} 
          enablePan={false} 
          makeDefault 
        />
      </Canvas>
    </div>
  )
}

useGLTF.preload('/iphone_17_pro_max.glb')