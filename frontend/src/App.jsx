import Navbar           from './components/Navbar'
import Hero             from './components/Hero'
import Problem          from './components/Problem'
import Solution         from './components/Solution'
import HowItWorks       from './components/HowItWorks'
import TryIt            from './components/TryIt'
import Privacy          from './components/Privacy'
import Footer           from './components/Footer'
import LiquidBackground from './components/LiquidBackground'

export default function App() {
  return (
    <>
      <LiquidBackground />
      <div style={{ position: 'relative', zIndex: 1 }}>
        <Navbar />
        <Hero />
        <Problem />
        <Solution />
        <HowItWorks />
        <TryIt />
        <Privacy />
        <Footer />
      </div>
    </>
  )
}
