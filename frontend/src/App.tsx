import { useEffect } from 'react'
import ReactGA from 'react-ga4'
import Chat from './components/Chat'

function App() {
  useEffect(() => {
    // Initialize Google Analytics (optional - only if measurement ID is set)
    const GA_MEASUREMENT_ID = import.meta.env.VITE_GA_MEASUREMENT_ID
    if (GA_MEASUREMENT_ID) {
      ReactGA.initialize(GA_MEASUREMENT_ID)
      console.log('Google Analytics initialized')
    }
  }, [])

  return (
    <div className="min-h-screen bg-gray-50">
      <Chat />
    </div>
  )
}

export default App
