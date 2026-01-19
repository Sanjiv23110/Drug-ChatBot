import { useEffect } from 'react'
import ReactGA from 'react-ga4'
import Chat from './components/Chat'
import { ThemeProvider } from './context/ThemeContext'

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
    <ThemeProvider>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-950 transition-colors duration-200">
        <Chat />
      </div>
    </ThemeProvider>
  )
}

export default App
