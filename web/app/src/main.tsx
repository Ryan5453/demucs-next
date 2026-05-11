import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Home } from './components/pages/Home'
import { About } from './components/pages/About'
import { Privacy } from './components/pages/Privacy'
import { Benchmark } from './components/pages/Benchmark'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
        <BrowserRouter>
            <Routes>
                <Route element={<Layout />}>
                    <Route path="/" element={<Home />} />
                    <Route path="/about" element={<About />} />
                    <Route path="/privacy" element={<Privacy />} />
                    <Route path="/benchmark" element={<Benchmark />} />
                </Route>
            </Routes>
        </BrowserRouter>
    </React.StrictMode>,
)

