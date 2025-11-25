import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

st.set_page_config(page_title="CoralMD Dashboard", layout="wide")

# === Tailwind-styled HTML layout ===
components.html("""
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">

<div class="min-h-screen bg-gradient-to-br from-blue-900 via-indigo-900 to-cyan-700 text-white font-sans">

  <!-- Hero section -->
  <header class="text-center py-10">
    <img src="https://upload.wikimedia.org/wikipedia/commons/0/0a/Coral_Reef.jpg" 
         alt="Coral Reef" class="mx-auto rounded-3xl shadow-lg w-2/3 md:w-1/2 mb-6">
    <h1 class="text-5xl font-bold mb-2">🌊 CoralMD — Precision Medicine Platform</h1>
    <p class="text-lg text-gray-200">Integrating genomics, wearables, and clinical data to illuminate health risks early.</p>
  </header>

  <!-- Metric Cards -->
  <section class="flex flex-wrap justify-center gap-6 px-8 mb-10">
    <div class="bg-white bg-opacity-10 backdrop-blur-lg rounded-xl p-6 w-60 text-center shadow-lg hover:bg-opacity-20 transition">
      <h2 class="text-2xl font-bold text-cyan-300 mb-2">Genomic Risk</h2>
      <p>🧬 View variant-based predispositions</p>
    </div>
    <div class="bg-white bg-opacity-10 backdrop-blur-lg rounded-xl p-6 w-60 text-center shadow-lg hover:bg-opacity-20 transition">
      <h2 class="text-2xl font-bold text-teal-300 mb-2">Wearable Analytics</h2>
      <p>⌚ Track HR, sleep, glucose, and activity trends</p>
    </div>
    <div class="bg-white bg-opacity-10 backdrop-blur-lg rounded-xl p-6 w-60 text-center shadow-lg hover:bg-opacity-20 transition">
      <h2 class="text-2xl font-bold text-blue-300 mb-2">EHR Dashboard</h2>
      <p>🏥 Aggregate lab and clinical insights</p>
    </div>
    <div class="bg-white bg-opacity-10 backdrop-blur-lg rounded-xl p-6 w-60 text-center shadow-lg hover:bg-opacity-20 transition">
      <h2 class="text-2xl font-bold text-indigo-300 mb-2">Ethical Interface</h2>
      <p>⚖️ Transparency and interpretability metrics</p>
    </div>
  </section>

  <!-- Footer -->
  <footer class="text-center text-sm text-gray-400 py-8">
    <p>© 2025 CoralMD | Built for Pomona College Data Science Capstone</p>
  </footer>

</div>
""", height=900)
