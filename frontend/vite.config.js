import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'

// Cible du backend pour le proxy de développement.
//  - En Docker : le service est joignable via le hostname `backend` (réseau compose)
//    -> on passe VITE_API_TARGET=http://backend:8000 (cf. docker-compose.yml).
//  - En local  : par défaut http://localhost:8000 (uvicorn lancé sur la machine hôte).
// Le front fait des appels RELATIFS (`fetch('/api/stats')`), Vite proxifie vers la
// cible : plus d'URL en dur, plus de problème de CORS (même origine côté navigateur).
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.VITE_API_TARGET || 'http://localhost:8000'

  // Chemins servis par le backend (cf. routes FastAPI). `ws: true` pour le WebSocket ESP.
  const proxy = {
    '/api': { target, changeOrigin: true },
    '/scan': { target, changeOrigin: true },
    '/predict': { target, changeOrigin: true },
    '/health': { target, changeOrigin: true },
    '/metrics': { target, changeOrigin: true },
    '/ws': { target, changeOrigin: true, ws: true },
  }

  return {
    plugins: [vue()],
    server: {
      host: '0.0.0.0',
      port: 8080,
      proxy,
    },
  }
})
