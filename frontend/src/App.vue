<template>
  <div class="dashboard">
    <header>
      <h1>🛰️ Smart Campus</h1>
      <nav class="tabs">
        <button :class="{ active: tab === 'live' }" @click="tab = 'live'">📡 Monitoring Live</button>
        <button :class="{ active: tab === 'ia' }" @click="tab = 'ia'">🤖 Prévisions IA</button>
      </nav>
      <span class="refresh" v-if="tab === 'live'">⟳ {{ intervalMs / 1000 }}s
        <span v-if="error" class="err">— API injoignable</span>
      </span>
    </header>

    <!-- ===================== ONGLET MONITORING LIVE ===================== -->
    <template v-if="tab === 'live'">
      <section class="stats">
        <div class="stat ok"><div class="num">{{ stats.total_success }}</div><div class="lbl">Accès autorisés</div></div>
        <div class="stat ko"><div class="num">{{ stats.total_refus }}</div><div class="lbl">Accès refusés</div></div>
        <div class="stat"><div class="num">{{ stats.total_scans }}</div><div class="lbl">Total scans</div></div>
        <div class="stat"><div class="num">{{ stats.readers_online }}/{{ stats.readers_total }}</div><div class="lbl">ESP32 en ligne</div></div>
      </section>

      <div class="grid">
        <section class="panel">
          <h2>📡 Lecteurs ESP32</h2>
          <div v-if="!readers.length" class="empty">Aucun lecteur connu</div>
          <div v-for="r in readers" :key="r.mac_address" class="reader">
            <div class="reader-head">
              <span class="dot" :class="r.statut"></span>
              <strong>{{ r.nom || 'ESP32' }}</strong>
              <span class="badge-status" :class="r.statut">{{ r.statut }}</span>
            </div>
            <div class="reader-meta">
              <div><span>Salle</span> {{ r.salle || '—' }}</div>
              <div><span>MAC</span> <code>{{ r.mac_address }}</code></div>
              <div><span>IP</span> {{ r.ip_address || '—' }}</div>
              <div><span>Vu</span> {{ fmt(r.last_seen) }}</div>
            </div>
          </div>
        </section>

        <section class="panel wide">
          <h2>🪪 Historique des badges scannés</h2>
          <table>
            <thead><tr><th>Heure</th><th>RFID UID</th><th>Utilisateur</th><th>Salle</th><th>Type</th><th>Résultat</th></tr></thead>
            <tbody>
              <tr v-if="!events.length"><td colspan="6" class="empty">Aucun scan</td></tr>
              <tr v-for="e in events" :key="e.id" :class="e.resultat ? 'row-ok' : 'row-ko'">
                <td>{{ fmt(e.timestamp) }}</td>
                <td><code>{{ e.rfid_uid }}</code></td>
                <td>{{ e.user || '—' }}</td>
                <td>{{ e.salle || e.mac_address }}</td>
                <td>{{ e.type_evenement }}</td>
                <td><span class="pill" :class="e.resultat ? 'ok' : 'ko'">{{ e.resultat ? '✓ autorisé' : '✗ refusé' }}</span></td>
              </tr>
            </tbody>
          </table>
        </section>
      </div>
    </template>

    <!-- ===================== ONGLET PRÉVISIONS IA ===================== -->
    <template v-else>
      <section class="ia-controls">
        <label>📅 Date de prévision
          <input type="date" v-model="iaDate" @change="loadPredictions" />
        </label>
        <button @click="loadPredictions" :disabled="iaLoading">
          {{ iaLoading ? 'Calcul…' : 'Prévoir l\'affluence' }}
        </button>
        <span class="model-info" v-if="!modelReady">⏳ Modèle IA en cours d'entraînement…</span>
        <span class="model-info ready" v-else>✅ Modèle KNN prêt</span>
      </section>

      <p class="ia-subtitle">
        Nombre de personnes prévu par salle (modèle KNN probabiliste · intervalle p10–p90).
      </p>

      <div v-if="iaError" class="empty">Prédictions indisponibles — le modèle n'est peut-être pas encore entraîné.</div>
      <div class="ia-grid">
        <div v-for="p in predictions" :key="p.room_id" class="ia-card" :class="'lvl-' + (p.level || 'na')">
          <div class="ia-head">
            <strong>{{ p.room_nom }}</strong>
            <span class="kind">{{ p.kind || '—' }}</span>
          </div>
          <div class="ia-predicted">
            {{ p.predicted }}<span class="unit"> pers.</span>
          </div>
          <div class="ia-interval">intervalle <b>{{ p.lower }}–{{ p.upper }}</b> · cap. {{ p.capacity }}</div>

          <div class="fill-bar">
            <div class="fill" :style="{ width: Math.min(100, (p.taux_remplissage || 0) * 100) + '%' }"></div>
          </div>
          <div class="ia-fill-lbl">remplissage {{ Math.round((p.taux_remplissage || 0) * 100) }}%</div>

          <div class="ia-foot">
            <span class="level" :class="p.level">{{ p.level || 'n/a' }}</span>
            <span class="conf">confiance {{ Math.round((p.confidence || 0) * 100) }}%</span>
          </div>
          <div class="proba" v-if="p.level_proba">
            <span v-for="(v, k) in p.level_proba" :key="k" :title="k + ': ' + Math.round(v*100) + '%'">
              {{ k[0].toUpperCase() }} {{ Math.round(v * 100) }}%
            </span>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

// Appels RELATIFS : Vite (en dev) ou le reverse-proxy (en prod) redirige vers le
// backend. Plus d'URL en dur -> fonctionne en local comme en Docker, sans CORS.
const intervalMs = 2000

// fetch JSON avec timeout (l'API peut être lente au 1er démarrage : le modèle ML
// s'entraîne en tâche de fond ~1 min). On annule au-delà du délai pour ne pas
// empiler les requêtes du polling.
const fetchJSON = async (url, { timeout = 8000 } = {}) => {
  const ctrl = new AbortController()
  const id = setTimeout(() => ctrl.abort(), timeout)
  try {
    const res = await fetch(url, { signal: ctrl.signal })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return await res.json()
  } finally {
    clearTimeout(id)
  }
}

// --- état commun ---
const tab = ref('live')
const fmt = (iso) => {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleTimeString('fr-FR') + ' ' + d.toLocaleDateString('fr-FR')
}

// --- monitoring live ---
const stats = ref({ total_success: 0, total_refus: 0, total_scans: 0, readers_online: 0, readers_total: 0 })
const readers = ref([])
const events = ref([])
const error = ref(false)
let timer = null

const refresh = async () => {
  try {
    const [s, r, e] = await Promise.all([
      fetchJSON('/api/stats'),
      fetchJSON('/api/readers'),
      fetchJSON('/api/events?limit=50'),
    ])
    stats.value = s; readers.value = r; events.value = e; error.value = false
  } catch { error.value = true }
}

// --- prévisions IA ---
const iaDate = ref(new Date(Date.now() + 86400000).toISOString().slice(0, 10)) // demain
const predictions = ref([])
const iaLoading = ref(false)
const iaError = ref(false)
const modelReady = ref(false)

const checkModel = async () => {
  try {
    const s = await fetchJSON('/api/ml/status')
    modelReady.value = s.model_trained
  } catch { modelReady.value = false }
}

const loadPredictions = async () => {
  iaLoading.value = true; iaError.value = false
  try {
    // L'entraînement initial peut durer ~1 min : on laisse un timeout plus large.
    const data = await fetchJSON(`/api/ml/predict?date=${iaDate.value}`, { timeout: 20000 })
    predictions.value = data.predictions || []
  } catch { iaError.value = true } finally { iaLoading.value = false }
}

onMounted(() => {
  refresh()
  timer = setInterval(refresh, intervalMs)
  checkModel().then(loadPredictions)
})
onUnmounted(() => clearInterval(timer))
</script>

<style>
* { box-sizing: border-box; }
body { font-family: system-ui, Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; }
.dashboard { padding: 1.5rem; max-width: 1400px; margin: 0 auto; }
header { display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap; }
header h1 { margin: 0; font-size: 1.5rem; }
.tabs button { background: #1e293b; color: #94a3b8; border: none; padding: .5rem 1rem; border-radius: 8px; margin-right: .5rem; cursor: pointer; font-size: .95rem; }
.tabs button.active { background: #3b82f6; color: #fff; }
.refresh { margin-left: auto; font-size: .8rem; color: #94a3b8; }
.refresh .err { color: #f87171; }

.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1.25rem 0; }
.stat { background: #1e293b; border-radius: 12px; padding: 1.25rem; text-align: center; border-top: 4px solid #475569; }
.stat.ok { border-top-color: #22c55e; } .stat.ko { border-top-color: #ef4444; }
.stat .num { font-size: 2.2rem; font-weight: 700; } .stat .lbl { color: #94a3b8; font-size: .85rem; margin-top: .25rem; }

.grid { display: grid; grid-template-columns: 340px 1fr; gap: 1rem; align-items: start; }
.panel { background: #1e293b; border-radius: 12px; padding: 1rem 1.25rem; }
.panel h2 { margin: 0 0 1rem; font-size: 1.05rem; }
.empty { color: #64748b; padding: 1rem 0; text-align: center; }
.reader { background: #0f172a; border-radius: 8px; padding: .75rem; margin-bottom: .75rem; }
.reader-head { display: flex; align-items: center; gap: .5rem; }
.dot { width: 10px; height: 10px; border-radius: 50%; background: #64748b; }
.dot.online { background: #22c55e; box-shadow: 0 0 8px #22c55e; } .dot.offline { background: #ef4444; }
.badge-status { margin-left: auto; font-size: .7rem; text-transform: uppercase; padding: 2px 8px; border-radius: 99px; background: #334155; }
.badge-status.online { background: #14532d; color: #86efac; } .badge-status.offline { background: #450a0a; color: #fca5a5; }
.reader-meta { margin-top: .5rem; font-size: .8rem; display: grid; gap: .2rem; }
.reader-meta span { color: #64748b; display: inline-block; width: 42px; } .reader-meta code { color: #93c5fd; }
table { width: 100%; border-collapse: collapse; font-size: .85rem; }
th, td { text-align: left; padding: .55rem .5rem; border-bottom: 1px solid #334155; }
th { color: #94a3b8; } td code { color: #93c5fd; }
.row-ko td { background: rgba(239,68,68,.06); }
.pill { padding: 2px 10px; border-radius: 99px; font-size: .75rem; }
.pill.ok { background: #14532d; color: #86efac; } .pill.ko { background: #450a0a; color: #fca5a5; }

/* --- IA --- */
.ia-controls { display: flex; align-items: center; gap: 1rem; margin: 1.25rem 0 .5rem; flex-wrap: wrap; }
.ia-controls label { font-size: .9rem; color: #cbd5e1; }
.ia-controls input { background: #1e293b; border: 1px solid #334155; color: #e2e8f0; padding: .4rem; border-radius: 6px; margin-left: .5rem; }
.ia-controls button { background: #3b82f6; color: #fff; border: none; padding: .55rem 1.1rem; border-radius: 8px; cursor: pointer; }
.ia-controls button:disabled { opacity: .6; }
.model-info { font-size: .8rem; color: #fbbf24; } .model-info.ready { color: #86efac; }
.ia-subtitle { color: #94a3b8; font-size: .85rem; margin: 0 0 1rem; }

.ia-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 1rem; }
.ia-card { background: #1e293b; border-radius: 12px; padding: 1rem; border-left: 4px solid #475569; }
.ia-card.lvl-faible { border-left-color: #22c55e; }
.ia-card.lvl-moyen { border-left-color: #f59e0b; }
.ia-card.lvl-fort { border-left-color: #ef4444; }
.ia-head { display: flex; justify-content: space-between; align-items: baseline; gap: .5rem; }
.ia-head strong { font-size: .95rem; } .kind { font-size: .7rem; color: #64748b; text-transform: uppercase; }
.ia-predicted { font-size: 2.4rem; font-weight: 700; margin: .3rem 0; }
.ia-predicted .unit { font-size: .9rem; color: #94a3b8; font-weight: 400; }
.ia-interval { font-size: .8rem; color: #94a3b8; } .ia-interval b { color: #cbd5e1; }
.fill-bar { height: 8px; background: #0f172a; border-radius: 99px; margin: .6rem 0 .2rem; overflow: hidden; }
.fill { height: 100%; background: linear-gradient(90deg, #22c55e, #f59e0b, #ef4444); }
.ia-fill-lbl { font-size: .72rem; color: #64748b; }
.ia-foot { display: flex; justify-content: space-between; align-items: center; margin-top: .6rem; }
.level { font-size: .72rem; text-transform: uppercase; padding: 2px 8px; border-radius: 99px; background: #334155; }
.level.faible { background: #14532d; color: #86efac; }
.level.moyen { background: #4a3209; color: #fcd34d; }
.level.fort { background: #450a0a; color: #fca5a5; }
.conf { font-size: .72rem; color: #94a3b8; }
.proba { display: flex; gap: .5rem; margin-top: .5rem; font-size: .68rem; color: #64748b; }

@media (max-width: 900px) { .stats { grid-template-columns: repeat(2, 1fr); } .grid { grid-template-columns: 1fr; } }
</style>
