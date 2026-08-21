// Chat transcript — inline price-chart artifacts.
//
// A skill publishes a JSON artifact with the `application/vnd.agentos.chart+json`
// mime; the artifact renderer emits a mount placeholder for it (instead of the
// usual download chip) and this module fetches the payload and draws a
// candlestick chart into that placeholder.
//
// Two surfaces, mirroring artifacts.ts / tools.ts:
//   1. Pure helpers (top-level exports) — mime match, payload normalization,
//      theme colors. No DOM, no network, no lightweight-charts import. This is
//      the unit-test surface (chart.test.ts).
//   2. `createChartMounter(deps)` — the imperative mounter the artifact renderer
//      composes. It lazy-imports lightweight-charts so the ~59 KB gz library
//      lands in its own chunk and never loads for a chat that shows no chart.
//
// SECURITY: every payload-derived string reaches the DOM through `textContent`,
// never `innerHTML`. Chart titles carry token names and symbols, which are
// fully attacker-controlled on-chain metadata (see the gmgn-token skill's
// untrusted-data warning) — they are display data, never markup.

// Type-only — erased at compile time, so it does not pull the library into the
// Chat chunk. The runtime import stays dynamic, inside `draw`.
import { t } from '@/i18n'
import '@/i18n/en/chat'

import type { MouseEventParams, Time, UTCTimestamp } from 'lightweight-charts'

import type { Artifact } from './artifacts'

/** The mime a skill publishes to get an inline chart instead of a download chip. */
export const CHART_ARTIFACT_MIME = 'application/vnd.agentos.chart+json'

/** Rendered height of a chart card, in CSS pixels. Mirrors the CSS reservation. */
export const CHART_HEIGHT = 320

/* ── Payload shape ──────────────────────────────────────────────────────── */

/** One OHLC candle. `time` is a Unix timestamp in seconds. */
export interface ChartCandle {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

/** The artifact body a skill writes. Only `candles` is required. */
export interface ChartPayload {
  type: 'candlestick'
  title: string
  subtitle: string
  candles: ChartCandle[]
}

/* ── Pure helpers (unit-tested) ─────────────────────────────────────────── */

/** True when the artifact should render as an inline chart. */
export function isChartArtifact(artifact: Artifact | null | undefined): boolean {
  if (!artifact || !artifact.mime) return false
  return String(artifact.mime).toLowerCase().split(';')[0]?.trim() === CHART_ARTIFACT_MIME
}

function finiteNumber(value: unknown): number | null {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

/**
 * Normalize a raw timestamp to Unix **seconds**.
 *
 * GMGN's kline route returns milliseconds while lightweight-charts expects
 * seconds, and the conversion happens skill-side. This is the backstop for a
 * payload that slipped through in milliseconds: anything past ~5138-11-16
 * (1e11 seconds) is far outside any real candle range, so treat it as ms.
 */
function normalizeTime(value: unknown): number | null {
  const raw = finiteNumber(value)
  if (raw === null || raw <= 0) return null
  const seconds = raw > 1e11 ? Math.floor(raw / 1000) : Math.floor(raw)
  return seconds > 0 ? seconds : null
}

function normalizeCandle(raw: unknown): ChartCandle | null {
  if (!raw || typeof raw !== 'object') return null
  const row = raw as Record<string, unknown>
  const time = normalizeTime(row.time)
  const open = finiteNumber(row.open)
  const high = finiteNumber(row.high)
  const low = finiteNumber(row.low)
  const close = finiteNumber(row.close)
  if (time === null || open === null || high === null || low === null || close === null) return null
  const volume = finiteNumber(row.volume)
  const candle: ChartCandle = { time, open, high, low, close }
  if (volume !== null && volume >= 0) candle.volume = volume
  return candle
}

function textField(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

/**
 * Validate and normalize an artifact body into a drawable payload, or return
 * null when there is nothing to draw.
 *
 * Candles are coerced to numbers, invalid rows dropped, then sorted ascending
 * and de-duplicated by timestamp — lightweight-charts asserts on unordered or
 * repeated times and would otherwise throw during `setData`, taking the whole
 * transcript render down with it. On a duplicate timestamp the later entry
 * wins, matching how an exchange restates the most recent candle.
 */
export function normalizeChartPayload(raw: unknown): ChartPayload | null {
  if (!raw || typeof raw !== 'object') return null
  const body = raw as Record<string, unknown>
  if (!Array.isArray(body.candles)) return null

  const byTime = new Map<number, ChartCandle>()
  for (const entry of body.candles) {
    const candle = normalizeCandle(entry)
    if (candle) byTime.set(candle.time, candle)
  }
  if (byTime.size === 0) return null

  const candles = [...byTime.values()].sort((a, b) => a.time - b.time)
  return {
    type: 'candlestick',
    title: textField(body.title),
    subtitle: textField(body.subtitle),
    candles,
  }
}

/** True when at least one candle carries a volume, so the histogram is worth drawing. */
export function hasVolume(payload: ChartPayload): boolean {
  return payload.candles.some((candle) => typeof candle.volume === 'number')
}

/**
 * Convert a `canvas.toDataURL()` result into a Blob for download.
 *
 * The data-URL is `data:image/png;base64,<payload>`. We strip the prefix,
 * decode the base64 into a Uint8Array, and wrap it in a Blob — no fetch
 * needed, no extra library.
 */
function dataURLtoBlob(dataUrl: string): Blob {
  const comma = dataUrl.indexOf(',')
  if (comma < 0) return new Blob([], { type: 'image/png' })
  const header = dataUrl.slice(0, comma)
  const encoded = dataUrl.slice(comma + 1)
  const mimeMatch = header.match(/:(.*?);/)
  const mime = mimeMatch?.[1] ?? 'image/png'
  const binary = atob(encoded)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return new Blob([bytes], { type: mime })
}

/** One volume-histogram row per candle, colored by that candle's direction. */
export function volumeSeriesData(
  payload: ChartPayload,
  theme: ChartTheme,
): Array<{ time: UTCTimestamp; value: number; color: string }> {
  return payload.candles.map((candle) => ({
    time: candle.time as UTCTimestamp,
    value: candle.volume ?? 0,
    color: candle.close >= candle.open ? theme.volumeUpColor : theme.volumeDownColor,
  }))
}

export type ChartThemeMode = 'dark' | 'light'

/** The palette a chart uses for a theme. Kept pure so the colors are testable. */
export interface ChartTheme {
  background: string
  textColor: string
  gridColor: string
  borderColor: string
  upColor: string
  downColor: string
  volumeUpColor: string
  volumeDownColor: string
}

const DARK_THEME: ChartTheme = {
  background: 'transparent',
  textColor: '#8b949e',
  gridColor: 'rgba(139, 148, 158, 0.12)',
  borderColor: 'rgba(139, 148, 158, 0.25)',
  upColor: '#26a69a',
  downColor: '#ef5350',
  volumeUpColor: 'rgba(38, 166, 154, 0.4)',
  volumeDownColor: 'rgba(239, 83, 80, 0.4)',
}

const LIGHT_THEME: ChartTheme = {
  background: 'transparent',
  textColor: '#57606a',
  gridColor: 'rgba(87, 96, 106, 0.12)',
  borderColor: 'rgba(87, 96, 106, 0.22)',
  upColor: '#0f9d81',
  downColor: '#e03131',
  volumeUpColor: 'rgba(15, 157, 129, 0.35)',
  volumeDownColor: 'rgba(224, 49, 49, 0.35)',
}

/** The palette for a theme mode. */
export function chartTheme(mode: ChartThemeMode): ChartTheme {
  return mode === 'dark' ? DARK_THEME : LIGHT_THEME
}

/**
 * Decimal places to show for a price scale, derived from the smallest close.
 *
 * Meme tokens trade at prices like 0.0000000123, which the library's default
 * 2-decimal formatter would flatten to "0.00" on every axis label.
 */
export function priceDecimals(payload: ChartPayload): number {
  let smallest = Infinity
  for (const candle of payload.candles) {
    const value = Math.abs(candle.close)
    if (value > 0 && value < smallest) smallest = value
  }
  if (!Number.isFinite(smallest)) return 2
  if (smallest >= 1) return 2
  // One extra digit past the leading zeros keeps small moves visible.
  const leadingZeros = Math.floor(-Math.log10(smallest))
  return Math.min(12, leadingZeros + 4)
}

/* ── Crosshair readout ──────────────────────────────────────────────────── */

/** One candle rendered for the readout strip, every field already a string. */
export interface CandleReadout {
  time: string
  open: string
  high: string
  low: string
  close: string
  /** Close against open, signed and suffixed — "+1.27%". */
  change: string
  direction: 'up' | 'down'
  /** Compact USD volume, or '' when the candle carries none. */
  volume: string
}

/** Thousands-grouped short form, so a volume column cannot widen the card. */
export function compactNumber(value: number): string {
  const abs = Math.abs(value)
  if (abs >= 1e9) return `${(value / 1e9).toFixed(1)}B`
  if (abs >= 1e6) return `${(value / 1e6).toFixed(1)}M`
  if (abs >= 1e3) return `${(value / 1e3).toFixed(1)}K`
  return `${Math.round(value)}`
}

/**
 * Format a candle timestamp in UTC.
 *
 * lightweight-charts plots UTC unless a timezone is configured, so the readout
 * has to agree with the axis underneath it rather than with the reader's clock.
 */
export function formatCandleTime(seconds: number): string {
  const at = new Date(seconds * 1000)
  const pad = (value: number): string => String(value).padStart(2, '0')
  const date = `${at.getUTCFullYear()}-${pad(at.getUTCMonth() + 1)}-${pad(at.getUTCDate())}`
  return `${date} ${pad(at.getUTCHours())}:${pad(at.getUTCMinutes())} UTC`
}

/**
 * Everything the readout shows for one candle.
 *
 * The percentage is close against open — the move *within* the candle, which is
 * what a candle body draws — not against the previous close.
 */
export function candleReadout(candle: ChartCandle, decimals: number): CandleReadout {
  const price = (value: number): string => value.toFixed(decimals)
  const delta = candle.open === 0 ? 0 : ((candle.close - candle.open) / candle.open) * 100
  const rounded = Number(delta.toFixed(2))
  return {
    time: formatCandleTime(candle.time),
    open: price(candle.open),
    high: price(candle.high),
    low: price(candle.low),
    close: price(candle.close),
    change: `${rounded >= 0 ? '+' : ''}${rounded.toFixed(2)}%`,
    // A flat candle draws as an up candle (see volumeSeriesData); match it.
    direction: candle.close >= candle.open ? 'up' : 'down',
    volume: typeof candle.volume === 'number' ? compactNumber(candle.volume) : '',
  }
}

/* ── Injected mounter dependencies ──────────────────────────────────────── */

/**
 * Load lightweight-charts once, however many charts a transcript holds.
 *
 * The import stays dynamic so the library keeps its own chunk and never loads
 * for a chat with no charts. Caching the promise means several charts arriving
 * together share one resolution instead of each racing its own.
 */
let libraryPromise: Promise<typeof import('lightweight-charts')> | null = null

function loadChartLibrary(): Promise<typeof import('lightweight-charts')> {
  libraryPromise ??= import('lightweight-charts')
  return libraryPromise
}

/**
 * Build the readout strip's cells once and return an updater for them.
 *
 * The cells are created here rather than in the placeholder markup so the
 * strip stays empty until a chart actually draws, and every value lands
 * through `textContent`.
 */
function mountReadout(host: HTMLElement, decimals: number): ((candle: ChartCandle) => void) | null {
  const strip = host.querySelector<HTMLElement>('.msg-artifact-chart__readout')
  if (!strip) return null
  strip.replaceChildren()

  const values = new Map<string, HTMLElement>()
  const cells = new Map<string, HTMLElement>()
  const addCell = (key: string, label: string): void => {
    const cell = document.createElement('span')
    cell.className = 'msg-artifact-chart__readout-cell'
    if (label) {
      const tag = document.createElement('span')
      tag.className = 'msg-artifact-chart__readout-label'
      tag.textContent = label
      cell.appendChild(tag)
    }
    const value = document.createElement('span')
    value.className = 'msg-artifact-chart__readout-value'
    cell.appendChild(value)
    strip.appendChild(cell)
    values.set(key, value)
    cells.set(key, cell)
  }

  addCell('time', '')
  addCell('open', 'O')
  addCell('high', 'H')
  addCell('low', 'L')
  addCell('close', 'C')
  addCell('change', '')
  addCell('volume', 'Vol')

  return (candle: ChartCandle): void => {
    const readout = candleReadout(candle, decimals)
    values.get('time')!.textContent = readout.time
    values.get('open')!.textContent = readout.open
    values.get('high')!.textContent = readout.high
    values.get('low')!.textContent = readout.low
    values.get('close')!.textContent = readout.close
    values.get('change')!.textContent = readout.change
    values.get('volume')!.textContent = readout.volume
    // Colour the move the same way the candle body is coloured.
    cells.get('change')!.dataset.direction = readout.direction
    cells.get('volume')!.hidden = readout.volume === ''
  }
}

/** A drawn chart, held by its host element so the mounter can sweep it later. */
interface LiveChart {
  dispose: () => void
  applyTheme: (mode: ChartThemeMode) => void
}

export interface ChartMounterDeps {
  /** Fetch a chart artifact body from its (authenticated) URL. */
  fetchPayload: (url: string) => Promise<unknown>
  /** The active theme mode, read at mount time and on every theme change. */
  getTheme: () => ChartThemeMode
  /** chat.js `_chatDiag` — the diagnostics ring. Default: no-op. */
  diag?: (event: string, detail: Record<string, unknown>) => void
}

/* ── Factory ────────────────────────────────────────────────────────────── */

/**
 * Create the chart mounter bound to the transcript's fetch + theme surface.
 *
 * `mountCharts(root)` is idempotent: it only picks up placeholders that have
 * not been mounted yet, so the streaming path may call it after every artifact
 * append and the history path after a bulk replay, without double-drawing.
 */
export function createChartMounter(deps: ChartMounterDeps) {
  const diag = deps.diag ?? ((): void => {})
  /** Hosts this mounter has picked up — payload in flight or chart drawn. */
  const claimed = new Set<HTMLElement>()
  /** Drawn charts, keyed by host so a detached row can be swept (`pruneDetached`). */
  const live = new Map<HTMLElement, LiveChart>()

  function disposeQuietly(entry: LiveChart): void {
    try {
      entry.dispose()
    } catch {
      // A chart the library already tore down is fine to skip.
    }
  }

  /**
   * Dispose every chart whose host has left the document.
   *
   * The transcript rebuilds its rows wholesale — on a session switch and on
   * "load earlier" — which detaches all existing chart hosts. Each stranded
   * chart otherwise keeps a canvas, a ResizeObserver and a theme callback alive
   * for as long as the Chat route stays mounted.
   */
  function pruneDetached(): void {
    for (const host of [...claimed]) {
      if (host.isConnected) continue
      claimed.delete(host)
      const entry = live.get(host)
      if (!entry) continue
      live.delete(host)
      disposeQuietly(entry)
    }
  }

  function setStatus(host: HTMLElement, message: string): void {
    const status = host.querySelector<HTMLElement>('.msg-artifact-chart__status')
    if (!status) return
    // textContent, never innerHTML — see the security note at the top.
    status.textContent = message
    status.hidden = message === ''
  }

  async function draw(host: HTMLElement, payload: ChartPayload): Promise<void> {
    const canvasHost = host.querySelector<HTMLElement>('.msg-artifact-chart__canvas')
    if (!canvasHost) return

    // Prefer the payload's own title over the artifact filename. Both are
    // attacker-influenced token metadata, so this stays on textContent.
    if (payload.title) {
      const label = host.querySelector<HTMLElement>('.msg-artifact-chart__name')
      if (label) label.textContent = payload.title
    }
    if (payload.subtitle) {
      const label = host.querySelector<HTMLElement>('.msg-artifact-chart__name')
      if (label) label.title = payload.subtitle
    }

    const lib = await loadChartLibrary()
    const theme = chartTheme(deps.getTheme())
    const decimals = priceDecimals(payload)

    const chart = lib.createChart(canvasHost, {
      height: CHART_HEIGHT,
      width: canvasHost.clientWidth || 600,
      layout: {
        background: { type: lib.ColorType.Solid, color: theme.background },
        textColor: theme.textColor,
        // TradingView asks that the attribution logo stay visible on the
        // Apache-2.0 build; leaving the default in place keeps us in the clear.
        attributionLogo: true,
      },
      grid: {
        vertLines: { color: theme.gridColor },
        horzLines: { color: theme.gridColor },
      },
      rightPriceScale: { borderColor: theme.borderColor },
      timeScale: { borderColor: theme.borderColor, timeVisible: true, secondsVisible: false },
      // No chart-level `localization.priceFormatter` here on purpose: it wins
      // over every series' own priceFormat, so a price precision wide enough
      // for meme tokens would also print volume as "2739.0000000000". Each
      // series formats its own scale instead.
    })

    const candleSeries = chart.addSeries(lib.CandlestickSeries, {
      upColor: theme.upColor,
      downColor: theme.downColor,
      borderUpColor: theme.upColor,
      borderDownColor: theme.downColor,
      wickUpColor: theme.upColor,
      wickDownColor: theme.downColor,
      priceFormat: { type: 'price', precision: decimals, minMove: 10 ** -decimals },
    })
    // `Time` is a branded number in lightweight-charts; our normalized Unix
    // seconds satisfy the UTCTimestamp contract.
    candleSeries.setData(
      payload.candles.map((candle) => ({ ...candle, time: candle.time as UTCTimestamp })),
    )

    let volumeSeries: ReturnType<typeof chart.addSeries> | null = null
    if (hasVolume(payload)) {
      volumeSeries = chart.addSeries(lib.HistogramSeries, {
        priceScaleId: 'volume',
        priceFormat: { type: 'volume' },
      })
      chart.priceScale('volume').applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      })
      volumeSeries.setData(volumeSeriesData(payload, theme))
    }

    chart.timeScale().fitContent()

    let crosshairUnsubscribe: (() => void) | null = null
    // lightweight-charts ships no tooltip: a crosshair alone cannot say what a
    // candle opened at or how far it moved, so the readout strip does. It rests
    // on the newest candle and follows the cursor while it is over the chart.
    const updateReadout = mountReadout(host, decimals)
    const byTime = new Map(payload.candles.map((entry) => [entry.time, entry]))
    const newest = payload.candles[payload.candles.length - 1]
    if (updateReadout && newest) {
      updateReadout(newest)
      const onCrosshairMove = (param: MouseEventParams<Time>): void => {
        const at = typeof param.time === 'number' ? param.time : null
        updateReadout((at === null ? undefined : byTime.get(at)) ?? newest)
      }
      chart.subscribeCrosshairMove(onCrosshairMove)
      crosshairUnsubscribe = (): void => chart.unsubscribeCrosshairMove(onCrosshairMove)
    }

    // Re-theme in place rather than tearing the chart down, so a theme toggle
    // does not reset the user's pan/zoom.
    const applyTheme = (mode: ChartThemeMode): void => {
      const next = chartTheme(mode)
      chart.applyOptions({
        layout: {
          background: { type: lib.ColorType.Solid, color: next.background },
          textColor: next.textColor,
        },
        grid: {
          vertLines: { color: next.gridColor },
          horzLines: { color: next.gridColor },
        },
        rightPriceScale: { borderColor: next.borderColor },
        timeScale: { borderColor: next.borderColor },
      })
      candleSeries.applyOptions({
        upColor: next.upColor,
        downColor: next.downColor,
        borderUpColor: next.upColor,
        borderDownColor: next.downColor,
        wickUpColor: next.upColor,
        wickDownColor: next.downColor,
      })
      volumeSeries?.setData(volumeSeriesData(payload, next))
    }

    // The transcript column resizes with the window and the sidebar; the chart
    // is canvas-based so it cannot reflow on its own.
    let observer: ResizeObserver | null = null
    if (typeof ResizeObserver === 'function') {
      observer = new ResizeObserver(() => {
        const width = canvasHost.clientWidth
        if (width > 0) chart.applyOptions({ width })
      })
      observer.observe(canvasHost)
    }

    const entry: LiveChart = {
      applyTheme,
      dispose: () => {
        // Unsubscribe before remove(): the chart tears its internals down and
        // will not accept the call afterwards.
        crosshairUnsubscribe?.()
        observer?.disconnect()
        chart.remove()
      },
    }
    // The row can be rebuilt while the payload and the library are in flight —
    // a chart drawn into a detached host has nobody to show it.
    if (!host.isConnected) {
      claimed.delete(host)
      disposeQuietly(entry)
      return
    }
    live.set(host, entry)
    setStatus(host, '')

    // Export chart as PNG image instead of downloading the raw JSON artifact.
    // The download <a> in artifacts.ts points at the artifact URL (JSON); we
    // intercept the click here, capture the canvas, and trigger a PNG download.
    const downloadLink = host.querySelector<HTMLAnchorElement>('.msg-artifact-chart__download')
    if (downloadLink) {
      downloadLink.addEventListener('click', (event: MouseEvent) => {
        event.preventDefault()
        const canvasEl = canvasHost.querySelector('canvas')
        if (!canvasEl) return
        try {
          const dataUrl = canvasEl.toDataURL('image/png')
          const blob = dataURLtoBlob(dataUrl)
          const blobUrl = URL.createObjectURL(blob)
          const link = document.createElement('a')
          link.href = blobUrl
          // Derive filename from the chart title or fall back to "chart".
          const title = host.querySelector<HTMLElement>('.msg-artifact-chart__name')?.textContent
          link.download = `${title || 'chart'}.png`
          document.body.appendChild(link)
          link.click()
          document.body.removeChild(link)
          URL.revokeObjectURL(blobUrl)
        } catch {
          // Canvas tainted or toDataURL unavailable — fall back to the
          // original JSON download by doing nothing (the default <a> action
          // would have handled it).
        }
      })
    }
  }

  async function mountOne(host: HTMLElement): Promise<void> {
    const url = host.dataset.chartSrc || ''
    if (!url) {
      setStatus(host, t('chat.chartUnavailable'))
      return
    }
    try {
      diag('chart.mount.start', { url })
      const raw = await deps.fetchPayload(url)
      const payload = normalizeChartPayload(raw)
      if (!payload) {
        setStatus(host, t('chat.chartUnreadable'))
        diag('chart.mount.empty', { url })
        return
      }
      await draw(host, payload)
      diag('chart.mount.done', { url, candles: payload.candles.length })
    } catch (error) {
      setStatus(host, t('chat.chartFailed'))
      diag('chart.mount.error', { url, error: String(error) })
    }
  }

  /** Mount every not-yet-mounted chart placeholder inside `root`. */
  function mountCharts(root: HTMLElement | null | undefined): void {
    if (!root) return
    // Any row rebuild that adds a chart also detached the previous ones.
    pruneDetached()
    const hosts = root.querySelectorAll<HTMLElement>('[data-chart-src]')
    hosts.forEach((host) => {
      if (claimed.has(host)) return
      claimed.add(host)
      void mountOne(host)
    })
  }

  /** Push a theme change into every live chart. */
  function applyTheme(mode: ChartThemeMode): void {
    pruneDetached()
    live.forEach((entry) => {
      try {
        entry.applyTheme(mode)
      } catch {
        // A chart removed mid-toggle must not break the remaining ones.
      }
    })
  }

  /** Tear down every live chart (route unmount). */
  function destroyAll(): void {
    live.forEach(disposeQuietly)
    live.clear()
    claimed.clear()
  }

  return { mountCharts, applyTheme, destroyAll, pruneDetached }
}

export type ChartMounter = ReturnType<typeof createChartMounter>
