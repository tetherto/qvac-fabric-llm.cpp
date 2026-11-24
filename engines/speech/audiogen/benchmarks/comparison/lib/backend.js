'use strict'

function canonicalBackend (value) {
  const backend = String(value || '').toLowerCase()
  if (backend.includes('cpu-fallback') || backend.includes('fallback')) return 'cpu-fallback'
  if (backend.includes('metal') || backend.includes('mtl')) return 'metal'
  if (backend.includes('vulkan') || backend.includes('vk')) return 'vulkan'
  if (backend.includes('opencl')) return 'opencl'
  if (backend.includes('cuda')) return 'cuda'
  if (backend.includes('cpu') || backend.includes('blas') || backend.includes('accelerate')) return 'cpu'
  return backend
}

function backendMatches (actual, expected) {
  return canonicalBackend(actual) === canonicalBackend(expected)
}

function detectQvacBackend (logText) {
  const text = String(logText || '')
  const ditLine = text.match(/DiT\/VAE on GPU backend:\s+(\S+)/i)
  if (ditLine) return ditLine[1]
  const backendsLine = text.match(/backends:\s+enc=(\S+)\s+lm=(\S+)\s+detok=(\S+)\s+dit\/vae=(\S+)/i)
  if (backendsLine) return backendsLine[4]
  if (/GPU requested but no GPU backend available; using CPU/i.test(text)) return 'cpu-fallback'
  if (/using CPU/i.test(text) && /GPU requested/i.test(text)) return 'cpu-fallback'
  return 'cpu'
}

function detectAcestepBackend (logText) {
  const text = String(logText || '')
  if (/falling back to CPU/i.test(text) || /fallback to cpu/i.test(text)) return 'cpu-fallback'
  if (/ggml_metal|ggml-metal/i.test(text)) return 'Metal'
  const metal = text.match(/\b(MTL\d*|Metal)\b/i)
  const vulkan = text.match(/\bVulkan\d*\b/i)
  const cuda = text.match(/\bCUDA\d*\b/i)
  if (metal) return metal[1]
  if (vulkan) return vulkan[0]
  if (cuda) return cuda[0]
  return 'cpu'
}

function assertBackend (actual, expected, options = {}) {
  const failOnFallback = options.failOnFallback !== false
  const canonicalActual = canonicalBackend(actual)
  if (canonicalBackend(expected) !== 'cpu' && canonicalActual === 'cpu-fallback') {
    const error = new Error(`silent GPU-to-CPU fallback detected (actual=${actual}, expected=${expected})`)
    error.code = 'GPU_FALLBACK'
    throw error
  }
  if (failOnFallback && canonicalBackend(expected) !== 'cpu' && canonicalActual === 'cpu') {
    const error = new Error(`requested ${expected} but runtime backend was CPU (${actual})`)
    error.code = 'BACKEND_MISMATCH'
    throw error
  }
  if (!backendMatches(actual, expected) && canonicalBackend(expected) !== 'cpu') {
    const error = new Error(`backend mismatch: actual=${actual}, expected=${expected}`)
    error.code = 'BACKEND_MISMATCH'
    throw error
  }
  return true
}

module.exports = {
  assertBackend,
  backendMatches,
  canonicalBackend,
  detectAcestepBackend,
  detectQvacBackend
}
