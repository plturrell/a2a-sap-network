/**
 * Service Worker for A2A Portal
 * Provides offline functionality and caching strategies
 */

const CACHE_NAME = "a2a-portal-v1.0.0";
const OFFLINE_URL = "./offline.html";
const API_CACHE_NAME = "a2a-api-cache-v1";
const STATIC_CACHE_NAME = "a2a-static-cache-v1";

// Resources to cache for offline use
const OFFLINE_FALLBACK_RESOURCES = [
    "./",
    "./index.html",
    "./offline.html",
    "./resources/sap-ui-core.js",
    "./css/style.css",
    "./css/design-tokens.css",
    "./css/shared-components.css",
    "./css/loading-states.css",
    "./i18n/i18n.properties",
    "./resources/pwa/icon-192.png",
    "./resources/pwa/icon-512.png"
];

// API endpoints that should be cached
const CACHEABLE_API_PATTERNS = [
    /\/api\/v1\/projects/,
    /\/api\/v1\/agents/,
    /\/api\/v1\/workflows/,
    /\/api\/v1\/analytics/,
    /\/i18n\//
];

// Install event - cache essential resources
self.addEventListener("install", event => {
    // // console.log("[SW] Installing service worker");

    event.waitUntil(
        Promise.all([
            // Cache essential resources
            caches.open(CACHE_NAME)
                .then(cache => {
                    // // console.log("[SW] Pre-caching offline resources");
                    return cache.addAll(OFFLINE_FALLBACK_RESOURCES);
                }),

            // Initialize API cache
            caches.open(API_CACHE_NAME),

            // Initialize static cache
            caches.open(STATIC_CACHE_NAME)
        ])
            .then(() => {
                // // console.log("[SW] Service worker installed successfully");
                return self.skipWaiting();
            })
            .catch(error => {
                // console.error("[SW] Installation failed:", error);
            })
    );
});

// Activate event - clean up old caches
self.addEventListener("activate", event => {
    // // console.log("[SW] Activating service worker");

    event.waitUntil(
        caches.keys()
            .then(cacheNames => {
                return Promise.all(
                    cacheNames.map(cacheName => {
                        if (cacheName !== CACHE_NAME &&
                cacheName !== API_CACHE_NAME &&
                cacheName !== STATIC_CACHE_NAME) {
                            // // console.log("[SW] Deleting old cache:", cacheName);
                            return caches.delete(cacheName);
                        }
                    })
                );
            })
            .then(() => {
                // // console.log("[SW] Service worker activated");
                return self.clients.claim();
            })
    );
});

// Fetch event - handle all network requests
self.addEventListener("fetch", event => {
    const { request } = event;
    const url = new URL(request.url);

    // Handle different types of requests
    if (request.method === "GET") {
    // API requests
        if (url.pathname.startsWith("/api/")) {
            event.respondWith(handleApiRequest(request));
        }
        // Static resources (JS, CSS, images)
        else if (isStaticResource(request)) {
            event.respondWith(handleStaticResource(request));
        }
        // Navigation requests
        else if (request.mode === "navigate") {
            event.respondWith(handleNavigationRequest(request));
        }
        // Other GET requests
        else {
            event.respondWith(handleGenericRequest(request));
        }
    }
    // POST, PUT, DELETE requests
    else {
        event.respondWith(handleMutationRequest(request));
    }
});

/**
 * Handle API requests with caching strategy
 * @param {Request} request - The fetch request
 * @returns {Promise<Response>} Response
 */
function handleApiRequest(request) {
    const url = new URL(request.url);
    const shouldCache = CACHEABLE_API_PATTERNS.some(pattern => pattern.test(url.pathname));

    if (shouldCache) {
        return handleCacheFirstStrategy(request, API_CACHE_NAME);
    }
    return handleNetworkFirstStrategy(request, API_CACHE_NAME);

}

/**
 * Handle static resources (JS, CSS, images) with cache-first strategy
 * @param {Request} request - The fetch request
 * @returns {Promise<Response>} Response
 */
function handleStaticResource(request) {
    return handleCacheFirstStrategy(request, STATIC_CACHE_NAME);
}

/**
 * Handle navigation requests with network-first fallback to offline page
 * @param {Request} request - The fetch request
 * @returns {Promise<Response>} Response
 */
async function handleNavigationRequest(request) {
    try {
    // Try network first
        const response = await fetch(request);

        if (response.ok) {
            // Cache successful navigation responses
            const cache = await caches.open(CACHE_NAME);
            cache.put(request, response.clone());
            return response;
        }
        throw new Error("Network response not ok");

    } catch (error) {
        // // console.log("[SW] Navigation request failed, serving offline page:", error);

        // Try to serve cached version
        const cachedResponse = await caches.match(request);
        if (cachedResponse) {
            return cachedResponse;
        }

        // Fallback to offline page
        const offlineResponse = await caches.match(OFFLINE_URL);
        return offlineResponse || new Response("Offline", {
            status: 503,
            statusText: "Service Unavailable"
        });
    }
}

/**
 * Handle generic requests with network-first strategy
 * @param {Request} request - The fetch request
 * @returns {Promise<Response>} Response
 */
function handleGenericRequest(request) {
    return handleNetworkFirstStrategy(request, CACHE_NAME);
}

/**
 * Handle mutation requests (POST, PUT, DELETE) with background sync
 * @param {Request} request - The fetch request
 * @returns {Promise<Response>} Response
 */
async function handleMutationRequest(request) {
    try {
        const response = await fetch(request);
        return response;
    } catch (error) {
        // // console.log("[SW] Mutation request failed, queuing for background sync:", error);

        // Queue request for background sync
        await queueBackgroundSync(request);

        // Return offline response
        return new Response(
            JSON.stringify({
                error: "Request queued for retry when connection is restored",
                queued: true
            }),
            {
                status: 202,
                statusText: "Accepted",
                headers: { "Content-Type": "application/json" }
            }
        );
    }
}

/**
 * Cache-first strategy: Check cache first, then network
 * @param {Request} request - The fetch request
 * @param {string} cacheName - Cache name to use
 * @returns {Promise<Response>} Response
 */
async function handleCacheFirstStrategy(request, cacheName) {
    try {
    // Check cache first
        const cachedResponse = await caches.match(request);
        if (cachedResponse) {
            return cachedResponse;
        }

        // Fallback to network
        const response = await fetch(request);

        if (response.ok) {
            // Cache the response
            const cache = await caches.open(cacheName);
            cache.put(request, response.clone());
        }

        return response;
    } catch (error) {
        // console.error("[SW] Cache-first strategy failed:", error);
        return new Response("Resource unavailable offline", {
            status: 503,
            statusText: "Service Unavailable"
        });
    }
}

/**
 * Network-first strategy: Try network first, then cache
 * @param {Request} request - The fetch request
 * @param {string} cacheName - Cache name to use
 * @returns {Promise<Response>} Response
 */
async function handleNetworkFirstStrategy(request, cacheName) {
    try {
    // Try network first
        const response = await fetch(request);

        if (response.ok) {
            // Update cache with fresh data
            const cache = await caches.open(cacheName);
            cache.put(request, response.clone());
        }

        return response;
    } catch (error) {
        // // console.log("[SW] Network failed, trying cache:", error);

        // Fallback to cache
        const cachedResponse = await caches.match(request);
        if (cachedResponse) {
            return cachedResponse;
        }

        // No cache available
        return new Response("Resource unavailable", {
            status: 503,
            statusText: "Service Unavailable"
        });
    }
}

/**
 * Check if request is for a static resource
 * @param {Request} request - The fetch request
 * @returns {boolean} True if static resource
 */
function isStaticResource(request) {
    const url = new URL(request.url);
    const extension = url.pathname.split(".").pop();

    return ["js", "css", "png", "jpg", "jpeg", "gif", "svg", "ico", "woff", "woff2"].includes(extension);
}

/**
 * Queue request for background sync
 * @param {Request} request - The fetch request
 */
async function queueBackgroundSync(request) {
    try {
        const requestData = {
            url: request.url,
            method: request.method,
            headers: Object.fromEntries(request.headers.entries()),
            body: request.method !== "GET" ? await request.text() : undefined,
            timestamp: Date.now()
        };

        // Store in IndexedDB for persistence
        const db = await openBackgroundSyncDB();
        const transaction = db.transaction(["sync_queue"], "readwrite");
        const store = transaction.objectStore("sync_queue");
        await store.add(requestData);

        // // console.log("[SW] Request queued for background sync:", requestData.url);
    } catch (error) {
        // console.error("[SW] Failed to queue request for background sync:", error);
    }
}

/**
 * Open IndexedDB for background sync storage
 * @returns {Promise<IDBDatabase>} Database instance
 */
function openBackgroundSyncDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open("a2a-background-sync", 1);

        request.onupgradeneeded = event => {
            const db = event.target.result;
            if (!db.objectStoreNames.contains("sync_queue")) {
                db.createObjectStore("sync_queue", { keyPath: "id", autoIncrement: true });
            }
        };

        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

// Background sync event
self.addEventListener("sync", event => {
    if (event.tag === "background-sync") {
        event.waitUntil(processBackgroundSync());
    }
});

/**
 * Process queued background sync requests
 */
async function processBackgroundSync() {
    try {
        const db = await openBackgroundSyncDB();
        const transaction = db.transaction(["sync_queue"], "readwrite");
        const store = transaction.objectStore("sync_queue");
        const requests = await store.getAll();

        for (const requestData of requests) {
            try {
                const request = new Request(requestData.url, {
                    method: requestData.method,
                    headers: requestData.headers,
                    body: requestData.body
                });

                const response = await fetch(request);

                if (response.ok) {
                    // Remove successful request from queue
                    await store.delete(requestData.id);
                    // // console.log("[SW] Background sync successful:", requestData.url);
                }
            } catch (error) {
                // console.error("[SW] Background sync failed for:", requestData.url, error);
            }
        }
    } catch (error) {
        // console.error("[SW] Background sync processing failed:", error);
    }
}

// Message handling for communication with main thread
self.addEventListener("message", event => {
    if (event.data && event.data.type === "SKIP_WAITING") {
        self.skipWaiting();
    }

    if (event.data && event.data.type === "GET_CACHE_STATUS") {
        getCacheStatus().then(status => {
            event.ports[0].postMessage(status);
        });
    }

    if (event.data && event.data.type === "CLEAR_CACHE") {
        clearAllCaches().then(() => {
            event.ports[0].postMessage({ success: true });
        });
    }
});

/**
 * Get current cache status
 * @returns {Promise<Object>} Cache status information
 */
async function getCacheStatus() {
    try {
        const cacheNames = await caches.keys();
        const cacheInfo = {};

        for (const cacheName of cacheNames) {
            const cache = await caches.open(cacheName);
            const keys = await cache.keys();
            cacheInfo[cacheName] = {
                size: keys.length,
                items: keys.map(request => request.url)
            };
        }

        return { success: true, caches: cacheInfo };
    } catch (error) {
        return { success: false, error: error.message };
    }
}

/**
 * Clear all caches
 * @returns {Promise<void>}
 */
async function clearAllCaches() {
    try {
        const cacheNames = await caches.keys();
        await Promise.all(cacheNames.map(cacheName => caches.delete(cacheName)));
        // // console.log("[SW] All caches cleared");
    } catch (error) {
        // console.error("[SW] Failed to clear caches:", error);
    }
}

// Handle offline/online events
self.addEventListener("online", () => {
    // // console.log("[SW] Back online, processing background sync");
    self.registration.sync.register("background-sync");
});

self.addEventListener("offline", () => {
    // // console.log("[SW] Gone offline");
});