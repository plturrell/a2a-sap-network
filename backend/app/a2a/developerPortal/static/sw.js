/**
 * Service Worker for SAP A2A Developer Portal
 * Enterprise-grade offline capabilities with intelligent caching
 */

const CACHE_VERSION = 'v2.1.0';
const CACHE_NAME = `a2a-portal-${CACHE_VERSION}`;
const DATA_CACHE_NAME = `a2a-data-${CACHE_VERSION}`;
const DYNAMIC_CACHE_NAME = `a2a-dynamic-${CACHE_VERSION}`;

// Cache configuration
const CACHE_CONFIG = {
    // Static assets cache
    static: {
        name: CACHE_NAME,
        strategy: 'cache-first',
        maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days
        maxItems: 100
    },
    // API data cache
    data: {
        name: DATA_CACHE_NAME,
        strategy: 'network-first',
        maxAge: 30 * 60 * 1000, // 30 minutes
        maxItems: 500
    },
    // Dynamic content cache
    dynamic: {
        name: DYNAMIC_CACHE_NAME,
        strategy: 'stale-while-revalidate',
        maxAge: 24 * 60 * 60 * 1000, // 24 hours
        maxItems: 200
    }
};

// Resources to cache on install
const STATIC_CACHE_RESOURCES = [
    '/',
    '/index.html',
    '/manifest.json',
    '/resources/sap-ui-core.js',
    '/resources/sap/ui/core/themes/sap_horizon/library.css',
    '/resources/sap/m/themes/sap_horizon/library.css',
    '/resources/sap/f/themes/sap_horizon/library.css',
    '/static/css/app.css',
    '/static/js/app.js',
    '/static/utils/HelpProvider.js',
    '/static/utils/OfflineManager.js',
    '/static/images/logo.png',
    '/static/images/offline-icon.svg',
    // Offline pages
    '/static/offline.html'
];

// API endpoints that should be cached
const CACHEABLE_API_PATTERNS = [
    /^\/api\/v1\/projects(\?.*)?$/,
    /^\/api\/v1\/agents(\?.*)?$/,
    /^\/api\/v1\/templates(\?.*)?$/,
    /^\/api\/v1\/workflows(\?.*)?$/,
    /^\/api\/v1\/business-partners(\?.*)?$/
];

// API endpoints that should NOT be cached
const UNCACHEABLE_API_PATTERNS = [
    /^\/api\/v1\/projects\/.*\/execute$/,
    /^\/api\/v1\/agents\/.*\/execute$/,
    /^\/api\/v1\/workflows\/.*\/execute$/,
    /^\/api\/v1\/auth\/.*$/,
    /^\/api\/v1\/monitoring\/.*$/
];

/* =========================================================== */
/* Service Worker Event Handlers                              */
/* =========================================================== */

/**
 * Install event - cache static resources
 */
self.addEventListener('install', (event) => {
    console.log('[ServiceWorker] Installing...');
    
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                console.log('[ServiceWorker] Caching static resources');
                return cache.addAll(STATIC_CACHE_RESOURCES);
            })
            .then(() => {
                console.log('[ServiceWorker] Installation complete');
                // Force activation of new service worker
                return self.skipWaiting();
            })
            .catch((error) => {
                console.error('[ServiceWorker] Installation failed:', error);
            })
    );
});

/**
 * Activate event - cleanup old caches
 */
self.addEventListener('activate', (event) => {
    console.log('[ServiceWorker] Activating...');
    
    event.waitUntil(
        caches.keys()
            .then((cacheNames) => {
                return Promise.all(
                    cacheNames.map((cacheName) => {
                        // Delete old caches
                        if (cacheName.startsWith('a2a-portal-') && cacheName !== CACHE_NAME ||
                            cacheName.startsWith('a2a-data-') && cacheName !== DATA_CACHE_NAME ||
                            cacheName.startsWith('a2a-dynamic-') && cacheName !== DYNAMIC_CACHE_NAME) {
                            console.log('[ServiceWorker] Deleting old cache:', cacheName);
                            return caches.delete(cacheName);
                        }
                    })
                );
            })
            .then(() => {
                console.log('[ServiceWorker] Activation complete');
                // Take control of all clients immediately
                return self.clients.claim();
            })
    );
});

/**
 * Fetch event - handle network requests with caching strategies
 */
self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);
    
    // Skip non-HTTP requests
    if (!request.url.startsWith('http')) {
        return;
    }
    
    // Skip requests with cache control headers indicating no cache
    if (request.headers.get('cache-control') === 'no-cache' ||
        request.headers.get('cache-control') === 'no-store') {
        return;
    }
    
    // Handle different types of requests
    if (request.method === 'GET') {
        if (isStaticResource(url)) {
            event.respondWith(handleStaticRequest(request));
        } else if (isApiRequest(url)) {
            event.respondWith(handleApiRequest(request));
        } else if (isNavigationRequest(request)) {
            event.respondWith(handleNavigationRequest(request));
        } else {
            event.respondWith(handleDynamicRequest(request));
        }
    } else if (request.method === 'POST' || request.method === 'PUT' || request.method === 'DELETE') {
        // Handle data modification requests
        event.respondWith(handleDataMutationRequest(request));
    }
});

/**
 * Background sync event
 */
self.addEventListener('sync', (event) => {
    console.log('[ServiceWorker] Background sync triggered:', event.tag);
    
    if (event.tag === 'background-sync') {
        event.waitUntil(performBackgroundSync());
    }
});

/**
 * Push notification event
 */
self.addEventListener('push', (event) => {
    console.log('[ServiceWorker] Push notification received');
    
    const options = {
        body: event.data ? event.data.text() : 'New update available',
        icon: '/static/images/logo.png',
        badge: '/static/images/badge.png',
        tag: 'a2a-notification',
        requireInteraction: false,
        actions: [
            {
                action: 'view',
                title: 'View',
                icon: '/static/images/view-icon.png'
            },
            {
                action: 'dismiss',
                title: 'Dismiss'
            }
        ]
    };
    
    event.waitUntil(
        self.registration.showNotification('A2A Portal', options)
    );
});

/**
 * Notification click event
 */
self.addEventListener('notificationclick', (event) => {
    event.notification.close();
    
    if (event.action === 'view') {
        event.waitUntil(
            clients.openWindow('/')
        );
    }
});

/* =========================================================== */
/* Request Handling Functions                                  */
/* =========================================================== */

/**
 * Handle static resource requests (cache-first strategy)
 */
async function handleStaticRequest(request) {
    try {
        const cachedResponse = await caches.match(request);
        if (cachedResponse) {
            return cachedResponse;
        }
        
        const networkResponse = await fetch(request);
        if (networkResponse.ok) {
            const cache = await caches.open(CACHE_NAME);
            await cache.put(request, networkResponse.clone());
        }
        
        return networkResponse;
    } catch (error) {
        console.error('[ServiceWorker] Static resource fetch failed:', error);
        return new Response('Static resource unavailable', { status: 503 });
    }
}

/**
 * Handle API requests with intelligent caching
 */
async function handleApiRequest(request) {
    const url = new URL(request.url);
    
    // Check if this API endpoint should be cached
    if (!isCacheableApiRequest(url)) {
        return handleUncacheableApiRequest(request);
    }
    
    try {
        // Network-first strategy for API data
        const networkResponse = await fetch(request);
        
        if (networkResponse.ok) {
            // Cache successful responses
            const cache = await caches.open(DATA_CACHE_NAME);
            await cache.put(request, networkResponse.clone());
            
            // Add offline metadata
            const response = networkResponse.clone();
            addOfflineHeaders(response);
            
            return response;
        } else {
            throw new Error(`API request failed: ${networkResponse.status}`);
        }
    } catch (error) {
        console.warn('[ServiceWorker] Network request failed, trying cache:', error);
        
        // Fall back to cache
        const cachedResponse = await caches.match(request);
        if (cachedResponse) {
            // Add offline indicators
            const response = cachedResponse.clone();
            addOfflineHeaders(response);
            return response;
        }
        
        // Return offline response
        return createOfflineApiResponse(request);
    }
}

/**
 * Handle uncacheable API requests
 */
async function handleUncacheableApiRequest(request) {
    try {
        return await fetch(request);
    } catch (error) {
        console.error('[ServiceWorker] Uncacheable API request failed:', error);
        
        // For mutations, queue for background sync
        if (request.method !== 'GET') {
            await queueRequestForSync(request);
            return new Response(JSON.stringify({
                success: true,
                queued: true,
                message: 'Request queued for synchronization'
            }), {
                status: 202,
                headers: { 'Content-Type': 'application/json' }
            });
        }
        
        return new Response(JSON.stringify({
            error: 'Service temporarily unavailable',
            offline: true
        }), {
            status: 503,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}

/**
 * Handle navigation requests (app shell pattern)
 */
async function handleNavigationRequest(request) {
    try {
        const networkResponse = await fetch(request);
        return networkResponse;
    } catch (error) {
        console.warn('[ServiceWorker] Navigation request failed, serving app shell');
        
        // Serve app shell from cache
        const cachedResponse = await caches.match('/index.html');
        if (cachedResponse) {
            return cachedResponse;
        }
        
        // Fallback to offline page
        return caches.match('/static/offline.html');
    }
}

/**
 * Handle dynamic content requests
 */
async function handleDynamicRequest(request) {
    try {
        // Stale-while-revalidate strategy
        const cache = await caches.open(DYNAMIC_CACHE_NAME);
        const cachedResponse = await cache.match(request);
        
        const networkPromise = fetch(request).then(response => {
            if (response.ok) {
                cache.put(request, response.clone());
            }
            return response;
        });
        
        // Return cached response immediately if available
        if (cachedResponse) {
            // Update cache in background
            networkPromise.catch(() => {
                // Ignore network errors when using cached response
            });
            return cachedResponse;
        }
        
        // Wait for network response if no cache
        return await networkPromise;
    } catch (error) {
        console.error('[ServiceWorker] Dynamic request failed:', error);
        return new Response('Content unavailable offline', { status: 503 });
    }
}

/**
 * Handle data mutation requests (POST, PUT, DELETE)
 */
async function handleDataMutationRequest(request) {
    try {
        const response = await fetch(request);
        
        if (response.ok) {
            // Invalidate related cached data
            await invalidateRelatedCache(request);
        }
        
        return response;
    } catch (error) {
        console.warn('[ServiceWorker] Data mutation failed, queuing for sync:', error);
        
        // Queue request for background sync
        await queueRequestForSync(request);
        
        return new Response(JSON.stringify({
            success: true,
            queued: true,
            message: 'Changes saved locally and will sync when connection is restored'
        }), {
            status: 202,
            headers: { 
                'Content-Type': 'application/json',
                'X-Offline-Queued': 'true'
            }
        });
    }
}

/* =========================================================== */
/* Utility Functions                                          */
/* =========================================================== */

/**
 * Check if request is for static resources
 */
function isStaticResource(url) {
    const staticPatterns = [
        /\.(js|css|html|png|jpg|jpeg|gif|svg|ico|woff|woff2|ttf|eot)$/,
        /^.*\/resources\/.*$/,
        /^.*\/static\/.*$/
    ];
    
    return staticPatterns.some(pattern => pattern.test(url.pathname));
}

/**
 * Check if request is for API
 */
function isApiRequest(url) {
    return url.pathname.startsWith('/api/');
}

/**
 * Check if request is navigation request
 */
function isNavigationRequest(request) {
    return request.mode === 'navigate' || 
           (request.method === 'GET' && request.headers.get('accept').includes('text/html'));
}

/**
 * Check if API request should be cached
 */
function isCacheableApiRequest(url) {
    // Check against uncacheable patterns first
    if (UNCACHEABLE_API_PATTERNS.some(pattern => pattern.test(url.pathname + url.search))) {
        return false;
    }
    
    // Check against cacheable patterns
    return CACHEABLE_API_PATTERNS.some(pattern => pattern.test(url.pathname + url.search));
}

/**
 * Add offline indicators to response headers
 */
function addOfflineHeaders(response) {
    if (response.headers) {
        response.headers.set('X-Served-From-Cache', 'true');
        response.headers.set('X-Cache-Timestamp', new Date().toISOString());
    }
}

/**
 * Create offline API response
 */
function createOfflineApiResponse(request) {
    const url = new URL(request.url);
    
    // Determine entity type from URL
    let entityType = 'unknown';
    if (url.pathname.includes('/projects')) entityType = 'projects';
    else if (url.pathname.includes('/agents')) entityType = 'agents';
    else if (url.pathname.includes('/workflows')) entityType = 'workflows';
    
    return new Response(JSON.stringify({
        error: 'Service temporarily unavailable',
        offline: true,
        entityType,
        message: 'Data may be available in offline storage',
        cached: false
    }), {
        status: 503,
        headers: { 
            'Content-Type': 'application/json',
            'X-Offline-Response': 'true'
        }
    });
}

/**
 * Queue request for background synchronization
 */
async function queueRequestForSync(request) {
    try {
        // Store request details in IndexedDB
        const requestData = {
            url: request.url,
            method: request.method,
            headers: Object.fromEntries(request.headers.entries()),
            body: request.method !== 'GET' ? await request.text() : null,
            timestamp: Date.now()
        };
        
        // Open IndexedDB
        const db = await openIndexedDB();
        const transaction = db.transaction(['syncQueue'], 'readwrite');
        const objectStore = transaction.objectStore('syncQueue');
        
        await objectStore.add({
            id: generateId(),
            request: requestData,
            status: 'pending',
            timestamp: Date.now(),
            retries: 0
        });
        
        console.log('[ServiceWorker] Request queued for sync:', request.url);
    } catch (error) {
        console.error('[ServiceWorker] Failed to queue request for sync:', error);
    }
}

/**
 * Perform background synchronization
 */
async function performBackgroundSync() {
    try {
        console.log('[ServiceWorker] Performing background sync...');
        
        const db = await openIndexedDB();
        const transaction = db.transaction(['syncQueue'], 'readwrite');
        const objectStore = transaction.objectStore('syncQueue');
        
        const pendingRequests = await objectStore.index('status').getAll('pending');
        
        for (const queueItem of pendingRequests) {
            try {
                const { request: requestData } = queueItem;
                
                // Reconstruct request
                const request = new Request(requestData.url, {
                    method: requestData.method,
                    headers: requestData.headers,
                    body: requestData.body
                });
                
                // Attempt to sync
                const response = await fetch(request);
                
                if (response.ok) {
                    // Remove from queue
                    await objectStore.delete(queueItem.id);
                    console.log('[ServiceWorker] Sync successful for:', requestData.url);
                } else {
                    throw new Error(`Sync failed: ${response.status}`);
                }
            } catch (error) {
                console.error('[ServiceWorker] Sync failed for request:', queueItem.request.url, error);
                
                // Update retry count
                queueItem.retries = (queueItem.retries || 0) + 1;
                
                if (queueItem.retries >= 5) {
                    // Mark as failed after 5 retries
                    queueItem.status = 'failed';
                } else {
                    // Keep as pending for retry
                    queueItem.lastAttempt = Date.now();
                }
                
                await objectStore.put(queueItem);
            }
        }
        
        console.log('[ServiceWorker] Background sync completed');
    } catch (error) {
        console.error('[ServiceWorker] Background sync failed:', error);
    }
}

/**
 * Invalidate related cache entries
 */
async function invalidateRelatedCache(request) {
    const url = new URL(request.url);
    
    // Determine what cache entries to invalidate based on the request
    let patternsToInvalidate = [];
    
    if (url.pathname.includes('/projects')) {
        patternsToInvalidate = ['/api/v1/projects'];
    } else if (url.pathname.includes('/agents')) {
        patternsToInvalidate = ['/api/v1/agents', '/api/v1/projects'];
    } else if (url.pathname.includes('/workflows')) {
        patternsToInvalidate = ['/api/v1/workflows', '/api/v1/projects'];
    }
    
    // Clear matching cache entries
    const cache = await caches.open(DATA_CACHE_NAME);
    const requests = await cache.keys();
    
    for (const cachedRequest of requests) {
        const cachedUrl = new URL(cachedRequest.url);
        if (patternsToInvalidate.some(pattern => cachedUrl.pathname.startsWith(pattern))) {
            await cache.delete(cachedRequest);
        }
    }
}

/**
 * Open IndexedDB connection
 */
function openIndexedDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open('A2APortalOffline', 3);
        
        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result);
        
        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            
            if (!db.objectStoreNames.contains('syncQueue')) {
                const objectStore = db.createObjectStore('syncQueue', { keyPath: 'id' });
                objectStore.createIndex('status', 'status');
                objectStore.createIndex('timestamp', 'timestamp');
            }
        };
    });
}

/**
 * Generate unique ID
 */
function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2, 9);
}

/* =========================================================== */
/* Cache Management                                           */
/* =========================================================== */

/**
 * Clean up old cache entries
 */
async function cleanupOldCaches() {
    const cacheNames = await caches.keys();
    
    for (const cacheName of cacheNames) {
        const cache = await caches.open(cacheName);
        const requests = await cache.keys();
        
        // Get cache configuration
        let config = CACHE_CONFIG.static;
        if (cacheName === DATA_CACHE_NAME) config = CACHE_CONFIG.data;
        else if (cacheName === DYNAMIC_CACHE_NAME) config = CACHE_CONFIG.dynamic;
        
        // Remove expired entries
        for (const request of requests) {
            const response = await cache.match(request);
            const cacheTime = response.headers.get('X-Cache-Timestamp');
            
            if (cacheTime) {
                const age = Date.now() - new Date(cacheTime).getTime();
                if (age > config.maxAge) {
                    await cache.delete(request);
                }
            }
        }
        
        // Limit cache size
        const remainingRequests = await cache.keys();
        if (remainingRequests.length > config.maxItems) {
            const excessCount = remainingRequests.length - config.maxItems;
            for (let i = 0; i < excessCount; i++) {
                await cache.delete(remainingRequests[i]);
            }
        }
    }
}

// Periodic cache cleanup
setInterval(cleanupOldCaches, 60 * 60 * 1000); // Every hour