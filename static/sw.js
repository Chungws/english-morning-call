const CACHE = 'morning-call-v1';
const ASSETS = ['/', '/static/index.html', '/static/icon-192.png', '/static/icon-512.png'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)));
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys => 
    Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
  ));
  self.clients.claim();
});

self.addEventListener('fetch', e => {
  // Network first for API/WS, cache first for static
  if (e.request.url.includes('/api/') || e.request.url.includes('/ws')) return;
  e.respondWith(
    fetch(e.request).catch(() => caches.match(e.request))
  );
});
