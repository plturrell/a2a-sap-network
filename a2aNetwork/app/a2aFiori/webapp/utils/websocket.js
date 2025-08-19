sap.ui.define([
    "sap/base/Log"
], function(Log) {
    "use strict";

    /**
     * WebSocket utility for real-time communication
     * Handles connection management, authentication, and event handling
     */
    return {
        _socket: null,
        _reconnectAttempts: 0,
        _maxReconnectAttempts: 5,
        _reconnectDelay: 1000,
        _eventHandlers: new Map(),

        /**
         * Initialize WebSocket connection
         * @param {Object} config - Configuration object
         * @param {string} config.token - Authentication token
         * @param {string} [config.url] - WebSocket server URL
         * @returns {Promise} Promise that resolves when connected
         */
        connect(config) {
            return new Promise((resolve, reject) => {
                if (this._socket && this._socket.connected) {
                    Log.info("WebSocket already connected");
                    return resolve(this._socket);
                }

                // Determine WebSocket URL
                const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
                const host = window.location.host;
                const wsUrl = config.url || `${protocol}//${host}`;

                // Import socket.io client library dynamically
                if (typeof io === "undefined") {
                    // Load socket.io client from CDN if not available
                    const script = document.createElement("script");
                    script.src = "https://cdn.socket.io/4.7.4/socket.io.min.js";
                    script.onload = () => {
                        this._createConnection(wsUrl, config, resolve, reject);
                    };
                    script.onerror = () => {
                        reject(new Error("Failed to load socket.io client library"));
                    };
                    document.head.appendChild(script);
                } else {
                    this._createConnection(wsUrl, config, resolve, reject);
                }
            });
        },

        /**
         * Create WebSocket connection
         * @private
         */
        _createConnection(wsUrl, config, resolve, reject) {
            try {
                this._socket = io(wsUrl, {
                    auth: {
                        token: config.token
                    },
                    transports: ["websocket", "polling"],
                    reconnection: true,
                    reconnectionAttempts: this._maxReconnectAttempts,
                    reconnectionDelay: this._reconnectDelay
                });

                this._setupEventHandlers(resolve, reject);
                Log.info("WebSocket connection initiated", { url: wsUrl });
            } catch (error) {
                Log.error("Failed to create WebSocket connection", error);
                reject(error);
            }
        },

        /**
         * Setup WebSocket event handlers
         * @private
         */
        _setupEventHandlers(resolve, reject) {
            const socket = this._socket;

            socket.on("connect", () => {
                Log.info("WebSocket connected", { socketId: socket.id });
                this._reconnectAttempts = 0;
                resolve(socket);
            });

            socket.on("connected", (data) => {
                Log.info("WebSocket connection confirmed", data);
            });

            socket.on("disconnect", (reason) => {
                Log.warning("WebSocket disconnected", { reason });
            });

            socket.on("connect_error", (error) => {
                Log.error("WebSocket connection error", error);
                this._reconnectAttempts++;

                if (this._reconnectAttempts >= this._maxReconnectAttempts) {
                    reject(new Error(`WebSocket connection failed after ${this._maxReconnectAttempts} attempts: ${error.message}`));
                }
            });

            socket.on("error", (error) => {
                Log.error("WebSocket error", error);
            });

            socket.on("subscribed", (data) => {
                Log.info("Subscribed to topics", data);
            });

            socket.on("unsubscribed", (data) => {
                Log.info("Unsubscribed from topics", data);
            });
        },

        /**
         * Subscribe to topics for real-time updates
         * @param {Array} topics - Array of topic names
         * @returns {Promise} Promise that resolves when subscribed
         */
        subscribe(topics) {
            return new Promise((resolve, reject) => {
                if (!this._socket || !this._socket.connected) {
                    return reject(new Error("WebSocket not connected"));
                }

                if (!Array.isArray(topics)) {
                    return reject(new Error("Topics must be an array"));
                }

                this._socket.emit("subscribe", topics);

                // Wait for subscription confirmation
                const timeout = setTimeout(() => {
                    reject(new Error("Subscription timeout"));
                }, 5000);

                this._socket.once("subscribed", (data) => {
                    clearTimeout(timeout);
                    resolve(data);
                });

                this._socket.once("error", (error) => {
                    clearTimeout(timeout);
                    reject(error);
                });
            });
        },

        /**
         * Unsubscribe from topics
         * @param {Array} topics - Array of topic names
         * @returns {Promise} Promise that resolves when unsubscribed
         */
        unsubscribe(topics) {
            return new Promise((resolve, reject) => {
                if (!this._socket || !this._socket.connected) {
                    return reject(new Error("WebSocket not connected"));
                }

                if (!Array.isArray(topics)) {
                    return reject(new Error("Topics must be an array"));
                }

                this._socket.emit("unsubscribe", topics);

                const timeout = setTimeout(() => {
                    reject(new Error("Unsubscription timeout"));
                }, 5000);

                this._socket.once("unsubscribed", (data) => {
                    clearTimeout(timeout);
                    resolve(data);
                });
            });
        },

        /**
         * Listen for events on specific topics
         * @param {string} event - Event name
         * @param {Function} handler - Event handler function
         */
        on(event, handler) {
            if (!this._socket) {
                Log.warning("WebSocket not initialized, handler will be queued");
                this._eventHandlers.set(event, handler);
                return;
            }

            this._socket.on(event, handler);
            Log.debug("Event handler registered", { event });
        },

        /**
         * Remove event listener
         * @param {string} event - Event name
         * @param {Function} [handler] - Specific handler to remove
         */
        off(event, handler) {
            if (this._socket) {
                this._socket.off(event, handler);
            }
            this._eventHandlers.delete(event);
        },

        /**
         * Disconnect WebSocket
         */
        disconnect() {
            if (this._socket) {
                this._socket.disconnect();
                this._socket = null;
                Log.info("WebSocket disconnected");
            }
        },

        /**
         * Check if WebSocket is connected
         * @returns {boolean} Connection status
         */
        isConnected() {
            return this._socket && this._socket.connected;
        },

        /**
         * Get connection info
         * @returns {Object} Connection information
         */
        getConnectionInfo() {
            if (!this._socket) {
                return { connected: false };
            }

            return {
                connected: this._socket.connected,
                socketId: this._socket.id,
                transport: this._socket.io.engine.transport.name,
                reconnectAttempts: this._reconnectAttempts
            };
        }
    };
});