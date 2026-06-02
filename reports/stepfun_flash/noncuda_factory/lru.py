import threading


class LRUCache:
    """Thread-safe Least-Recently-Used cache with a fixed capacity."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._lock = threading.Lock()
        # Doubly-linked list nodes stored in a dict: key -> (prev, next, value)
        self._head = None   # most-recently-used sentinel
        self._tail = None   # least-recently-used sentinel
        self._map = {}      # key -> [prev, next, value]

    # ------------------------------------------------------------------ #
    # internal helpers (must be called with self._lock held)              #
    # ------------------------------------------------------------------ #

    def _add_to_front(self, key):
        """Insert key as most-recently-used."""
        if self._head is None:
            self._head = key
            self._tail = key
            self._map[key] = [None, None, self._map[key][2]]
        else:
            old_head = self._head
            self._map[key][0] = old_head
            self._map[key][1] = None
            self._map[old_head][1] = key
            self._head = key

    def _remove_node(self, key):
        """Unlink key from the doubly-linked list, reconnecting neighbours."""
        prev, nxt, _ = self._map[key]
        if prev is not None:
            self._map[prev][1] = nxt
        else:
            self._head = nxt
        if nxt is not None:
            self._map[nxt][0] = prev
        else:
            self._tail = prev

    def _evict_lru(self):
        """Remove the least-recently-used key."""
        if self._tail is None:
            return
        key = self._tail
        self._remove_node(key)
        del self._map[key]

    def _touch(self, key):
        """Move key to front (most-recently-used)."""
        if self._head == key:
            return
        self._remove_node(key)
        self._add_to_front(key)

    # ------------------------------------------------------------------ #
    # public API                                                           #
    # ------------------------------------------------------------------ #

    def get(self, key):
        with self._lock:
            if key not in self._map:
                return None
            self._touch(key)
            return self._map[key][2]

    def put(self, key, value):
        with self._lock:
            if key in self._map:
                self._map[key][2] = value
                self._touch(key)
                return
            # new entry
            self._map[key] = [None, None, value]
            self._add_to_front(key)
            if len(self._map) > self._capacity:
                self._evict_lru()

    def __len__(self):
        with self._lock:
            return len(self._map)

    def __contains__(self, key):
        with self._lock:
            return key in self._map
